# -*- coding: utf-8 -*-
"""Copy of Modelo Octubre 22.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vqqBry80e8wWL9Dr_3vv7dC7-ewmyqQI
"""

# Commented out IPython magic to ensure Python compatibility.
# 1

# %load_ext autoreload
# %autoreload 2

# 2
import sentence_transformers
import json
import numpy as np
from sklearn.cluster import SpectralClustering, DBSCAN, BisectingKMeans, KMeans
from sklearn.metrics import adjusted_mutual_info_score
import os
from PIL import Image
from torchvision import transforms
import torch
from torch import nn
from torch import Tensor
from tqdm.auto import tqdm
from sklearn.metrics import classification_report

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ResNetExtract(nn.Module):
    def __init__(
            self,
            resnet):
        super().__init__()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        pass

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)[:, :, 0, 0]

        return x


# clasificador
class MoE(nn.Module):

    def __init__(self, dims, clusters):
        super().__init__()
        self.l1 = nn.Linear(dims, 100)
        self.l2 = nn.ModuleList()
        for _ in range(clusters):
            self.l2.append(nn.Linear(100, 10))
        self.l3 = nn.Linear(10, 1)

    def forward(self, x, cluster, routed=True):
        x = self.l1(x)
        x = nn.functional.leaky_relu(x)
        all_x = []
        for m in self.l2:
            x1 = m(x)
            x1 = nn.functional.leaky_relu(x1)
            x1 = torch.unsqueeze(x1, dim=1)
            all_x.append(x1)
        all_x = torch.concat(all_x, dim=1)
        if routed:
            x = all_x[torch.arange(x.shape[0]).to(device), cluster, :]
        else:
            # x = torch.max(all_x, dim=1)[0]
            x = torch.mean(all_x, dim=1)
        x = self.l3(x)
        return torch.sigmoid(x)[:, 0]


class Procedimiento:

    def __init__(self, MODELO_TEXTO):
        with open('/Users/magaliboulanger/Documents/Dataset/train_sr_final.jsonl', 'rt', encoding='utf-8') as f:
            self.train = [json.loads(l) for l in f]

        with open('/Users/magaliboulanger/Documents/Dataset/test.jsonl', 'rt', encoding='utf-8') as f:
            self.test = [json.loads(l) for l in f]

        with open('/Users/magaliboulanger/Documents/Dataset/dev_sr_final.jsonl', 'rt', encoding='utf-8') as f:
            self.dev = [json.loads(l) for l in f]
        self.modelo_texto=MODELO_TEXTO
        if MODELO_TEXTO=='A':
            self.s_file = '/Users/magaliboulanger/Documents/Dataset/text_sr_all-miniLM-L6-V2.npz'
        if MODELO_TEXTO=='B':
            self.s_file = '/Users/magaliboulanger/Documents/Dataset/text_sr.npz'
        self.i_file = '/Users/magaliboulanger/Documents/Dataset/img_sr.npz'

    def encode_images(self, model, dataset, preprocess, device, batch_size=100):
        res = []
        with tqdm(total=len(dataset)) as pbar:
            batch = []
            for instance in dataset:
                if len(batch) == batch_size:
                    with torch.no_grad():
                        b = torch.concat(batch, dim=0).to(device)
                        res.append(model(b).cpu().numpy())
                        pbar.update(len(batch))
                        batch.clear()
                file = instance['img'].split('/')[1]
                img = Image.open(f'/Users/magaliboulanger/Documents/Dataset/img/{file}').convert('RGB')
                img = preprocess(img).unsqueeze(0)
                batch.append(img)
            if len(batch) == batch_size:
                with torch.no_grad():
                    b = torch.concat(batch, dim=0).to(device)
                    res.append(model(b).cpu().numpy())
                    pbar.update(len(batch))
                    batch.clear()
        return np.concatenate(res, axis=0)

    def generate_embeddings_text(self):
        if self.modelo_texto == 'B':
            st = sentence_transformers.SentenceTransformer('bert-base-uncased')
        if self.modelo_texto == 'A':
            st = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        if not os.path.exists(self.s_file):
            train_text = st.encode([t['text'] for t in self.train], show_progress_bar=True)
            test_text = st.encode([t['text'] for t in self.test], show_progress_bar=True)
            dev_text = st.encode([t['text'] for t in self.dev], show_progress_bar=True)
            np.savez_compressed(self.s_file, train=train_text, test=test_text, dev=dev_text)
        else:
            f = np.load(self.s_file)
            train_text = f['train']
            test_text = f['test']
            dev_text = f['dev']
            del f
        return train_text, dev_text, test_text

    def extract_features_images(self):
        if not os.path.exists(self.i_file):
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            model = ResNetExtract(model)
            print(f'Device: {device}')
            model.to(device)
            train_img = self.encode_images(model, self.train, preprocess, device, 191)
            train_img = train_img / np.linalg.norm(train_img, axis=1, keepdims=True)
            test_img = self.encode_images(model, self.test, preprocess, device)
            test_img = test_img / np.linalg.norm(test_img, axis=1, keepdims=True)
            dev_img = self.encode_images(model, self.dev, preprocess, device, 179)
            dev_img = dev_img / np.linalg.norm(dev_img, axis=1, keepdims=True)
            np.savez_compressed(self.i_file, train=train_img, test=test_img, dev=dev_img)
            del model
        else:
            f = np.load(self.i_file)
            train_img = f['train']
            test_img = f['test']
            dev_img = f['dev']
            del f
        return train_img, dev_img, test_img


    def proceder(self, threshold, n_clusters, clustering_model, routed):
        # clustering
        train_img, dev_img, test_img = self.extract_features_images()
        train_text, dev_text, test_text = self.generate_embeddings_text()
        if clustering_model=='B':
            cluster = BisectingKMeans(n_clusters, random_state=42)
        if clustering_model=='K':
            cluster = KMeans(n_clusters, random_state=42)
        train_i_c = cluster.fit_predict(train_img)
        adjusted_mutual_info_score([t['label'] for t in self.train], train_i_c)
        dev_i_c = cluster.predict(dev_img)

        x = np.concatenate((train_img, train_text), axis=1)
        y = np.asarray([t['label'] for t in self.train])

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = MoE(x.shape[1], n_clusters)
        x1 = torch.from_numpy(x)
        mean = torch.mean(x1, axis=0)
        std = torch.std(x1, axis=0)
        x1 = (x1 - mean) / (std)
        x_std = torch.std(x1, dim=0).to(device)
        c = torch.from_numpy(train_i_c).type(torch.LongTensor)
        y1 = torch.from_numpy(y)

        ds = torch.utils.data.dataset.TensorDataset(x1, c, y1)
        dl = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=True)

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters())
        loss = nn.BCELoss()
        losses = []
        pbar = tqdm(range(100))
        for i in pbar:
            losses = []
            for x_i, c_i, y_i in dl:
                x_i = x_i.to(device)
                # rand = 2 * (torch.rand(x_i.shape) - 0.5).to(device) * x_std
                # x_i = rand + x_i
                c_i = c_i.to(device)
                y_i = y_i.to(device).type(torch.float32)
                optimizer.zero_grad()
                pred = model(x_i, c_i)
                l = loss(pred, y_i)
                losses.append(l.item())
                l.backward()
                optimizer.step()
            pbar.set_description(f'Loss {np.mean(losses)}')
        pbar.close()

        x_test = x = np.concatenate((dev_img, dev_text), axis=1)
        y_test = np.asarray([t['label'] for t in self.dev])

        x_test1 = torch.from_numpy(x_test)
        x_test1 = (x_test1 - mean) / (std)
        c_test = torch.from_numpy(dev_i_c).type(torch.LongTensor)
        y_pred = []
        with torch.no_grad():
            for i in range(0, x_test1.shape[0], 100):
                x_i = x_test1[i:i + 100, ...].to(device)
                c_i = c_test[i:i + 100].to(device)
                y_pred.append(model(x_i, c_i, routed).cpu().numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        return y_pred, classification_report(y_test, y_pred > threshold, output_dict=True)

