import pandas as pd
from scipy import spatial
import numpy as np
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
from tensorflow.keras.preprocessing.image import load_img
import os

class Helper():
    def __init__(self, text_processor):
        self.data_dir = "/hate/Dataset"
        self.img_path = self.data_dir + "/img_without_text/"
        self.train_path = self.data_dir + "/train_sr.jsonl"
        self.dev_path = self.data_dir + "/dev.jsonl"
        self.test_path = self.data_dir + "/test.jsonl"
        #self.train_without_confounders_img = self.data_dir + "train_without_confounders_img.jsonl"
        self.data_train = pd.read_json(self.train_path, lines=True)
        self.data_test = pd.read_json(self.test_path, lines=True)
        self.data_dev = pd.read_json(self.dev_path, lines=True)
        #self.data_train_w_c_img = pd.read_json(self.train_without_confounders_img, lines=True)
        self.sentences_train = self.data_train['text'].to_list()
        self.sentences_test = self.data_test['text'].to_list()
        self.sentences_dev = self.data_dev['text'].to_list()

        self.text_processing = text_processor


    def get_texts_of_json(self, json_name):
        if json_name == 'train':
            return self.sentences_train
        if json_name == 'dev':
            return self.sentences_dev
        if json_name == 'test':
            return self.sentences_test


    def get_names_of_json(self, json_name):
        if json_name == 'train':
            json = self.data_train
        if json_name == 'dev':
            json = self.data_dev
        if json_name == 'test':
            json = self.data_test
        if json_name == 'train_without_confounders_img':
            json = self.data_train_w_c_img
        list_aux = json['img'].to_list()
        list_of_names = list()
        for name in list_aux:
            list_of_names.append(name[4:])
        return (list_of_names)

    def prepare_image_text_info_by_clusters(self, process, clusters, imgp):
        image_info = []
        for num_cluster in clusters:
            filenames = process.get_filenames_by_cluster(num_cluster)
            for text in filenames:
                row = self.data_train.loc[self.data_train['img'] == 'img/'+text]
                label = row['label'].values[0]
                text = row['text'].values[0]
                image = row['img'].values[0]
                image = image[4:]
                img_vector = imgp.prepare_image(self.data_dir + '/img/' + image)
                image_info.append(
                    {'n_cluster': num_cluster, 'img': image, 'label': label, 'img_vector': img_vector,
                     'text': text})
        return image_info

    def get_distances(self, images_info, text_encoded):
        # images info es resultado del metodo prepare_image_text_info_by_clusters
        sentences_distances = []
        for i in range(0, len(images_info)):
            dist = {'image': images_info[i]['img'], 'cluster': images_info[i]['n_cluster'],
                    'label': images_info[i]['hate'],
                    'distance': abs(1 - spatial.distance.cosine(images_info[i]['img_vector'], text_encoded))}
            sentences_distances.append(dist)
        return sorted(sentences_distances, key=lambda x: x['distance'], reverse=True)

    def get_accuracy_text_image(self, clustering, top_n_clusters, num_clusters, imgp):
        success = 0
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        predicciones = {}
        for i in range(0, len(self.data_dev.index)):
            print("Número de iteración:")
            print(i)
            text_prueba = self.data_dev.iloc[i]
            name_img = text_prueba['img']
            name_img = name_img[4:]
            prep_text= self.text_processing.encode_sentences([text_prueba['text']])
            # predecimos clusters mas similares
            #print('Text prueba  codificado')
            #print(prep_text)
            #print(len(prep_text))

            #pred = clustering.get_top_n_similar_clusters(top_n_clusters, prep_text, num_clusters)
            pred = clustering.predict(prep_text, num_clusters)

            #creo vector solo con los numeros del cluster
            top_clusters = []
            for i in range(len(pred)):
                top_clusters.append(pred[i]['cluster'])

            #top_clusters = clustering.predict([prep_text], num_clusters)
            print("top clusters", top_clusters)

            #predicciones.append({"img": name_img, "prediccion":top_clusters[0]})
            if top_clusters[0] in predicciones:
                new_val = predicciones.get(top_clusters[0])+1
                predicciones.update({top_clusters[0]:new_val})
            else:
                predicciones[top_clusters[0]]=1


            # codifico el texto de cada imagen en los top n clusters mas similares
            sentences_top_n = self.prepare_image_text_info_by_clusters(clustering, top_clusters,imgp)
            img_vector = imgp.prepare_image(self.img_path + name_img)
            img_distances = self.get_distances(sentences_top_n, img_vector)

            print('Image: ')
            print(name_img)
            print('Image label:')
            print(text_prueba['label'])
            #print("Calculamos la distancia contra ", len(img_distances), "elementos del cluster nro. ", pred)
            #print("sentences top n : ",len(sentences_top_n))


            print(img_distances)
            print("--------------------------------------------")

            hate = img_distances[0]['label']

            if text_prueba['label'] == hate:
                success += 1
                if hate:
                    tp += 1
                else:
                    tn += 1
            else:
                if hate:
                    fn += 1
                else:
                    fp += 1

        metrics = {'accuracy': success / len(self.data_dev.index), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "predicciones": predicciones}
        # 'precision':tp/(tp+fp), 'recall':tp/(tp+fn)}
        return metrics


    def get_accuracy_image_text(self, images, top_n_clusters, num_clusters, clustering):
        success = 0
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in range(0, len(self.data_dev.index)):
            print("Número de iteración:")
            print(i)
            img_prueba = self.data_dev.iloc[i]
            name_img = img_prueba['img']
            name_img = name_img[4:]
            prep_im = images.prepare_image(self.data_dir + '/img/' + name_img)
            # predecimos clusters mas similares
            pred = clustering.get_top_n_similar_clusters(top_n_clusters, prep_im, num_clusters)

            # creo vector solo con los numeros del cluster
            top_clusters = []
            for i in range(len(pred)):
                top_clusters.append(pred[i]['cluster'])

            # codifico el texto de cada imagen en los top n clusters mas similares
            sentences_top_n = self.prepare_image_text_info_by_clusters(clustering, top_clusters, images)
            print(sentences_top_n)
            text_distances = self.text_processing.get_similarity(sentences_top_n, img_prueba['text'])

            print('Image: ')
            print(name_img)
            print('Image label:')
            print(img_prueba['label'])
            print(text_distances)
            print("--------------------------------------------")

            hate = text_distances[0]['label']

            if img_prueba['label'] == hate:
                success += 1
                if hate:
                    tp += 1
                else:
                    tn += 1
            else:
                if hate:
                    fn += 1
                else:
                    fp += 1

        metrics = {'accuracy': success / len(self.data_dev.index), "tp": tp, "tn": tn, "fp": fp, "fn": fn}
        # 'precision':tp/(tp+fp), 'recall':tp/(tp+fn)}
        return metrics

    def remove_confounders_images(self):
        data = self.data_train
        data = data.assign(mark = 0)
        out = []
        names = self.get_names_of_json("train")
        for i in range(0,len(data)):
            img_prueba = data.iloc[i]
            name_img = img_prueba['img']
            name_img = name_img[4:]
            print("iteracion:", i)
            print("imagen:", name_img)
            print("marca: ", img_prueba['mark'])
            if img_prueba['mark'] != 1:
                img = load_img(self.img_path+name_img, target_size=(224,224))
                image = np.array(img)
                similars = self.get_similar_images(image, names, self.img_path)
                out.append({"name":name_img, "similars": similars})
                print("similars: ", similars)
                for m in similars:
                    data.loc[data['img'] == "img/"+m, ["mark"]] = 1
                    names.remove(m)
                print("ceros: ", data.loc[data['mark'] == 1])
                print("OUT: ", out)

        return out

    def get_similar_images(self, image, names, path):
        similarity_output = []
        image = image.reshape(1,224,224,3)
        with os.scandir(path) as files:
      # loops through each file in the directory
         for file in files:
              if file.name.endswith('.png') :
                if names and file.name in names:# load the image as a 224x224 array
                  img = load_img(path+file.name, target_size=(224,224))
                  img = np.array(img)
                  reshaped_img = img.reshape(1,224,224,3)
                  valor = mse(image,reshaped_img)
                  if valor < 1000:
                    similarity_output.append(file.name)
        return similarity_output