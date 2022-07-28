from sklearn import cluster
import json
import pandas as pd
from text_processing import TextProcessing, InferSentProcessing
from scipy import spatial


class Helper():
    def __init__(self, text_processor):
        self.data_dir = "/hate/Dataset"
        self.img_path = self.data_dir + "/img_without_text/"
        self.train_path = self.data_dir + "/train.jsonl"
        self.dev_path = self.data_dir + "/dev.jsonl"
        self.test_path = self.data_dir + "/test.jsonl"

        self.data_train = pd.read_json(self.train_path, lines=True)
        self.data_test = pd.read_json(self.test_path, lines=True)
        self.data_dev = pd.read_json(self.dev_path, lines=True)
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
        list_aux = json['img'].to_list()
        list_of_names = list()
        for name in list_aux:
            list_of_names.append(name[4:])
        return (list_of_names)

    def prepare_image_text_info_by_clusters(self, images, clusters, imgp):
        image_info = []
        for num_cluster in clusters:
            filenames = images.get_filenames_by_cluster(num_cluster)
            for text in filenames:
                row = self.data_train.loc[self.data_train['img'] == 'img/'+text]
                label = row['label'].values[0]
                text = row['text'].values[0]
                image = row['img'].values[0]
                image = image[4:]
                img_vector = imgp.prepare_image(self.data_dir + '/img_without_text/' + image)
                image_info.append(
                    {'n_cluster': num_cluster, 'img': image, 'hate': label, 'img_vector': img_vector,
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

    def get_accuracy(self, clustering, top_n_clusters, num_clusters, imgp):
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

            pred = clustering.get_top_n_similar_clusters(top_n_clusters, prep_text, num_clusters)
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


