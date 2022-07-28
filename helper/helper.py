from sklearn import cluster
import json
import pandas as pd
from text_processing.text_processing import TextProcessing

class Helper():
  def __init__(self):
    self.data_dir = "/hate/Dataset"
    self.train_path = self.data_dir + "/train.jsonl"
    self.dev_path = self.data_dir + "/dev.jsonl"
    self.test_path = self.data_dir + "/test.jsonl"

    self.data_train = pd.read_json(self.train_path, lines = True)
    self.data_test = pd.read_json(self.test_path, lines = True)
    self.data_dev = pd.read_json(self.dev_path, lines=True)
    self.sentences_train = self.data_train['text'].to_list()
    self.sentences_test = self.data_test['text'].to_list()

    self.text_processing = TextProcessing()

  def prepare_image_text_info_by_clusters(self, images, clusters):
    image_info = []
    for num_cluster in clusters:
      filenames = images.get_filenames_by_cluster(num_cluster)
      for image in filenames:
        row = self.data_train.loc[self.data_train['img']=='img/'+image]
        label = row['label'].values[0]
        text = row['text'].values[0]
        image_info.append({'n_cluster': num_cluster, 'img': image, 'label': label, 'text': text})
    return image_info


  def get_names_of_json(self,json_name):
    if json_name=='train':
      json=self.data_train
    if json_name=='dev':
      json=self.data_dev
    if json_name=='test':
      json=self.data_test        
    list_aux = json['img'].to_list()
    list_of_names= list()
    for name in list_aux:
      list_of_names.append(name[4:])
    return(list_of_names)
  
  def get_accuracy(self, images, top_n_clusters, num_clusters):
    success = 0 
    tp= 0
    fp =0
    fn =0
    tn =0
    for i in range (0,len(self.data_dev.index)): 
      print("Número de iteración:")
      print(i)
      img_prueba = self.data_dev.iloc[i]
      name_img = img_prueba['img']
      name_img=name_img[4:]
      prep_im = images.prepare_image(self.data_dir + '/img/' + name_img)
      #predecimos clusters mas similares
      pred = images.get_top_n_similar_clusters(top_n_clusters, prep_im, num_clusters)

      #creo vector solo con los numeros del cluster
      top_clusters = []
      for i in range(len(pred)):
        top_clusters.append(pred[i]['cluster'])

      #codifico el texto de cada imagen en los top n clusters mas similares
      sentences_top_n = self.prepare_image_text_info_by_clusters(images, top_clusters)
      text_distances = self.text_processing.get_similarity(sentences_top_n, img_prueba['text'])

      print('Image: ')
      print(name_img)
      print('Image label:')
      print(img_prueba['label'])
      print(text_distances)
      print("--------------------------------------------")

      hate = text_distances[0]['label']
      
      if img_prueba['label'] == hate :
        success+=1
        if hate:
          tp+=1
        else:
          tn+=1
      else :
        if hate:
          fn+=1
        else:
          fp+=1
    
    
    metrics = {'accuracy':success/len(self.data_dev.index), "tp": tp, "tn": tn, "fp":fp, "fn":fn}
               #'precision':tp/(tp+fp), 'recall':tp/(tp+fn)}
    return metrics
  
 
