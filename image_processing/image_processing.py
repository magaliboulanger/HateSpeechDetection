from operator import mod
from numpy.ma.core import array
# for loading/processing the images 
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.applications.vgg16 import preprocess_input 

# models 
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.models import Model
#import resnet

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
from scipy import spatial

class ImageProcessing():
  def __init__(self,path,dataset_names):
    self.path=path
    os.chdir(path)
    self.model = VGG16()
    self.model = Model(inputs = self.model.inputs, outputs = self.model.layers[-2].output)
    self.kmeans = None

# this list holds all the image filename
    self.memes = []

# creates a ScandirIterator aliased as files
    with os.scandir(self.path) as files:
  # loops through each file in the directory
     for file in files:
          if file.name.endswith('.png') :
            if dataset_names and file.name in dataset_names:
          # adds only the image files to the memes list
              self.memes.append(file.name)
    print (len(self.memes))

  def prepare_image(self, img ):
    feat = self.extract_features(img)
    return self.pca.transform(feat)



  def extract_features(self,file):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = self.model.predict(imgx, use_multiprocessing=True)
    return features
  
  def prepare_model(self):
    data = {}
    p = r"error.log"

    # lop through each image in the datase
    for meme in self.memes:
        # try to extract the features and update the dictionary
        try:
            feat = self.extract_features(meme)
            data[meme] = feat
        # if something fails, save the extracted features as a pickle file (optional)
        except:
            print("Exception")
            with open(p,'wb') as file:
                pickle.dump(data,file)
              
    
    # get a list of the filenames

    self.filenames = np.array(list(data.keys()))
    print("filenames")
    # get a list of just the features
    feat = np.array(list(data.values()))
    print("features")
    feat.shape
    (210, 1, 4096)

    # reshape so that there are 210 samples of 4096 vectors
    feat = feat.reshape(-1,4096)
    feat.shape
    (210, 4096)
    
    self.pca = PCA(n_components=10, random_state=22)
    self.pca.fit(feat)
    self.x = self.pca.transform(feat)


  def group(self,n_clusters):
    self.prepare_model()
    self.kmeans = KMeans(n_clusters=n_clusters, random_state=22)
    self.kmeans.fit(self.x)
    self.groups = {}
    for file, cluster in zip(self.filenames,self.kmeans.labels_):
        if cluster not in self.groups.keys():
            self.groups[cluster] = []
        self.groups[cluster].append(file)
    
  def view_cluster(self,cluster):
    plt.figure(figsize = (25,25));
      # gets the list of filenames for a cluster
    files = self.groups[cluster]
      # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
      # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

  def k_optimo_codo(self, max_k):
    # metodo del codo
    self.prepare_model()
    sse = []
    list_k = list(range(3, max_k))

    for k in list_k:
        km = KMeans(n_clusters=k, random_state=22)
        km.fit(self.x)
        
        sse.append(km.inertia_)

    # Plot sse against k
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse)
    plt.xlabel('Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    
  def k_optimo_silueta(self,max_k):
    #metodo de la silueta
    self.prepare_model()
    silueta = []
    from sklearn.metrics import silhouette_score
    K= range(2,max_k)
    for k in K :
      kmeans = KMeans(n_clusters=k).fit(self.x)
      labels = kmeans.labels_ 
      #calcular el puntaje de la silueta promedio
      silueta.append(silhouette_score(self.x,labels,metric = 'euclidean'))

    #graficar los valores obtenidos
    plt.plot(K,silueta, 'bx-')
    plt.xlabel('Clusteres')
    plt.ylabel('Puntaje de la Silueta')
    plt.title('Metodo de la Silueta')
    plt.show()

    #El valor de k viene dado por el valor mas alto de la grafica (maximo global)

    
  def predict(self, x, n_clusters):
    #predice cuál es el cluster al cual x pertenece 
    if self.kmeans is None:
      self.group(n_clusters)

    return self.kmeans.predict(x)



#saves y loads. TO DO: refactorizar
  def save_kmeans(self):
    #lo guardo en el arch kmeans.pkl. w=write. b=binary.
    pickle.dump(self.kmeans, open("kmeans.pkl", "wb"))

  def save_pca(self):
    pickle.dump(self.pca, open("pca.pkl", "wb"))

  def save_groups(self):
    pickle.dump(self.groups, open("groups.pkl", "wb"))

  def load_kmeans(self):
    #abro el arch en el que guarde clusters
    self.kmeans = pickle.load(open("data16clusters/kmeans.pkl", "rb"))
  
  def load_pca(self):
    self.pca = pickle.load(open("data16clusters/pca.pkl", "rb"))

  def load_groups(self):
    self.groups = pickle.load(open("data16clusters/groups.pkl", "rb"))

  def save(self):
    self.save_kmeans()
    self.save_pca()
    self.save_groups()

  def load(self):
    self.load_kmeans()
    self.load_pca()
    self.load_groups()

  def get_top_n_similar_clusters(self, n, img, n_clusters):
    if n>self.kmeans.n_clusters :
      n= self.kmeans.n_clusters
    top_n = []
    for i in range(0,n_clusters):
      vector = self.kmeans.cluster_centers_[i]
      dist = {'cluster':i, 'distance':abs(1 - spatial.distance.cosine(img, vector))}
      top_n.append(dist);
    top_n=sorted(top_n, key = lambda x: x['distance'], reverse=True)
    return top_n[:n]

  def get_filenames_by_cluster(self, cluster):
    return self.groups[cluster]
