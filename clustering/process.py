# for loading/processing the images
from tensorflow.keras.preprocessing.image import load_img

# import resnet

# clustering and dimension reduction
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, OPTICS

# for everything else
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import spatial

from sklearn.mixture import GaussianMixture


BASE=2

class Clustering():
    def __init__(self):
        self.kmeans = None

    def group(self, n_clusters, features, filenames):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=22)
        #self.kmeans = GaussianMixture(n_components=n_clusters)
        #self.kmeans = Birch(threshold=0.03, n_clusters=n_clusters)
        #self.kmeans = AgglomerativeClustering(n_clusters=n_clusters)
        #self.kmeans = OPTICS(metric='cosine')
        self.kmeans.fit(features)
        self.groups = {}
        for file, cluster in zip(filenames, self.kmeans.labels_):
            if cluster not in self.groups.keys():
                self.groups[cluster] = []
            self.groups[cluster].append(file)

    def view_cluster_imgs(self, cluster):
        plt.figure(figsize=(25, 25));
        # gets the list of filenames for a cluster
        files = self.groups[cluster]
        # only allow up to 30 images to be shown at a time
        if len(files) > 30:
            print(f"Clipping cluster size from {len(files)} to 30")
            files = files[:29]
        # plot each image in the cluster
        for index, file in enumerate(files):
            plt.subplot(10, 10, index + 1);
            img = load_img(file)
            img = np.array(img)
            plt.imshow(img)
            plt.axis('off')

    def view_cluster_texts(self, cluster):
        files = self.groups[cluster]
        # only allow up to 30 images to be shown at a time
        if len(files) > 30:
            print(f"Clipping cluster size from {len(files)} to 30")
            files = files[:29]
        # plot each image in the cluster
        for file in files:
            print(file)

    def view_number_of_elements_per_cluster(self):
        print("Esta es la distribución para nro de clusters:", len(self.groups))
        print("NRO. DE CLUSTER , CANT. ELEM.")
        k=0
        view_as_array = []
        for i in self.groups:
            k+=1
            print(k, len(self.groups[i]))
            view_as_array.append(len(self.groups[i]))
        print(view_as_array)

    def k_optimo_codo(self, max_k, features):
        # metodo del codo
        sse = []
        list_k = list(range(BASE, max_k))

        for k in list_k:
            km = KMeans(n_clusters=k, random_state=22)
            km.fit(features)

            sse.append(km.inertia_)

        # Plot sse against k
        plt.figure(figsize=(6, 6))
        plt.plot(list_k, sse)
        plt.xlabel('Number of clusters *k*')
        plt.ylabel('Sum of squared distance')


    def k_optimo_silueta(self, max_k, features, n_optimo):
        # metodo de la silueta
        #self.prepare_model()
        silueta = []
        from sklearn.metrics import silhouette_score
        K = range(BASE, max_k)
        for k in K:
            print(k)
            kmeans = KMeans(n_clusters=k).fit(features)
            labels = kmeans.labels_
            # calcular el puntaje de la silueta promedio
            silueta.append(silhouette_score(features, labels, metric='euclidean'))

        # graficar los valores obtenidos
        print(silueta)
        plt.plot(K, silueta, 'bx-')
        plt.xlabel('Clusteres')
        plt.ylabel('Puntaje de la Silueta')
        plt.title('Metodo de la Silueta')
        plt.show()
        return self.top_optimos(silueta, n_optimo)
        # El valor de k viene dado por el valor mas alto de la grafica (maximo global)



    def top_optimos(self, silueta, n_optimo):
      sorted_silueta = sorted(silueta, reverse=True)
      optimos = []
      if n_optimo<=len(silueta):
        del sorted_silueta[n_optimo:]
      for i in sorted_silueta:
        optimos.append(silueta.index(i)+BASE)
      return optimos


    def predict(self, x, n_clusters, features = None, filenames=None):
        # predice cuál es el cluster al cual x pertenece
        if self.kmeans is None:
            self.group(n_clusters,features,filenames)

        return self.kmeans.predict(x)

    # saves y loads. TO DO: refactorizar
    def save_kmeans(self):
        # lo guardo en el arch kmeans.pkl. w=write. b=binary.
        pickle.dump(self.kmeans, open("kmeans.pkl", "wb"))


    def save_groups(self):
        pickle.dump(self.groups, open("groups.pkl", "wb"))

    def load_kmeans(self):
        # abro el arch en el que guarde clusters
        self.kmeans = pickle.load(open("kmeans.pkl", "rb"))


    def load_groups(self):
        self.groups = pickle.load(open("groups.pkl", "rb"))

    def save(self):
        self.save_kmeans()
        self.save_groups()

    def load(self):
        self.load_kmeans()
        self.load_groups()

    def get_top_n_similar_clusters(self, n, img, n_clusters):
        if n > self.kmeans.n_clusters:
            n = self.kmeans.n_clusters
        top_n = []
        for i in range(0, n_clusters):
            vector = self.kmeans.cluster_centers_[i]
            print("vector, centros de cluster")
            print(len(vector))
            print("imagen")
            print(len(img))
            dist = {'cluster': i, 'distance': abs(1 - spatial.distance.cosine(img, vector))}
            top_n.append(dist)
        top_n = sorted(top_n, key=lambda x: x['distance'], reverse=True)
        return top_n[:n]

    def get_filenames_by_cluster(self, cluster):
        return self.groups[cluster]
