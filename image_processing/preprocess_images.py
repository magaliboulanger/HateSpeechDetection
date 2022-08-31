#models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

# for loading/processing the images
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input

# import resnet

# clustering and dimension reduction
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import pickle

class ImageProcessing():
    def __init__(self, path, dataset_names):
        self.path = path
        os.chdir(path)
        self.model = VGG16()
        self.model = Model(inputs=self.model.inputs, outputs=self.model.layers[-2].output)
        self.pca = 0
        # this list holds all the image filename
        self.memes = []

        # creates a ScandirIterator aliased as files
        with os.scandir(self.path) as files:
            # loops through each file in the directory
            for file in files:
                if file.name.endswith('.png'):
                    if dataset_names and file.name in dataset_names:
                        # adds only the image files to the memes list
                        self.memes.append(file.name)
        print(len(self.memes))

    def prepare_image(self, img):

        feat = self.extract_features(img)
        return self.pca.transform(feat)

    def extract_features(self, file):
        # load the image as a 224x224 array
        img = load_img(file, target_size=(224, 224))
        # convert from 'PIL.Image.Image' to numpy array
        img = np.array(img)
        # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
        reshaped_img = img.reshape(1, 224, 224, 3)
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
                with open(p, 'wb') as file:
                    pickle.dump(data, file)

        # get a list of the filenames

        self.filenames = np.array(list(data.keys()))
        print("filenames")
        # get a list of just the features
        feat = np.array(list(data.values()))
        print("features")
        feat.shape
        (210, 1, 4096)

        # reshape so that there are 210 samples of 4096 vectors
        feat = feat.reshape(-1, 4096)
        feat.shape
        (210, 4096)
        if self.pca == 0:
            self.pca = PCA(n_components=10, random_state=22)
            self.pca.fit(feat)
        self.x = self.pca.transform(feat)
        return self.x


    def save_pca(self):
        pickle.dump(self.pca, open("pca.pkl", "wb"))


    def load_pca(self):
        self.pca = pickle.load(open("pca.pkl", "rb"))
