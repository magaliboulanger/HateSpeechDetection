
import pandas as pd
import os

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential,Input,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import math

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class MemeSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array([
            resize(imread(file_name), (224, 224,3))
               for file_name in batch_x]), np.array(batch_y)

def get_names_of_json(json_name,directory):

    train_path = directory + "/train.jsonl"
    dev_path = directory + "/dev.jsonl"
    test_path = directory + "/test.jsonl"

    data_train = pd.read_json(train_path, lines=True)
    data_test = pd.read_json(test_path, lines=True)
    data_dev = pd.read_json(dev_path, lines=True)

    if json_name == 'train':
        json = data_train
    if json_name == 'dev':
        json = data_dev
    if json_name == 'test':
        json = data_test

    list_aux = json['img'].to_list()
    labels = json['label'].to_list()

    list_of_names = dict()
    for i in range(0, len(list_aux)):
        list_of_names[list_aux[i][4:]] = labels[i]
    return list_of_names

def get_dataset(dataset, memes, labels, directory):
  dataset_names = get_names_of_json(dataset,directory)
  directory += "/img"
  with os.scandir(directory) as files:
    # loops through each file in the directory
    for file in files:
      print(file)
      if file.name.endswith('.png') :
        if dataset_names and file.name in dataset_names:
            # adds only the image files to the memes list
          labels.append(dataset_names[file.name])
          filepath = os.path.join(directory, file.name)
          memes.append(filepath)

INIT_LR = 1e-3
epochs = 6
batch_size = 100
memes=[]
labels=[]
get_dataset('train',memes,labels,"/hate/Dataset")
print(memes[:10])
sequence = MemeSequence(memes,labels,batch_size)
print(len(sequence))
classes = np.unique(labels)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(224,224,3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(32, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
# model.build((224,224,3))
#
# model.summary()

model.compile(loss=keras.losses.binary_crossentropy, optimizer=Adagrad(learning_rate=INIT_LR, decay=INIT_LR / 100),metrics=['accuracy'])
#y son las labels de cada meme
train_dropout = model.fit(sequence, batch_size=batch_size, epochs=epochs, verbose=1)

# guardamos la red, para reutilizarla en el futuro, sin tener que volver a entrenar
model.save("model_saved.h5py")

#evaluate
memes=[]
labels=[]
get_dataset('dev', memes, labels)
test_sequence= MemeSequence(memes,labels,10)
test= sport_model.evaluate(test_sequence, batch_size=batch_size)
print('Test loss:', test[0])
print('Test accuracy:', test[1])