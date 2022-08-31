from text_processing.text_processing import TextProcessing
from image_processing.preprocess_images import ImageProcessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from helper.helper import Helper

tp = TextProcessing()
helper = Helper(tp)
#texts = helper.get_texts_of_json('train')
names =helper.get_names_of_json('train_without_confounders_img')
ip = ImageProcessing('/Users/magaliboulanger/Documents/Dataset/img',names)


X_embedded = TSNE(n_components=2,init='random').fit_transform(ip.prepare_model())
plt.scatter(X_embedded[:,0], X_embedded[:,1])
plt.show()