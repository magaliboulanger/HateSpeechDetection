
from helper_2 import Helper
from text_processing import InferSentProcessing
from preprocess_images import ImageProcessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tp = InferSentProcessing()
helper = Helper(tp)
texts = helper.get_texts_of_json('train')
names =helper.get_names_of_json('train')
ip = ImageProcessing('/hate/Dataset/img_without_text',names)


X_embedded = TSNE(n_components=2,init='random').fit_transform(ip.prepare_model())
plt.scatter(X_embedded[:,0], X_embedded[:,1])
plt.show()