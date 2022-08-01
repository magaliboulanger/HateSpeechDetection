from helper_2 import Helper
from process import Clustering
from preprocess_images import ImageProcessing
from text_processing import TextProcessing
from PIL import Image
import os

PATH="/hate/Dataset/img_without_text"
MAX_K_SILUETA=300
TOP_OPTIMOS_K=10

tp = TextProcessing()
helper = Helper(tp)
images = ImageProcessing(PATH, helper.get_names_of_json('train'))
clustering = Clustering()
texts = helper.get_texts_of_json('train')
print(texts)
print(clustering.k_optimo_silueta(MAX_K_SILUETA, tp.encode_sentences(texts), TOP_OPTIMOS_K))
