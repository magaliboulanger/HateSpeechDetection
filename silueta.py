from helper import Helper
from clustering.process import Clustering
from image_processing.preprocess_images import ImageProcessing
from text_processing.text_processing import TextProcessing

PATH="/hate/Dataset/img_without_text"
MAX_K_SILUETA=8500
TOP_OPTIMOS_K=10

tp = TextProcessing()
helper = Helper(tp)
images = ImageProcessing(PATH, helper.get_names_of_json('train'))
clustering = Clustering()
texts = helper.get_texts_of_json('train')
print(texts)
print(clustering.k_optimo_silueta(MAX_K_SILUETA, tp.encode_sentences(texts), TOP_OPTIMOS_K))
