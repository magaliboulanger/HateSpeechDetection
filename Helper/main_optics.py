from helper_2 import Helper
from process import Clustering
from preprocess_images import ImageProcessing
from text_processing import TextProcessing
PATH="/hate/Dataset/img_without_text"

print("Creating Helper")
tp = TextProcessing()
helper = Helper(tp)
print("Creating Image processing")
images = ImageProcessing(PATH, helper.get_names_of_json('train'))
#images.prepare_model()
images.load_pca()
print("Clustering...")
clustering = Clustering()
texts = helper.get_texts_of_json('train')
print('loading')

#creacion de clsters
#clustering.group(0, tp.encode_sentences(texts), helper.get_names_of_json('train'))
#clustering.save()

clustering.load()
#clustering.view_cluster_texts(1)
clustering.view_number_of_elements_per_cluster()



##   Pruebas
#print("METRICS: ")
print(helper.get_accuracy(clustering, 1, 190, images))
