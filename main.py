from helper.helper import Helper
from clustering.process import Clustering
from image_processing.preprocess_images import ImageProcessing
from text_processing.text_processing import TextProcessing, TextProcessingDistances
import json

TOP_N_CLUSTERS=1
NUM_CLUSTERS=8
PATH="/hate/Dataset/img_without_text"

print("Creating Helper")
tp = TextProcessingDistances()
helper = Helper(tp)
print("Creating Image processing")
images = ImageProcessing(PATH, helper.get_names_of_json('train'))
images.load_pca()
clustering = Clustering()
#new_data = helper.remove_confounders_images()
#print(new_data)

new = helper.construct_new_data()


#with open("lista_survivors.txt","w") as file:
#    file.write(str(new_data))

#with open('lista_survivors.json', 'w') as fout:
#    json.dump(new_data, fout)
#print("Clustering...")
#
#texts = helper.get_texts_of_json('train')
#print('loading')

#creacion de clsters
#clustering.group(NUM_CLUSTERS, images.prepare_model(), helper.get_names_of_json('train_without_confounders_img'))
#clustering.save()

#clustering.load()
#clustering.view_cluster_texts(1)
#clustering.view_number_of_elements_per_cluster()

##   Pruebas
#print("METRICS: ")
#print(helper.get_accuracy_image_text(images, TOP_N_CLUSTERS, NUM_CLUSTERS, clustering))
