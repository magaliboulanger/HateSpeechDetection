from helper.helper import Helper
from image_processing.image_processing import ImageProcessing
from PIL import Image
import os
TOP_N_CLUSTERS=5
NUM_CLUSTERS=16
print("Creating Helper")
helper = Helper()
print("Creating Image processing")
images = ImageProcessing("/hate/Dataset/img_without_text", helper.get_names_of_json('train'))
print("Loading groups...")


##   Pruebas
print("METRICS: ")
print(helper.get_accuracy(images, TOP_N_CLUSTERS, NUM_CLUSTERS))


