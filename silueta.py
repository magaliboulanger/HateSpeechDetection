from helper.helper import Helper
from image_processing.image_processing import ImageProcessing

MAX_K = 500

helper = Helper()
print("Creating Image processing")
images = ImageProcessing("/hate/Dataset/img_without_text", helper.get_names_of_json('train'))
images.k_optimo_silueta(MAX_K)