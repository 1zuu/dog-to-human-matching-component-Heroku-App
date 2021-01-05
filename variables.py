import os
seed = 1234

target_size = (224, 224)
input_shape = (224, 224, 3)

train_dir = 'data/Images/'
save_path = 'weights/numpy_images.npz'
cnn_converter_path = "weights/cnn_model.tflite"

ann_cols = ['Accomodation','Garden','Hours', 'Gender', 'Age', 'Size']
encoder_dict_path = 'weights/encoder dict.pickle'

csv_path = 'data/data.csv'
max_length = 30
trunc_type = 'post'

tokenizer_path = 'weights/tokenizer.pickle'
crn_converter_path = "weights/crn_model.tflite"

inference_save_path = 'weights/inference_images.npz'
dog_classes = {
            'shih tzu' : '0', 
            'papillon' : '1', 
            'maltese' : '2', 
            'afghan hound' : '3', 
            'beagle' : '4'
            }
n_neighbour_weights = 'weights/n_neighbor weights/nearest neighbour {}.pkl'
cloud_image_dir = "https://res.cloudinary.com/douc1omvg/image/upload/Dog_to_Human_Matching_Component/"
n_neighbour = 3
min_test_sample = 30
