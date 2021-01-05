import os
import re
import pickle
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from variables import*
np.random.seed(seed)

def preprocessing_function(img):
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def rescale_imgs(img):
    return (img * 127.5) + 127.5

def lemmatization(lemmatizer,sentence):
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    lem = set(lem)
    return [k for k in lem]

def remove_stop_words(stopwords_list,sentence):
    return [k for k in sentence if k not in stopwords_list]

def preprocess_one(review):
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    review = review.lower()
    remove_punc = tokenizer.tokenize(review) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_num if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_review = ' '.join(remove_stop)
    return updated_review

def preprocessed_data(reviews):
    updated_reviews = []
    if isinstance(reviews, np.ndarray) or isinstance(reviews, list):
        for review in reviews:
            updated_review = preprocess_one(review)
            updated_reviews.append(updated_review)
    elif isinstance(reviews, np.str_)  or isinstance(reviews, str):
        updated_reviews = [preprocess_one(reviews)]

    return np.array(updated_reviews)

def tokenizing_data(Xtrain, Xtest):
    if not os.path.exists(tokenizer_path):
        tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(Xtrain)
        save_tokenizer(tokenizer)

    else:
        tokenizer = load_tokenizer()

    Xtrain_seq = tokenizer.texts_to_sequences(Xtrain)
    Xtrain_pad = pad_sequences(Xtrain_seq, maxlen=max_length, truncating=trunc_type)

    Xtest_seq  = tokenizer.texts_to_sequences(Xtest)
    Xtest_pad = pad_sequences(Xtest_seq, maxlen=max_length)
    return Xtrain_pad, Xtest_pad

def tokenize_inference_text(X):
    tokenizer = load_tokenizer()
    X_seq = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(X_seq, maxlen=max_length, truncating=trunc_type)
    return X_pad

def save_tokenizer(tokenizer):
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer():
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def filter_images(image_labels):
    idxs = []
    for label in dog_classes:
        idx = np.where(image_labels==label)[0]
        if len(idx) > min_test_sample:
            idx = np.random.choice(idx, min_test_sample, replace=False)
        idxs.extend(idx.tolist())
    return idxs


def load_inference_data():
    if not os.path.exists(inference_save_path):
        print(" Inference Images are Saving ")
        df = pd.read_csv(csv_path, encoding='ISO 8859-1')
        df = df.drop_duplicates(subset=['ImageName'])
        df['Breed'] = df['Breed'].str.lower()
        df['Breed'] = df['Breed'].replace('afgan hound', 'afghan hound')

        img_names = df['ImageName'].str.strip().values 

        image_labels = []
        inference_images = []
        url_paths = []
        dog_folders = os.listdir(train_dir)
        for label in list(dog_folders):
            label_dir = os.path.join(train_dir, label)
            for img_name in os.listdir(label_dir):
                img_ = img_name.split('.')[0].strip()
                if img_ not in img_names:
                    label_class = dog_classes[label]
                    img_path = os.path.join(label_dir, img_name)
                    url_path = os.path.join(cloud_image_dir, label_class, img_name)
                    img = cv.imread(img_path)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img = cv.resize(img, target_size)
                    img = preprocessing_function(img)
                    inference_images.append(img)
                    image_labels.append(label)
                    url_paths.append(url_path)

        inference_images = np.array(inference_images).astype('float32')
        image_labels = np.array(image_labels).astype('str')
        image_urls = np.array(url_paths).astype('str')

        image_labels, inference_images, image_urls = shuffle(image_labels, inference_images, image_urls)
        idxs = filter_images(image_labels)

        inference_images = inference_images[idxs]
        image_labels = image_labels[idxs]
        image_urls = image_urls[idxs]

        np.savez(inference_save_path, name1=inference_images, name2=image_labels, name3=image_urls)

    else:
        print(" Inference Images are Loading ")
        data = np.load(inference_save_path, allow_pickle=True)
        inference_images = data['name1']
        image_labels = data['name2']
        image_urls = data['name3']

    return image_labels, inference_images, image_urls

def load_labeled_data(image_labels, inference_images, image_urls, label):
    idxs = (image_labels==label)
    labels = image_labels[idxs]
    images = inference_images[idxs]
    urls = image_urls[idxs]
    return labels, images, urls

def label_encoding(df_cat):
    if not os.path.exists(encoder_dict_path):
        encoder_dict = defaultdict(LabelEncoder)
        encoder = df_cat.apply(lambda x: encoder_dict[x.name].fit_transform(x))
        encoder.apply(lambda x: encoder_dict[x.name].inverse_transform(x))
        with open(encoder_dict_path, 'wb') as handle:
            pickle.dump(encoder_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(encoder_dict_path, 'rb') as handle:
            encoder_dict = pickle.load(handle)
    return df_cat.apply(lambda x: encoder_dict[x.name].transform(x))

def get_feature_data(feature_data):
    feature_data = eval(feature_data)
    df_dict = {col:[feature_value] for col, feature_value in zip(ann_cols, feature_data)}
    df = pd.DataFrame(df_dict,
                      columns=ann_cols)
    df_cat = df[['Accomodation','Garden', 'Gender', 'Age', 'Size']]
    df[['Accomodation','Garden', 'Gender', 'Age', 'Size']] = label_encoding(df_cat)
    return df.values.squeeze()

def get_prediction_data(data):
    text, label, feature_data = data["text"], data["label"], data['feature']
    feature = get_feature_data(feature_data)
    label = str(label).lower()
    text = preprocessed_data(text)
    text_pad = tokenize_inference_text(text)[0]
    text_pad = text_pad.reshape(1, -1)
    feature = feature.reshape(1, -1)
    return text_pad, feature, label