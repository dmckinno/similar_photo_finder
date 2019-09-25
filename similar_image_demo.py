"""
This demo reads images from a folder and sorts them into
a user-specified number of buckets based on Euclidean
distance between ResNet-50 embeddings.

Arguments:
    folder (str): Folder where the images to be sorted are
                  located relative to cwd.
    num_clusters (int): Number of buckets.

Returns:
    Nothing

To-do:
    * optimize distance calculation
    * improve model architecture/pooling
    * consider semi-supervision

Created on Sun Jul 21 2019
Author: Daniel McKinnon
"""

import os
import sys

import numpy as np

from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

from sklearn.cluster import KMeans

def read_imgs(folder):
    """
    Read images from file structure, reduce resolution to 224 x 224,
    return a dict full of images
    """
    cwd = os.getcwd()
    img_titles = {}
    for _, _, files in os.walk(cwd+'/'+folder):
        for name in files:
            img_titles[name] = image.load_img(cwd+'/'+folder+'/'+name, target_size=(224, 224))
    return img_titles

def preprocess_imgs(img_titles):
    """
    For each item in dict, convert image to np.array, add fourth dimension for Keras,
    and scale the inputs to match the expectations of ResNet (centered at 0, not normalized)
    """
    for img in img_titles:
        img_titles[img] = preprocess_input(np.expand_dims(image.img_to_array(img_titles[img]),
                                                          axis=0))
    return img_titles

def predict_imgs(img_titles):
    """
    Convert a 224x224 np.array into a 2048 vector using a ResNet-50 model
    trained on ImageNet. This step is slow.
    """
    base_model = ResNet50(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    for img in img_titles:
        img_titles[img] = model.predict(img_titles[img])
    return img_titles

def format_img_embeddings(img_titles):
    """
    Convert the image embeddings into a nice np
    array that our K-means function can use
    """
    labels = []
    embeddings = []
    for img in img_titles:
        labels.append(img)
        embeddings.append(img_titles[img])
    embeddings = np.array(embeddings).squeeze()
    return labels, embeddings

def calculate_nearest_neighbors(embeddings, num_clusters):
    """
    Sorted the images into classes based on Euclidean distance using the k-means algo
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    return kmeans

def assign_labels_to_imgs(labels, kmeans):
    """
    Merge k-means results with labels and generate a tuple pairing a label with an image
    """
    img_label = sorted(list(zip(labels, kmeans.labels_)), key=lambda x: x[1])
    return img_label

def sort_imgs_into_folders(img_label, folder, num_clusters):
    """
    Read images and labels and sort them into folders for QA/analysis
    """
    cwd = os.getcwd()
    cluster_list = list(range(0, num_clusters))
    for cluster in cluster_list:
        try:
            os.mkdir(cwd+'/'+str(cluster))
        except Exception as e:
            print(e)
    for doublet in img_label:
        os.rename(cwd+'/'+folder+'/'+doublet[0], cwd+'/'+str(doublet[1])+'/'+doublet[0])

def analyze_imgs(folder, num_clusters):
    """
    Run sorting pipeline
    """
    img_titles = read_imgs(folder)
    img_titles = preprocess_imgs(img_titles)
    img_titles = predict_imgs(img_titles)
    labels, embeddings = format_img_embeddings(img_titles)
    kmeans = calculate_nearest_neighbors(embeddings, num_clusters)
    img_label = assign_labels_to_imgs(labels, kmeans)
    sort_imgs_into_folders(img_label, folder, num_clusters)

if __name__ == '__main__':
    analyze_imgs(sys.argv[1], int(sys.argv[2]))