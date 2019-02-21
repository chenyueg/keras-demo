# Import TensorFlow
import tensorflow as tf

# Import Keras
from tensorflow import keras

# Import NumPy
import numpy as np

# Download the IMDB dataset
imdb = keras.datasets.imdb

# The argument num_words = 10000 keeps the top 10,000 most frequent occurring words in the training data.
(train_data, train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)

# Explore the data
print("Training entries: {}, labels: {}".format(len(train_data),len(train_labels)))

# Each integer represents a specific word in a dictionary.
print(train_data[0])

# Movie reviews may be of different lengths.
print(len(train_data[0]),len(train_data[1]))

# Convert the integers back to words
word_index = imdb.get_word_index()

