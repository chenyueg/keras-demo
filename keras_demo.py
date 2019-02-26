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

####################################

# Convert the integers back to words
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

####################################

# Display the text for the first review
print(decode_review(train_data[0]))

# Use pad_sequences to standardize the lengths
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# Look at the length of the examples
print(len(train_data[0]))
print(len(train_data[1]))

# Look at the padded first review
print(train_data[0])
print(decode_review(train_data[0]))

# The size (See line 13)
vocab_size = 10000

# The architecture

# This is a sequential model, meaning the layers are stacked.
model = keras.Sequential()

# The first layer is an Embedding layer. This layer takes the integer-encoded vocabulary and looks up
# the embedding vector for each word-index. These vectors are learned as the model trains. The vectors
# add a dimension to the output array. The resulting dimensions are: (batch, sequence, embedding).
model.add(keras.layers.Embedding(vocab_size, 16))

# A GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging
# over the sequence dimension. This allows the model to handle input of variable length, in the
# simplest way possible.
model.add(keras.layers.GlobalAveragePooling1D())

# This fixed-length output vector is piped through a fully-connected layer with 16 hidden units.
model.add(keras.layers.Dense(16, activation=tf.nn.relu))

# The last layer is densely connected with a single output node. Using the sigmoid activation function,
# this value is a float between 0 and 1, representing a probability, or confidence level.
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# A summary of the model
model.summary()

# Configure the model with an optimizer and a loss function
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Create a validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Train the model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# Evaluate the model
results = model.evaluate(test_data, test_labels)
print(results)