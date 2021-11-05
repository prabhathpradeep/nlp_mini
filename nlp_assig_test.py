
from keras.backend import sigmoid
from keras.layers.core import SpatialDropout1D
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd

import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import initializers, optimizers, layers


train = pd.read_csv('train.csv')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train[list_classes].values

list_sequences_train = train["comment_text"]
print(list_sequences_train.head())

max_features = 22000
tokenizer = Tokenizer(num_words=max_features)
train = tokenizer.fit_on_texts(list(list_sequences_train))

#Tokenizing and Indexing the comments
list_tokenized_train = tokenizer.texts_to_sequences(list_sequences_train)

maxlen = 200
X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)

# Model Layers
model = Sequential()
model.add(Embedding(max_features, 128,input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.1))
model.add(LSTM(60, dropout=0.1,recurrent_dropout=0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(6,activation='sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

# TRAINING
batch_size = 32
epochs = 1
model.fit(X_train,
          y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)

# Saving a model for deployment
model.save('model.h5')

