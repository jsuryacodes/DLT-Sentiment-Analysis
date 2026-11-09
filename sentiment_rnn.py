"""Basic Sentiment Analysis on IMDB reviews (DLT lab).

Run: python sentiment_rnn.py
"""
import os
import re

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

VOCAB_SIZE = 10_000  # keep only the 10k most frequent words
MAX_LEN = 200        # pad/truncate every review to 200 tokens
EMBED_DIM = 32
RESULTS = "results"

tf.random.set_seed(42)
np.random.seed(42)


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
    x_train = keras.utils.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = keras.utils.pad_sequences(x_test, maxlen=MAX_LEN)
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    print("Train:", x_train.shape, "Test:", x_test.shape, "Positive ratio:", y_train.mean())
