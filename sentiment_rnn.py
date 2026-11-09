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


def build_baseline():
    """(a) Embedding + GlobalAveragePooling: averages word vectors, ignores word order."""
    model = keras.Sequential([
        keras.Input(shape=(MAX_LEN,)),
        layers.Embedding(VOCAB_SIZE, EMBED_DIM),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="embedding_gap")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_evaluate(model, data, epochs):
    (x_train, y_train), (x_test, y_test) = data
    model.summary()
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=128,
                        validation_split=0.2, verbose=2)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{model.name}: test loss={loss:.4f}  test accuracy={acc:.4f}")
    return history, loss, acc


if __name__ == "__main__":
    data = load_data()
    train_and_evaluate(build_baseline(), data, epochs=10)
