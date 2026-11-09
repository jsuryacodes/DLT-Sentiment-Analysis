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


def build_rnn():
    """(b) Embedding + SimpleRNN: reads the review word by word, keeping a hidden state."""
    model = keras.Sequential([
        keras.Input(shape=(MAX_LEN,)),
        layers.Embedding(VOCAB_SIZE, EMBED_DIM),
        layers.SimpleRNN(32),
        layers.Dense(1, activation="sigmoid"),
    ], name="embedding_simplernn")
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


def plot_history(history, name):
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS, exist_ok=True)
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in ((ax1, "accuracy"), (ax2, "loss")):
        ax.plot(epochs, h[metric], "o-", label="train")
        ax.plot(epochs, h["val_" + metric], "s--", label="validation")
        ax.set(title=f"{name} - {metric}", xlabel="epoch", ylabel=metric)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS, f"{name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("saved", path)


_word_index = None


def encode(text):
    """Turn raw text into the same integer sequence format as keras.datasets.imdb."""
    global _word_index
    if _word_index is None:
        _word_index = keras.datasets.imdb.get_word_index()
    # imdb.load_data reserves 0=pad, 1=start, 2=out-of-vocabulary, so real word ids are shifted by 3
    ids = [1]
    for w in re.findall(r"[a-z0-9']+", text.lower()):
        i = _word_index.get(w, -1) + 3
        ids.append(i if 3 <= i < VOCAB_SIZE else 2)
    return keras.utils.pad_sequences([ids], maxlen=MAX_LEN)


def predict(model, text):
    p = float(model.predict(encode(text), verbose=0)[0, 0])
    return ("positive" if p >= 0.5 else "negative"), p


SAMPLES = [
    "This movie was absolutely wonderful, the acting was brilliant and I loved the story.",
    "A total waste of time. The plot was boring and the acting was terrible.",
    "It had a few good moments but overall I was disappointed.",
]

if __name__ == "__main__":
    data = load_data()
    models = [(build_baseline(), 10), (build_rnn(), 5)]
    rows = []
    for model, epochs in models:
        history, loss, acc = train_and_evaluate(model, data, epochs)
        plot_history(history, model.name)
        rows.append((model.name, model.count_params(), loss, acc))

    print("\n| Model | Params | Test loss | Test accuracy |\n|---|---|---|---|")
    for name, params, loss, acc in rows:
        print(f"| {name} | {params:,} | {loss:.4f} | {acc:.2%} |")

    for text in SAMPLES:
        print("\n" + text)
        for model, _ in models:
            label, p = predict(model, text)
            print(f"  {model.name}: {label} ({p:.3f})")
