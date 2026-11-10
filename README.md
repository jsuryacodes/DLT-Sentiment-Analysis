# Basic Sentiment Analysis on IMDB Reviews

Deep Learning Techniques (DLT) Lab Project

## Aim

To classify IMDB movie reviews as **positive** or **negative** using simple deep learning models built with TensorFlow/Keras:

- (a) Embedding + GlobalAveragePooling (baseline)
- (b) Embedding + SimpleRNN

## Dataset

The built-in `keras.datasets.imdb` dataset:

- 50,000 movie reviews: 25,000 for training and 25,000 for testing
- Labels: 1 = positive, 0 = negative (balanced 50/50)
- Reviews are already converted into sequences of word indices

## Preprocessing

1. Kept only the **10,000** most frequent words (`num_words=10000`). Rarer words are mapped to an "unknown" token.
2. Padded or truncated every review to **200** words using `pad_sequences`.
3. Used 20% of the training data as a validation set.

## Model Architectures

**(a) Embedding + GlobalAveragePooling**

| Layer | Output shape | Params |
|---|---|---|
| Embedding(10000, 32) | (200, 32) | 320,000 |
| GlobalAveragePooling1D | (32) | 0 |
| Dense(16, relu) | (16) | 528 |
| Dense(1, sigmoid) | (1) | 17 |
| **Total** | | **320,545** |

**(b) Embedding + SimpleRNN**

| Layer | Output shape | Params |
|---|---|---|
| Embedding(10000, 32) | (200, 32) | 320,000 |
| SimpleRNN(32) | (32) | 2,080 |
| Dense(1, sigmoid) | (1) | 33 |
| **Total** | | **322,113** |

**Training settings:** Adam optimizer, binary cross-entropy loss, batch size 128. The baseline trains for up to 10 epochs and the RNN for up to 5. Early stopping (patience 2) keeps the best weights based on validation loss.

## Results

| Model | Test loss | Test accuracy |
|---|---|---|
| Embedding + GlobalAveragePooling | 0.2953 | **87.78%** |
| Embedding + SimpleRNN | 0.3895 | 83.86% |

Accuracy/loss plots:

![Baseline](results/embedding_gap.png)
![SimpleRNN](results/embedding_simplernn.png)

Predictions on custom reviews:

| Review | Baseline | SimpleRNN |
|---|---|---|
| "This movie was absolutely wonderful, the acting was brilliant and I loved the story." | positive | positive |
| "A total waste of time. The plot was boring and the acting was terrible." | negative | negative |
| "It had a few good moments but overall I was disappointed." | positive ✗ | negative |
| "One of the best films I have seen this year." | positive | negative ✗ |

*(The numbers vary slightly between runs.)*

## How to Run

```bash
git clone https://github.com/jsuryacodes/DLT-Sentiment-Analysis.git
cd DLT-Sentiment-Analysis
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the script. It trains both models, saves the plots to `results/`, and prints the predictions:

```bash
python sentiment_rnn.py
```

Or open the notebook:

```bash
jupyter notebook sentiment_analysis.ipynb
```

To classify your own review in Python:

```python
import sentiment_rnn as S
data = S.load_data()
model = S.build_baseline()
S.train_and_evaluate(model, data, epochs=10)
print(S.predict(model, "Great movie, I enjoyed it a lot!"))
```

## Files

- `sentiment_rnn.py`: data loading, both models, training, plots, and the `predict()` function
- `sentiment_analysis.ipynb`: step-by-step notebook with outputs
- `results/`: accuracy/loss plots
- `requirements.txt`: dependencies (Python 3.12, since TensorFlow does not support newer versions yet)

## Conclusion

Both models learned to classify movie reviews. The simple Embedding + GlobalAveragePooling model did better (87.8%) than the SimpleRNN (83.9%), and it also trained much faster. Most of the sentiment in a review comes from individual words like "brilliant" or "terrible", so averaging the word embeddings works well. The SimpleRNN started overfitting after 2–3 epochs and has trouble remembering information across 200 words (the vanishing gradient problem). Both models make mistakes on mixed or short reviews. Using LSTM or GRU layers would be the next step to improve the RNN.
