from sentence_transformers import SentenceTransformer

import pandas as pd
import string
import nltk
import numpy as np

model = SentenceTransformer("bert-base-nli-mean-tokens")

import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

df = pd.read_csv("./data/Risks.csv")

texts = df["text"].tolist()
sentences = [sent_tokenize(text) for text in texts]
sentences = [sentence for sublist in sentences for sentence in sublist]

sentence_embeddings = model.encode(sentences)

from sklearn.model_selection import train_test_split

sentence_labels = df["ethical"].tolist()


text_embeddings = []
for i in range(len(texts)):
    text = texts[i]
    text_sentences = sent_tokenize(text)
    text_sentence_embeddings = model.encode(text_sentences)
    text_embedding = np.mean(text_sentence_embeddings, axis=0)
    text_embeddings.append(text_embedding)

train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    text_embeddings, sentence_labels, test_size=0.2, random_state=42
)


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(train_sentences, train_labels)

from sklearn.metrics import accuracy_score

pred_labels = clf.predict(test_sentences)
print("Accuracy:", accuracy_score(test_labels, pred_labels))
