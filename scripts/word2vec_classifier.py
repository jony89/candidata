import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import numpy as np

######################################
########### preprocessing ############
######################################

nltk.download("stopwords")

stop_words = stopwords.words("english")
tokenizer = nltk.RegexpTokenizer(r"\w+")


def remove_new_lines(sent):
    return sent.replace("\\n", " ")


# Load CSV file with semicolon separator
df = pd.read_csv("./data/800_jobs__clean_no_hebrew.csv")

corpus = df["title"] + " " + df["subTitle"] + " " + df["description"]
corpus = corpus.tolist()

## Remove stop words
stop_words = set(stopwords.words("english"))
corpus = [
    " ".join([word for word in description.split() if word not in stop_words])
    for description in corpus
]

parsed_corpus = [remove_new_lines(sent.lower()) for sent in corpus]

#### train the model

# Load a pre-trained transformer model
model = SentenceTransformer("bert-base-nli-mean-tokens")

# Encode the descriptions into feature vectors
X = model.encode(parsed_corpus)

# Convert the grades into numerical labels
y = df["applications"].tolist()

# Train a LogisticRegression model on the feature vectors and labels
clf = LogisticRegression()
clf.fit(X, y)

# Use the trained model to predict the grade for a new description
new_description = "The class is poorly organized and the instructor is unhelpful"
new_feature_vector = model.encode(new_description)
new_grade = clf.predict(new_feature_vector)
print("Predicted grade for new description: ", new_grade)
