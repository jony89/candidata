##############################################################
##### First we train and calculate for ethical indication ####
##############################################################

import pandas as pd
import string
import nltk
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import sent_tokenize


def preprocess_text_for_topic(texts):
    # Lowercase the text
    texts = [text.lower() for text in texts]

    # Remove punctuations
    texts = [
        text.translate(str.maketrans("", "", string.punctuation)) for text in texts
    ]

    # Remove numbers - not relevant for ethical
    texts = [text.translate(str.maketrans("", "", string.digits)) for text in texts]

    # Remove stop words and stem the text
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    texts_processed = []
    for text in texts:
        words = nltk.word_tokenize(text)
        words_filtered = [word for word in words if word not in stop_words]
        words_stemmed = [stemmer.stem(word) for word in words_filtered]
        texts_processed.append(" ".join(words_stemmed))

    return texts_processed


# Load the data
data_for_topic = pd.read_csv("./data/Risks.csv")

data_for_topic = data_for_topic.dropna(axis="rows", subset=["main_risk"])

# Pre-process the text data
texts = preprocess_text_for_topic(data_for_topic["text"].values)

# is_tos_encoding = 1 if data["TOS/PP"] == "TOS" else 0

main_risks = data_for_topic["main_risk"].values


model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


def encode_text_for_topic(text):
    return model.encode(sentences=text)


features = [encode_text_for_topic(text) for text in texts]


# Train the model

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, main_risks, test_size=0.2, random_state=42
)


##############################################
# Train the Logistic Regression classifier
##############################################

log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = log_clf.predict(X_test)

# Compute the accuracy
acc = accuracy_score(y_test, y_pred)
print("LogisticRegression Accuracy: {:.2f}%".format(acc * 100))


##############################################
# Train the SVM model
##############################################

clf = svm.SVC()
clf.fit(X_train, y_train)

# Predict on the test set
# X_test_vectored = [encode_text(sent) for sent in X_test]
y_pred = clf.predict(X_test)

# Compute the accuracy
acc = accuracy_score(y_test, y_pred)
print("SVM Accuracy: {:.2f}%".format(acc * 100))
# print("SVM Precision Score RF: ", precision_score(y_test, y_pred, average="weighted"))
# print("SVM Recall Score RF: ", recall_score(y_test, y_pred, average="weighted"))


##############################################
# Random forest
##############################################

rfc = RandomForestClassifier()

# Fit the model on the training data
rfc.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rfc.predict(X_test)

# Evaluate the model
acc = accuracy_score(y_test, y_pred)
print("Random forest Accuracy: {:.2f}%".format(acc * 100))

########################################################
# Train a Naive Bayes classifier on the training data
########################################################
clf = BernoulliNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Evaluate the model
acc = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy: {:.2f}%".format(acc * 100))


# Predict the topic for a new text
holdout_data = pd.read_csv("./data/holdout_result.csv")
texts_holdout = preprocess_text_for_topic(holdout_data["text"].values)
features_holdout = [encode_text_for_topic(text) for text in texts_holdout]
y_pred_holdout = log_clf.predict(features_holdout)

for (i, ethical) in enumerate(holdout_data["ethical"]):
    if ethical == 1:
        y_pred_holdout[i] = "NA"

## Save data
holdout_data["main_risk"] = y_pred_holdout
holdout_data.to_csv("./data/holdout_final.csv")
