import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import svm

# from sentence_transformers import SentenceTransformer
import numpy as np
import numpy as np
import gensim.downloader as api
from sklearn.metrics import accuracy_score

# Download pre-trained word2vec embeddings
embeddings = api.load("word2vec-google-news-300")

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


# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    corpus, df["applications"], test_size=0.2, random_state=42
)


#### train the model
def encode_text(text):
    return np.mean(
        [
            embeddings.word_vec(word)
            for word in text.split()
            if word in embeddings.key_to_index
        ],
        axis=0,
    )


X_train_vectored = [encode_text(sent) for sent in X_train]

###################################################
# Train a LogisticRegression model on the feature vectors and labels
###################################################
clf = LogisticRegression()
clf.fit(X_train_vectored, y_train)

## Test the accuracy score
# Predict on the test set
X_test_vectored = [encode_text(sent) for sent in X_test]
y_pred = clf.predict(X_test_vectored)

# Compute the accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(acc * 100))

###################################################
# Train a Random forest model on the feature vectors and labels
###################################################

# Initialize the model
rfc = RandomForestClassifier()

# Fit the model on the training data
rfc.fit(X_train_vectored, y_train)

# Make predictions on the test data
X_test_vectored = [encode_text(sent) for sent in X_test]
rfc_predictions = rfc.predict(X_test_vectored)

# Evaluate the model
print("Accuracy RF: ", accuracy_score(y_test, rfc_predictions))
print(
    "Precision Score RF: ", precision_score(y_test, rfc_predictions, average="weighted")
)
print("Recall Score RF: ", recall_score(y_test, rfc_predictions, average="weighted"))

# Train the SVM model
clf = svm.SVC()
clf.fit(X_train_vectored, y_train)

# Predict on the test set
X_test_vectored = [encode_text(sent) for sent in X_test]
y_pred = clf.predict(X_test_vectored)

# Compute the accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(acc * 100))
print(
    "Precision Score RF: ", precision_score(y_test, rfc_predictions, average="weighted")
)
print("Recall Score RF: ", recall_score(y_test, rfc_predictions, average="weighted"))

print("Thanks for coming here")
