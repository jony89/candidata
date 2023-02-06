import nltk
from nltk.corpus import stopwords
import pandas as pd

nltk.download("stopwords")

stop_words = stopwords.words("english")
snowball_stemmer = nltk.SnowballStemmer("english")


def remove_stopwords(sent):
    return [word for word in sent if word not in stop_words]


def word_stemming(sent):
    return [snowball_stemmer.stem(word) for word in sent]


tokenizer = nltk.RegexpTokenizer(r"\w+")

corpus = [
    "This is a sentence",
    "This is another sentence!",
    "The third Sentence is here",
    "There are four sentences",
]


sentences = [
    word_stemming(remove_stopwords(tokenizer.tokenize(sent.lower()))) for sent in corpus
]

# print(sentences)

processed_corpus = [" ".join(sent) for sent in sentences]

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

BOW = vectorizer.fit_transform(processed_corpus)

print(f"--------- BOW: \n {BOW}")
print(f"--------- BOW.toarray(): \n {BOW.toarray()}")
print(
    f"--------- vectorizer.get_feature_names_out(): \n {vectorizer.get_feature_names_out()}"
)

df = pd.DataFrame(data=BOW.toarray(), columns=vectorizer.get_feature_names_out())

print(f"--------- df: \n {df}")
