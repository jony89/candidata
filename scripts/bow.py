from nltk.tokenize import word_tokenize
from nltk import download

download("punkt")

corpus = [
    "This is a sentence",
    "This is another sentence!",
    "The third Sentence is here",
    "There are four sentences",
]

res = [word_tokenize(sent) for sent in corpus]

print(res)
