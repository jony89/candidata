import gensim.downloader as api
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier

nltk.download("stopwords")

stop_words = stopwords.words("english")
# snowball_stemmer = nltk.SnowballStemmer("english")

tokenizer = nltk.RegexpTokenizer(r"\w+")


def remove_stopwords(sent):
    return [word for word in sent if word not in stop_words]


# we will not use stemming as it requires fine tuning. the existing pre-trained models are not familiar with the stemmed words.
# stemmer = PorterStemmer()


def remove_new_lines(sent):
    return sent.replace("\\n", " ")


# def word_stemming(sent):
#     return [stemmer.stem(word) for word in sent]


# Load CSV file with semicolon separator
df = pd.read_csv("./data/800_jobs__clean_no_hebrew.csv")


# Download pre-trained word2vec embeddings
model = api.load("word2vec-google-news-300")

# Prepare dataset
corpus = df["description"].tolist()

train_texts = [
    remove_stopwords(tokenizer.tokenize(remove_new_lines(sent.lower())))
    for sent in corpus
]
train_labels = df["applications"].tolist()

print("train_texts", train_texts)

# Convert text inputs to word embeddings
train_embeddings = []
for text in train_texts:
    text_embeddings = []
    for word in text:
        try:
            word_vector = model.word_vec(word)
            text_embeddings.append(word_vector)
        except Exception as e:
            text_embeddings.append([])
            print(f"Could not find word vector for {word}. skipping.")
    train_embeddings.append(text_embeddings)


# Train a classifier on the word embeddings
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(train_embeddings, train_labels)

# Predict class for new text inputs
new_text_inputs = """
Elsec (Elbit Systems subsidiary, located at Sderot) is developing the next generation of web based observation systems

We are currently looking for a highly skilled and motivated software engineer

As part of this job, you will take part in planning, designing, leading and implementing the frontend part of our products

You will have a direct influence on our technology stack as well as on the product roadmap

You will interact with system engineers, UX experts and developers on a daily basis


Bsc in software engineering or Computer science

Highly skilled front-end developer with 5+ years of experience in leading development of complex web-applications

Hands on experience with major frontend framework: Angular 2+ (preferred), React, Vue

Familiarity with State Management concepts and frameworks (Redux, NGRX, MobX) – a plus

Excellent UX intuition and skills.

Fluency in Typescript, Javascript , ES6 features, CSS3, SCSS

Deep understanding of core web development concepts

Team player

Experience with Restful API, SignalR – advantage.

Video & WebRTC – advantage
""".split(
    "\n"
)

new_text_inputs = [
    remove_stopwords(tokenizer.tokenize(remove_new_lines(sent.lower())))
    for sent in new_text_inputs
]

new_text_embeddings = []
for text in new_text_inputs:
    text_embeddings = []
    for word in text.split():
        text_embeddings.append(model.word_vec(word))
    new_text_embeddings.append(text_embeddings)

predictions = classifier.predict(new_text_embeddings)

print(f"----------- predictions: {predictions}")
