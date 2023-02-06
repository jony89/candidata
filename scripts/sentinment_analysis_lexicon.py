import pandas as pd

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

#!pip install afinn
from afinn import Afinn

nltk.download("vader_lexicon")

# instantiate afinn
afn = Afinn()

# Initialize the VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

# Define the sentences to analyze
sentences = [
    "Awful experience. I would never buy this product again!",
    "Overall, I had a good time, even though I felt bad",
    "Not bad!!!",
    "Great service :-((((",
    "I'm so sorry for your loss",
]

results = []
for sentence in sentences:
    # AFINN
    afinn_polarity = afn.score(sentence)

    # VADER
    vader_scores = vader.polarity_scores(sentence)
    vader_polarity = vader_scores["compound"]

    # TextBlob
    textblob_polarity = TextBlob(sentence).sentiment.polarity

    # Apend results
    results.append(
        {
            "sentence": sentence,
            "AFINN": afinn_polarity,
            "VADER": vader_polarity,
            "TextBlob": textblob_polarity,
        }
    )

df = pd.DataFrame(results)
print(df)
