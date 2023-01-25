# First, you will need to import the necessary libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

df = pd.read_csv("./data/800_jobs__clean_no_hebrew.csv")


## barplot for most frequent words
from collections import Counter
import re
import pandas as pd
import nltk

nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

df = pd.read_csv("800_jobs__clean_no_hebrew.csv")
print(df)
# Extract the text column and preprocess it by lowercasing and removing punctuation
corpus = df["description"].apply(lambda x: str(x).lower())
corpus = corpus.apply(lambda x: re.sub(r"[^\w\s]", "", x))

# Tokenize the corpus and remove stop words
tokens = corpus.apply(nltk.word_tokenize)
filtered_tokens = tokens.apply(
    lambda x: [token for token in x if token not in stopwords.words("english")]
)

# Stem the filtered tokens
stemmer = PorterStemmer()
stemmed_tokens = filtered_tokens.apply(lambda x: [stemmer.stem(token) for token in x])

# Count the frequency of each stemmed token
tf = Counter(stemmed_tokens.sum())

# Extract the top 10 most frequent tokens
tf_top_10 = dict(tf.most_common(10))

# Extract the token names and frequencies
token_names = list(tf_top_10.keys())
token_counts = list(tf_top_10.values())

# Create a barplot of the term frequency
plt.figure(figsize=(20, 10))
plt.bar(token_names, token_counts)

# Add axis labels and a title
plt.xlabel("Token", fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel("Term frequency", fontsize=20)
plt.yticks(fontsize=20)
plt.title("Term frequency of top 10 stemmed tokens", fontsize=30)


# Display the plot
plt.show()

### create wordcloud
## filter for the 3 different buckets seperatly 


df.description = df["description"].apply(lambda x: str(x).lower())
df.description = df["description"].apply(lambda x: x.replace("product", ""))
df.description = df["description"].apply(lambda x: x.replace("team", ""))
df.description = df["description"].apply(lambda x: x.replace("will", ""))
df.description = df["description"].apply(lambda x: x.replace("work", ""))

filtered_df = df[(df["applications"] >= 101)]
# filtered_df = df[df["title"].str.contains("senior", case=False, na=False)]

# Extract the text column and preprocess it by lowercasing and removing punctuation
corpus = df["description"].apply(lambda x: str(x).lower())
corpus = corpus.apply(lambda x: re.sub(r"[^\w\s]", "", x))

# Tokenize the corpus and remove stop words
tokens = corpus.apply(nltk.word_tokenize)
filtered_tokens = tokens.apply(
    lambda x: [token for token in x if token not in stopwords.words("english")]
)

# Stem the filtered tokens
stemmer = PorterStemmer()
stemmed_tokens = filtered_tokens.apply(lambda x: [stemmer.stem(token) for token in x])

flat_tokens = [item for sublist in stemmed_tokens for item in sublist]
lowercase_tokens_cloud = " ".join(flat_tokens)

# Generate the wordcloud
wordcloud = WordCloud().generate(lowercase_tokens_cloud)

# Display the wordcloud
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud)


###### Bar plts jobs

# Clean the titles to avoid duplicates
df.title = df["title"].apply(lambda x: str(x).lower())
df.title = df["title"].apply(lambda x: x.replace("developer", "engineer"))
df.title = df["title"].apply(lambda x: x.replace("front end", "frontend"))


# calculate the ratio of number of jobs to number of applicants
df1 = (
    df.groupby("title")
    .agg(
        number_of_applicants=("applications", "sum"),
        number_of_jobs=("applications", "count"),
    )
    .reset_index()
)
df1["ratio"] = df1["number_of_applicants"] / df1["number_of_jobs"]


df_display = df1[["title", "ratio"]].sort_values("ratio", ascending=False).head(20)


# Create a barplot of the term frequency
plt.figure(figsize=(20, 10))
plt.bar(df_display.title, df_display.ratio)


# Add axis labels and a title
plt.xlabel("Sum", fontsize=20)
plt.xticks(rotation=75)
plt.xticks(fontsize=15)
plt.ylabel("Ratio of Applications to Number of listed Jobs", fontsize=20)
# plt.yticks(fontsize = 20)
plt.title("Most Poplular jobs by Number of applications per Job", fontsize=30)
