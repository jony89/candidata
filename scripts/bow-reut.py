# imports
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import re
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# read in the csv file and drop any missing values
df = pd.read_csv('/Users/reuteliash/Downloads/800_jobs__clean_no_hebrew (1).csv').dropna()

# Create a new column 'applications_range'
df['applications_range'] = 'low'
df.loc[(df['applications'] >= 50) & (df['applications'] <= 100), 'applications_range'] = 'medium'
df.loc[df['applications'] > 100, 'applications_range'] = 'high'

# Preprocessing
def preprocess(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.lower()
    return text
df['text'] = df['title'] + ' ' + df['subTitle'] + ' ' + df['description']
df['text'] = df['text'].apply(preprocess)

# Create a bag of words model
cv = CountVectorizer(stop_words='english')
dtm = cv.fit_transform(df['text'])

# Multinomial Naive Bayes
# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(dtm, df['applications_range'], test_size=0.2, random_state=42)

# Initialize the model
nb = MultinomialNB()

# Fit the model on the training data
nb.fit(X_train, y_train)

# Make predictions on the test data
nb_predictions = nb.predict(X_test)

# Evaluate the model
print("Accuracy MNB: ", accuracy_score(y_test, nb_predictions))
print("Precision Score MNB: ", precision_score(y_test, nb_predictions, average='weighted'))
print("Recall Score MNB: ", recall_score(y_test, nb_predictions, average='weighted'))


#  Random Forest Classifier
# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(dtm, df['applications_range'], test_size=0.2, random_state=42)

# Initialize the model
rfc = RandomForestClassifier()

# Fit the model on the training data
rfc.fit(X_train, y_train)

# Make predictions on the test data
rfc_predictions = rfc.predict(X_test)

# Evaluate the model
print("Accuracy RF: ", accuracy_score(y_test, rfc_predictions))
print("Precision Score RF: ", precision_score(y_test, rfc_predictions, average='weighted'))
print("Recall Score RF: ", recall_score(y_test, rfc_predictions, average='weighted'))


# Logistic Regression
# Splitting the data into train and validation sets
train_X = dtm[:int(dtm.shape[0]*0.8)]
test_X = dtm[int(dtm.shape[0]*0.8):]
train_y = df['applications_range'].values[:int(dtm.shape[0]*0.8)]
test_y = df['applications_range'].values[int(dtm.shape[0]*0.8):]

# Training the model
model = LogisticRegression()
model.fit(train_X, train_y)

# Predicting on the validation set
lr_predictions = model.predict(test_X)

# Evaluation
print("Accuracy LR: ", accuracy_score(test_y, lr_predictions))
print("Precision Score LR: ", precision_score(test_y, lr_predictions, average='weighted'))
print("Recall Score LR: ", recall_score(test_y, lr_predictions, average='weighted'))