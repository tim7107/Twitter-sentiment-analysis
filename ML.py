import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
sns.set()
from nltk.stem.porter import *
import nltk
import re
import string
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def Remove_stopwords(text):
    nltk_stopwords = nltk.corpus.stopwords.words("english")
    tokens= []
    for token in text.split():
        if token not in nltk_stopwords:
            tokens.append(token)
    return " ".join(tokens)
#--------------------------------------------------------    
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)
hashtags = re.compile(r"^#\S+|\s#\S+")
mentions = re.compile(r"^@\S+|\s@\S+")
urls = re.compile(r"https?://\S+")    

def process_text(text):
  text = hashtags.sub(' hashtag', text)
  text = mentions.sub(' entity', text)
  return text.strip().lower()  
  
print('Start loading data...')
train  = pd.read_csv('/home/tim7107/input/training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
train .columns = ['label', 'id', 'date', 'query', 'user', 'text']

print('Start data preprocessing...')
train.drop(['id', 'date', 'query', 'user'], inplace=True, axis=1)
train['label'] = train['label'].apply(lambda x: 0 if x == 0 else 1)
train['text'] = train['text'].apply(lambda x : remove_URL(x))
train['text'] = train['text'].apply(lambda x : remove_html(x))
train['text'] = train['text'].apply(lambda x : remove_emoji(x))
train['text'] = train['text'].apply(lambda x : remove_punct(x))
train.text = train.text.apply(process_text)

bow_vectorizer = CountVectorizer(stop_words='english')
bow = bow_vectorizer.fit_transform(train['text'])

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(train['text'])

X_train_bow, X_val_bow, y_train_bow, y_val_bow = train_test_split(bow, train['label'], test_size=0.2)
X_train_tfidf, X_val_tfidf, y_train_tfidf, y_val_tfidf = train_test_split(tfidf, train['label'], test_size=0.2)

def evaluate_model(model, title):
    model.fit(X_train_bow, y_train_bow)
    y_pred_bow = model.predict(X_val_bow)
    print("{} - Bag-of-Words accuracy: {}".format(title, accuracy_score(y_val_bow, y_pred_bow)))
    
    model.fit(X_train_tfidf, y_train_tfidf)
    y_pred_tfidf = model.predict(X_val_tfidf)
    print("{} - TF-IDF accuracy: {}".format(title, accuracy_score(y_val_tfidf, y_pred_tfidf)))

print('Start NAIVE Classfier...')
MNB = MultinomialNB()
evaluate_model(MNB, "Naive Bayes")

print('Start Logistic Regression...')
lr = LogisticRegression()
evaluate_model(lr, "Logistic Regression")

print('Start Decision Tree...')
tree = DecisionTreeClassifier()
evaluate_model(tree, "Decision Tree")

print('Start Random Forest...')
rf = RandomForestClassifier()
evaluate_model(rf, "Random Forest")