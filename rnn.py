import pandas as pd
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
import nltk
import re
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

"""----------------------------------------------------
                       Definition
----------------------------------------------------"""
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
    
#----Setting GPU----#
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:',torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')
model = SentimentModel()
model.to(device)

#----loading data----#
print('Start loading data...')
data = pd.read_csv('/home/tim7107/input/training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
data.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
data = data.drop(['ids', 'date', 'flag', 'user'], axis = 1)

nltk.download('stopwords')
punc = list(string.punctuation)
stopword_list = stopwords.words('english') + punc + ['rt', 'via', '..', '...']
print(stopword_list)
stemmer = SnowballStemmer('english')
TEXT_CLEANING_RE = '@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+'
stop_words = stopwords.words('english')
    
print('Start Preprocessing...')
data.drop(['id','date','query','user'],axis=1,inplace=True)
data['target'] = data['target'].apply(lambda x: 0 if x == 0 else 1)
data['text'] = data['text'].apply(lambda x : remove_URL(x))
data['text'] = data['text'].apply(lambda x : remove_html(x))
data['text'] = data['text'].apply(lambda x : remove_emoji(x))
data['text'] = data['text'].apply(lambda x : remove_punct(x))
data.text = data.text.apply(process_text)

print('\t--- Before ---\n', data['text'][0])
print('\n\t--- After ---\n', sample_tweet)
data.text = data.text.apply(lambda x: preprocess(x))
data['target'] = data['target'] / 4


df_train, df_test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
print("TRAIN size:", len(df_train))
print("TEST size:", len(df_test))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train.text)
vocab_size = len(tokenizer.word_index) + 1
print("Total words", vocab_size)

x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=300)
x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=300)
y_train = df_train.target
y_test = df_test.target

print("X_TRAIN size:", len(x_train))
print("Y_TRAIN size:", len(y_train))
print("X_TEST size:", len(x_test))
print("Y_TEST size:", len(y_test))
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

max_len = 300
model = Sequential()
model.add(Embedding(20000, 300, input_length = max_len))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='rmsprop', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=1024,
                    validation_split=0.2)

