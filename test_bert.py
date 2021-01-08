"""----------------------------------------------------
                       Import
----------------------------------------------------"""
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import transformers
from transformers import BertModel, BertTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

"""----------------------------------------------------
                       Definition
----------------------------------------------------"""
class Parameter():
    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    BERT_PRETRAINED = 'bert-base-uncased'
    TOKENIZER = BertTokenizer.from_pretrained(BERT_PRETRAINED)
    BERT_HIDDEN_SIZE = 768
    MAX_TOKENS_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    EPOCHS = 3
    SEED = 10
    LEARNING_RATE = 4e-5
    TESTING_SIZE = 0.1 
    VALIDATION_SIZE = 0.05
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
    
class Custom_Data_Loader():
    """
       @param: 
             modifyed_data_form: train, test, validataion
    
       @return: input_ids,attention_mask,token_type_ids,target <long-tesnor type>
    """
    def __init__(self, modifyed_data_form):
        self.mdf = modifyed_data_form
    
    def __len__(self):
        return self.mdf.shape[0]
    
    def __getitem__(self, index_num):
        row = self.mdf.iloc[index_num]
        tweet = row['text']
        target = int(row['target'])
        
        tw_bert_tok = Parameter.TOKENIZER(tweet)
        
        tw_input_ids = tw_bert_tok['input_ids']
        tw_mask = tw_bert_tok['attention_mask']
        tw_tt_ids = tw_bert_tok['token_type_ids']
    
        len_ = len(tw_input_ids)
        if len_ > Parameter.MAX_TOKENS_LEN:
          tw_input_ids = tw_input_ids[:Parameter.MAX_TOKENS_LEN-1]+[102]
          tw_mask = tw_mask[:Parameter.MAX_TOKENS_LEN]
          tw_tt_ids = tw_tt_ids[:Parameter.MAX_TOKENS_LEN]
        elif len_ < Parameter.MAX_TOKENS_LEN:
          pad_len = Parameter.MAX_TOKENS_LEN - len_
          tw_input_ids = tw_input_ids + ([0] * pad_len)
          tw_mask = tw_mask + ([0] * pad_len)
          tw_tt_ids = tw_tt_ids + ([0] * pad_len)
        return {
            'input_ids':torch.tensor(tw_input_ids, dtype=torch.long),
            'attention_mask':torch.tensor(tw_mask, dtype=torch.long),
            'token_type_ids':torch.tensor(tw_tt_ids, dtype=torch.long),
            'target':torch.tensor(target, dtype=torch.long)
        }


class SentimentModel(nn.Module):
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained(Parameter.BERT_PRETRAINED)
        self.drop_out = nn.Dropout(0.30)
        self.linear1 = nn.Linear(Parameter.BERT_HIDDEN_SIZE, 2)
    
    def forward(self, input_ids, attention_mask, tt_ids):
        out_, pooled_out = self.bert(input_ids, attention_mask, tt_ids)
        out = self.drop_out(pooled_out)
        out = self.linear1(out)
        return out

def Testing(data_loader, model, device, inference=False):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            target = batch['target'].to(device)
            outputs = model(input_ids, attention_mask, token_type_ids)
            outputs = torch.argmax(outputs, dim=1).to('cpu').numpy()
    return int(outputs)

"""----------------------------------------------------
                       Testing
----------------------------------------------------""" 
#----Setting GPU----#
if torch.cuda.is_available():
    device = torch.device('cuda')
    # print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    # print('Device name:',torch.cuda.get_device_name(0))
else:
    # print('No GPU available, using the CPU instead.')
    device = torch.device('cpu')

#----Loading Model file to evaluate testing data----#
print('Starting to load model...')
saved_state = torch.load("SentimentModel5L.bin")
model = SentimentModel()
model.load_state_dict(saved_state)
model.to(device)

#----Testing----#
temp = input("Please input a sentence:\n")
dic = {
    'target': [0], 
    'text': [temp],
    'tweet_size': [12]
}
custom_testing_data = pd.DataFrame(dic)
print('Starting to predict...')
text = Custom_Data_Loader(custom_testing_data)
text_loader = DataLoader(text, batch_size=Parameter.VALID_BATCH_SIZE)
x = Testing(text_loader, model, device, inference=True)
print('Input :',temp)
if x==0:
    print('Prediction : Negative')
else:
    print('Prediction : Positive')
