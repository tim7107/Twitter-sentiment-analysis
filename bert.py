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
def Remove_stopwords(text):
    nltk_stopwords = nltk.corpus.stopwords.words("english")
    tokens= []
    for token in text.split():
        if token not in nltk_stopwords:
            tokens.append(token)
    return " ".join(tokens)

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
  
#--------------------------------------------------------    
class Parameter():
    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    BERT_PRETRAINED = 'bert-base-uncased'
    TOKENIZER = BertTokenizer.from_pretrained(BERT_PRETRAINED)
    EPOCHS = 5
    BERT_HIDDEN_SIZE = 768
    MAX_TOKENS_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    TESTING_SIZE = 0.1 
    VALIDATION_SIZE = 0.05
    SEED = 10
    
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
        
        bert_tokenizer = Parameter.TOKENIZER(tweet)
        
        tw_input_ids = bert_tokenizer['input_ids']
        tw_mask = bert_tokenizer['attention_mask']
        tw_tt_ids = bert_tokenizer['token_type_ids']
    
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

def Training(data_loader, model, optimizer, scheduler, device):
    epoch_loss = 0
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        target = batch['target'].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)

        batch_loss = Loss(outputs, target)
        # back propagation
        batch_loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_loss += batch_loss.item()
    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss

def Evaluating(data_loader, model, device, inference=False):
    epoch_loss = 0
    model.eval()
    results = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            target = batch['target'].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)

            batch_loss = Loss(outputs, target)
            epoch_loss += batch_loss.item()

            outputs = torch.argmax(outputs, dim=1).to('cpu').numpy()
            target = target.to('cpu').numpy()
            results.extend(list(zip(outputs, target)))
    epoch_loss = epoch_loss / len(data_loader)
    return epoch_loss, np.array(results)     

"""----------------------------------------------------
                       Training
----------------------------------------------------""" 
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
data.columns = ('target','id', 'date', 'query', 'user', 'text')

#----data preprocessing----#   
print('Start Preprocessing...')
data.drop(['id','date','query','user'],axis=1,inplace=True)
data['target'] = data['target'].apply(lambda x: 0 if x == 0 else 1)
data['text'] = data['text'].apply(lambda x : remove_URL(x))
data['text'] = data['text'].apply(lambda x : remove_html(x))
data['text'] = data['text'].apply(lambda x : remove_emoji(x))
data['text'] = data['text'].apply(lambda x : remove_punct(x))
data.text = data.text.apply(process_text)
data['text_length'] = data['text'].apply(lambda x:len(x.split()))

#----Create DataFrame----#
sent_df = pd.DataFrame(None, columns=('target', 'text','tweet_size'))
sent_df['target'] = data['target']
sent_df['text'] = data['text']
sent_df['tweet_size'] = data['text_length']
print(sent_df.head(3))
print(np.max(sent_df['tweet_size']))

#----randomly choose 250000 data from positive and negative----#
print('Start Sampling...')
Sampling_data = sent_df[(sent_df['tweet_size']>0) & (sent_df['target']==0)].sample(n=250000, random_state=Parameter.SEED)
Sampling_data = Sampling_data.append(sent_df[(sent_df['tweet_size']>0) & (sent_df['target']==1)].sample(n=250000, random_state=Parameter.SEED))
print(Sampling_data)
print(type(Sampling_data))

#----split trainind,testing data----#
print('Start Spliting training data & testing data...')
train, test = train_test_split(Sampling_data, test_size = Parameter.TESTING_SIZE)
train, val = train_test_split(train, test_size = Parameter.VALIDATION_SIZE)

#----Create dataloader----#
train_dl = Custom_Data_Loader(train)
val_dl = Custom_Data_Loader(val)
test_dl = Custom_Data_Loader(test)

train_loader = DataLoader(train_dl, batch_size=Parameter.TRAIN_BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(val_dl, batch_size=Parameter.VALID_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dl, batch_size=Parameter.VALID_BATCH_SIZE, shuffle=True)

#----Model----#
Loss = nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']  
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
# optimizer object
optim = AdamW(optimizer_grouped_parameters, lr=Parameter.LEARNING_RATE)
# learning rate scheduling
num_train_steps = int((train_dl.__len__()/Parameter.TRAIN_BATCH_SIZE)*Parameter.EPOCHS)
num_warmup_steps = int(0.05*num_train_steps)
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps, num_train_steps)

#----Training----#
print('Start Training...')
scores = []
high_acc = 0
for epoch in range(Parameter.EPOCHS):
  _ = Training(train_loader, model, optim, scheduler, device)
  _, results = Evaluating(validation_loader, model, device)
  validation_f1 = round(f1_score(results[:,1], results[:,0]),4)
  accuracy = round(accuracy_score(results[:,1], results[:,0]), 4)
  print('epoch num: ', epoch, 'f1 score: ',validation_f1 , 'accuracy: ', accuracy)
  scores.append((validation_f1, accuracy))

  if accuracy > high_acc :
    torch.save(model.state_dict(),"Sentiment_bert.bin")
    high_acc = accuracy

"""----------------------------------------------------
                       Testing
----------------------------------------------------""" 
#----Loading Model file to evaluate testing data----#
saved_state = torch.load("Sentiment_bert.bin")
model = SentimentModel()
model.load_state_dict(saved_state)
model.to(device)

print('Start Testing...')
_, results = Evaluating(test_loader, model, device, inference=True)
print(classification_report(results[:,1], results[:,0]))
print(round(accuracy_score(results[:,1], results[:,0]),4))