#LIBRARY
import streamlit as st
from google_play_scraper import Sort, reviews 
import pandas as pd
import numpy as np 
import re
import string
import nltk
nltk.download('punkt') 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from transformers import BertTokenizer, BertForSequenceClassification, Trainer,TrainingArguments
from collections import Counter
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, recall_score, precision_score, ConfusionMatrixDisplay, accuracy_score, classification_report

#SCRAPPING DATA
with st.form(key='my-form'):

    url= st.text_input('Enter Link Apps')
counts = st.number_input('amount of data', min_value=50, step=1) 
submit = st.form_submit_button('Submit')
if "submits" not in st.session_state: st.session_state.submits = False
def callback(): st.session_state.submits = False
if submit or st.session_state.submits: st.session_state.submits = True try:
 
    result, continuation_token = reviews( url,
lang='id', country='id', sort=Sort.NEWEST, count=counts,
filter_score_with=None,
)
result, _ = reviews( url,
continuation_token=continuation_token
)
df = pd.DataFrame(np.array(result), columns=['review']) df= df.join(pd.DataFrame(df.pop('review').tolist()))
df = df[['userName', 'score', 'at', 'content']]
df = df.copy().rename(columns={'score': 'star'})
# Menambahkan kolom 'year' dengan mengekstrak tahun dari kolom 'at' df['year'] = pd.to_datetime(df['at']).dt.year
st.dataframe(df) 
st.download_button(label='Download CSV',
data=df.to_csv(index=False, encoding='utf8'), file_name=url+'.csv', on_click=callback) except Exception as e:
st.error(f"An error occurred: {e}")

#VISUALISASI DATA
# Update: Handling 3 classes (positive, negative, neutral) def score_sentiment(score):
if score == 'positive': return 0
elif score == 'negative': return 1
else:
return 2
biner = df['sentiment'].apply(score_sentiment)
X_train, X_test, Y_train, Y_test = train_test_split(df['text_clean'], biner, test_size=0.2, stratify=biner, random_state=42)
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
# Update: Change num_labels to 3
 model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p1', num_labels=3)

train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=128)
class SentimentDataset(torch.utils.data.Dataset): def  init (self, encodings, labels):
self.encodings = encodings self.labels = labels
def   getitem  (self, idx):
item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
item['labels'] = torch.tensor(self.labels[idx]) return item
def   len  (self): return len(self.labels)
train_dataset = SentimentDataset(train_encodings, Y_train.tolist()) test_dataset = SentimentDataset(test_encodings, Y_test.tolist()) training_args = TrainingArguments(
output_dir='./results', num_train_epochs=3, per_device_train_batch_size=8, per_device_eval_batch_size=8, warmup_steps=500, weight_decay=0.01, logging_dir='./logs', logging_steps=10, evaluation_strategy="epoch"
)
trainer = Trainer( model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset
)
trainer.train()
predictions, labels, _ = trainer.predict(test_dataset) predictions = np.argmax(predictions, axis=1)
 
# Performance metrics
st.write("BERT Accuracy score -> ", accuracy_score(predictions, Y_test)*100)
st.write("BERT Recall Score	-> ", recall_score(predictions, Y_test, average='macro')*100)
st.write("BERT Precision score -> ", precision_score(predictions, Y_test, average='macro')*100)
st.write("BERT f1 score	-> ", f1_score(predictions, Y_test, average='macro')*100)
st.write("=========================================")
st.subheader("Confusion Matrix") fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay.from_predictions(Y_test, predictions,
 
ax=ax)
 

disp.plot(ax=ax) st.pyplot(fig)
st.write("=========================================")
st.text('classification report : \n' + classification_report(Y_test,
 
predictions, zero_division=0))
st.write("=========================================")
except Exception as e: st.write("Error:", e) raise e
if  name == " main ": main()
