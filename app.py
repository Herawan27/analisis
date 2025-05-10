import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import string
from sklearn.utils import resample
from nltk.tokenize import TweetTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
import torch
import nltk

nltk.download('punkt')

# --- Inisialisasi ---
stopword_factory = StopWordRemoverFactory()
stopwords_list = stopword_factory.get_stop_words()
stemmer = StemmerFactory().create_stemmer()
tokenizer = TweetTokenizer()
bert_tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')

st.set_page_config(page_title="Analisis Sentimen JMO", layout="wide")

st.title("📱 Analisis Sentimen Ulasan Aplikasi JMO")

# --- Sidebar ---
menu = st.sidebar.radio("Pilih Menu", ["Upload Data", "Processing & Labeling", "Visualisasi Data", "Evaluasi Model"])

# --- Global Dataset State ---
if "data" not in st.session_state:
    st.session_state.data = None

# --- Upload Data ---
if menu == "Upload Data":
    uploaded_file = st.file_uploader("Upload file CSV hasil scraping", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
        st.success("✅ Data berhasil diupload!")
        st.write(df.head())

# --- Processing & Labeling ---
elif menu == "Processing & Labeling":
    if st.session_state.data is not None:
        df = st.session_state.data.copy()

        st.subheader("🔧 Label Sentimen")
        def label_sentiment(score):
            if score >= 4:
                return "positif"
            elif score == 3:
                return "netral"
            else:
                return "negatif"

        df['label'] = df['star'].apply(label_sentiment)

        st.subheader("🧹 Preprocessing")

        normalization_dict = {
            'gk': 'nggak', 'ga': 'nggak', 'tdk': 'tidak', 'bgt': 'banget',
            'sm': 'sama', 'aja': 'saja', 'udh': 'sudah', 'dr': 'dari',
        }

        def clean_text(text):
            text = text.lower()
            text = re.sub(r"http\S+|www\S+|https\S+", '', text)
            text = re.sub(r'\@\w+|\#','', text)
            text = re.sub(r'[^A-Za-z\s]', '', text)
            text = re.sub(r'\d+', '', text)
            text = text.translate(str.maketrans('', '', string.punctuation))
            return text

        def normalize_text(text):
            return ' '.join([normalization_dict.get(word, word) for word in text.split()])

        def full_preprocess(text):
            text = clean_text(text)
            text = normalize_text(text)
            tokens = tokenizer.tokenize(text)
            tokens = [word for word in tokens if word not in stopwords_list]
            return stemmer.stem(' '.join(tokens))

        df['content_stemmed'] = df['content'].astype(str).apply(full_preprocess)

        df['label_num'] = df['label'].map({'negatif': 0, 'netral': 1, 'positif': 2})

        st.session_state.data = df
        st.success("✅ Preprocessing & Labeling selesai")
        st.dataframe(df[['content', 'content_stemmed', 'label']].head())

        st.download_button("📥 Download Data Berlabel", df.to_csv(index=False), file_name="data_labeled.csv")
    else:
        st.warning("❗Silakan upload data terlebih dahulu di menu 'Upload Data'")

# --- Visualisasi Data ---
elif menu == "Visualisasi Data":
    if st.session_state.data is not None:
        df = st.session_state.data.copy()
        st.subheader("📊 Distribusi Sentimen")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='label', order=['positif', 'netral', 'negatif'], ax=ax)
        st.pyplot(fig)

        st.subheader("☁️ Wordcloud")
        selected_label = st.selectbox("Pilih label sentimen", ['positif', 'netral', 'negatif'])

        text = ' '.join(df[df['label'] == selected_label]['content_stemmed'])
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)

        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    else:
        st.warning("❗Silakan upload dan proses data terlebih dahulu.")

# --- Evaluasi Model ---
elif menu == "Evaluasi Model":
    if st.session_state.data is not None:
        df = st.session_state.data.copy()
        st.subheader("📈 Evaluasi Model BERT")

        # Oversampling
        df_major = df[df.label == 'positif']
        df_neg = df[df.label == 'negatif']
        df_net = df[df.label == 'netral']

        df_neg_up = resample(df_neg, replace=True, n_samples=len(df_major), random_state=42)
        df_net_up = resample(df_net, replace=True, n_samples=len(df_major), random_state=42)

        df_bal = pd.concat([df_major, df_neg_up, df_net_up])
        df_bal['label_num'] = df_bal['label'].map({'negatif': 0, 'netral': 1, 'positif': 2})

        X_train, X_val, y_train, y_val = train_test_split(
            df_bal['content_stemmed'], df_bal['label_num'], test_size=0.2, stratify=df_bal['label_num'], random_state=42
        )

        train_enc = bert_tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
        val_enc = bert_tokenizer(list(X_val), truncation=True, padding=True, max_length=128)

        class SentimentDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        train_dataset = SentimentDataset(train_enc, list(y_train))
        val_dataset = SentimentDataset(val_enc, list(y_val))

        model = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", num_labels=3)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(axis=-1)
            return {
                "accuracy": accuracy_score(labels, preds),
                "precision": precision_score(labels, preds, average='macro', zero_division=1),
                "recall": recall_score(labels, preds, average='macro', zero_division=1),
                "f1": f1_score(labels, preds, average='macro', zero_division=1)
            }

        args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_dir="./logs",
            do_eval=True,
            save_total_limit=1
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        with st.spinner("Training model..."):
            trainer.train()
            preds = trainer.predict(val_dataset)

        y_pred = preds.predictions.argmax(axis=-1)
        st.success("✅ Training selesai")

        st.subheader("📊 Metrics")
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='macro', zero_division=1)
        rec = recall_score(y_val, y_pred, average='macro', zero_division=1)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=1)

        st.write(f"**Accuracy**: {acc:.2%}")
        st.write(f"**Precision**: {prec:.2%}")
        st.write(f"**Recall**: {rec:.2%}")
        st.write(f"**F1 Score**: {f1:.2%}")

        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y_val, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negatif', 'Netral', 'Positif'],
                    yticklabels=['Negatif', 'Netral', 'Positif'], ax=ax_cm)
        st.pyplot(fig_cm)

    else:
        st.warning("❗Silakan upload dan proses data terlebih dahulu.")
