import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

st.title("Evaluasi Model BERT - Analisis Sentimen")

uploaded_file = st.file_uploader("Upload file berlabel (hasil preprocessing & labeling)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Pastikan kolom label/sentimen ada
        if 'label' not in df.columns and 'sentiment' not in df.columns:
            st.error("Kolom 'label' atau 'sentiment' tidak ditemukan di data.")
            st.stop()

        # Fungsi konversi label 3 kelas
        def score_sentiment(score):
            if score == 'positive' or score == 0:
                return 0
            elif score == 'negative' or score == 1:
                return 1
            else:
                return 2

        label_column = 'label' if 'label' in df.columns else 'sentiment'
        biner = df[label_column].apply(score_sentiment)

        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(
            df['text_clean'], biner, test_size=0.2, stratify=biner, random_state=42)

        st.info("Memuat tokenizer dan model IndoBERT...")
        tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        model = BertForSequenceClassification.from_pretrained(
            'indobenchmark/indobert-base-p1', num_labels=3)

        # Tokenisasi
        train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128)
        test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=128)

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

        train_dataset = SentimentDataset(train_encodings, Y_train.tolist())
        test_dataset = SentimentDataset(test_encodings, Y_test.tolist())

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,  # ubah ke 3 jika ingin training lebih lama
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        st.info("Melatih model BERT...")
        trainer.train()

        st.success("Evaluasi dimulai...")
        predictions, labels, _ = trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=1)

        # METRIK
        st.write("### Skor Evaluasi")
        st.write("**Accuracy**: {:.2f}%".format(accuracy_score(labels, predictions) * 100))
        st.write("**Recall**: {:.2f}%".format(recall_score(labels, predictions, average='macro') * 100))
        st.write("**Precision**: {:.2f}%".format(precision_score(labels, predictions, average='macro') * 100))
        st.write("**F1 Score**: {:.2f}%".format(f1_score(labels, predictions, average='macro') * 100))

        # CONFUSION MATRIX
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(labels, predictions, ax=ax)
        st.pyplot(fig)

        # CLASSIFICATION REPORT
        st.write("### Classification Report")
        st.text(classification_report(labels, predictions, zero_division=0))

    except Exception as e:
        st.error(f"Terjadi error: {e}")
