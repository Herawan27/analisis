import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.title("Analisis Sentimen Ulasan JMO Mobile (BERT, 3 Kelas)")

uploaded_file = st.file_uploader("Upload file dataset berlabel (CSV)", type=["csv"])
if uploaded_file:
    try:
        # Load Data
        df = pd.read_csv(uploaded_file)

        st.subheader("Contoh Data")
        st.dataframe(df.head())

        # Validasi kolom
        if 'text_clean' not in df.columns or 'label' not in df.columns:
            st.error("Dataset harus memiliki kolom 'text_clean' dan 'label'")
            st.stop()

        # Pembagian data
        X = df['text_clean']
        y = df['label'].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )

        # Load tokenizer dan model IndoBERT
        st.info("Memuat model IndoBERT dan tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        model = BertForSequenceClassification.from_pretrained(
            'indobenchmark/indobert-base-p1', num_labels=3
        )

        # Tokenisasi
        train_encodings = tokenizer(
            X_train.tolist(), truncation=True, padding=True, max_length=128
        )
        test_encodings = tokenizer(
            X_test.tolist(), truncation=True, padding=True, max_length=128
        )

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

        train_dataset = SentimentDataset(train_encodings, y_train.tolist())
        test_dataset = SentimentDataset(test_encodings, y_test.tolist())

        # Argument pelatihan
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="no"
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        st.info("Melatih model BERT... Harap tunggu (butuh waktu)")
        trainer.train()

        # Evaluasi
        st.success("Evaluasi Model")
        preds, labels, _ = trainer.predict(test_dataset)
        preds = np.argmax(preds, axis=1)

        st.write("**Accuracy:** {:.2f}%".format(accuracy_score(labels, preds) * 100))
        st.write("**Precision:** {:.2f}%".format(precision_score(labels, preds, average='macro') * 100))
        st.write("**Recall:** {:.2f}%".format(recall_score(labels, preds, average='macro') * 100))
        st.write("**F1 Score:** {:.2f}%".format(f1_score(labels, preds, average='macro') * 100))

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(labels, preds, ax=ax)
        st.pyplot(fig)

        # Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(labels, preds, zero_division=0))

    except Exception as e:
        st.error(f"Terjadi error: {e}")
