import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.title("Pelatihan Model BERT Akurasi Tinggi")

uploaded_file = st.file_uploader("Upload Dataset Berlabel", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Contoh Data")
        st.dataframe(df.head())

        if 'text_clean' not in df.columns or 'label' not in df.columns:
            st.error("Data harus memiliki kolom 'text_clean' dan 'label'")
            st.stop()

        # Ubah label jadi 0=pos, 1=neg, 2=neutral
        def encode_label(label):
            if label == 'positive' or label == 0:
                return 0
            elif label == 'negative' or label == 1:
                return 1
            else:
                return 2

        labels = df['label'].apply(encode_label)

        X_train, X_test, y_train, y_test = train_test_split(
            df['text_clean'], labels, test_size=0.15, stratify=labels, random_state=42)

        st.info("Memuat Tokenizer dan Model IndoBERT...")
        tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        model = BertForSequenceClassification.from_pretrained(
            'indobenchmark/indobert-base-p1', num_labels=3)

        train_enc = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128)
        test_enc = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=128)

        class IndoDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        train_ds = IndoDataset(train_enc, y_train.tolist())
        test_ds = IndoDataset(test_enc, y_test.tolist())

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

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
        )

        st.info("Training model BERT (bisa memakan waktu)...")
        trainer.train()

        st.success("Evaluasi Model")
        preds, labels, _ = trainer.predict(test_ds)
        preds = np.argmax(preds, axis=1)

        st.write("**Accuracy**: {:.2f}%".format(accuracy_score(labels, preds) * 100))
        st.write("**Precision**: {:.2f}%".format(precision_score(labels, preds, average='macro') * 100))
        st.write("**Recall**: {:.2f}%".format(recall_score(labels, preds, average='macro') * 100))
        st.write("**F1 Score**: {:.2f}%".format(f1_score(labels, preds, average='macro') * 100))

        st.write("### Classification Report")
        st.text(classification_report(labels, preds, zero_division=0))

        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(labels, preds, ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi error: {e}")
