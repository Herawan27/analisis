import streamlit as st
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.utils import class_weight
import re

st.title("Evaluasi Model BERT - Analisis Sentimen Ulasan JMO Mobile")

# Upload file CSV
uploaded_file = st.file_uploader("Upload file CSV berlabel (dengan kolom 'content' dan 'label')", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Contoh data:")
    st.write(df.head())

    # Preprocessing sederhana
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df['text_clean'] = df['content'].apply(clean_text)

    # Label encoding
    label_mapping = {'positif': 0, 'negatif': 1, 'netral': 2}
    df['label_num'] = df['label'].map(label_mapping)

    st.write("Distribusi label:")
    st.bar_chart(df['label'].value_counts())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['label_num'], 
        test_size=0.2, stratify=df['label_num'], random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p1', num_labels=3)

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

    train_dataset = SentimentDataset(train_encodings, y_train.tolist())
    test_dataset = SentimentDataset(test_encodings, y_test.tolist())

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    with st.spinner("Melatih model..."):
        trainer.train()

    st.success("Pelatihan selesai!")

    # Evaluasi model
    predictions, true_labels, _ = trainer.predict(test_dataset)
    predictions = np.argmax(predictions, axis=1)

    st.write(f"**Akurasi**: {accuracy_score(true_labels, predictions)*100:.2f}%")
    st.write(f"**Recall**: {recall_score(true_labels, predictions, average='macro')*100:.2f}%")
    st.write(f"**Precision**: {precision_score(true_labels, predictions, average='macro')*100:.2f}%")
    st.write(f"**F1-score**: {f1_score(true_labels, predictions, average='macro')*100:.2f}%")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay.from_predictions(true_labels, predictions, display_labels=['Positif', 'Negatif', 'Netral'], ax=ax)
    disp.plot(ax=ax)
    st.pyplot(fig)

    # Classification report
    st.subheader("Classification Report")
    report = classification_report(true_labels, predictions, target_names=['Positif', 'Negatif', 'Netral'], zero_division=0)
    st.text(report)
