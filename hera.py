import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def score_sentiment(score):
    if score == 'positive':
        return 0
    elif score == 'negative':
        return 1
    else:
        return 2

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

def main():
    st.title("Evaluasi Model BERT - Analisis Sentimen (3 Kelas)")

    uploaded_file = st.file_uploader("Upload file CSV yang sudah dilabeli dan dipreprocessing", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Konversi label ke format numerik
            biner = df['sentiment'].apply(score_sentiment)

            # Split data
            X_train, X_test, Y_train, Y_test = train_test_split(df['text_clean'], biner, test_size=0.2, stratify=biner, random_state=42)

            # Tokenisasi
            tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
            train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128)
            test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=128)

            # Dataset
            train_dataset = SentimentDataset(train_encodings, Y_train.tolist())
            test_dataset = SentimentDataset(test_encodings, Y_test.tolist())

            # Model
            model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p1', num_labels=3)

            # Training Arguments
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                evaluation_strategy="epoch"
            )

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset
            )

            # Train
            with st.spinner("Training model..."):
                trainer.train()

            # Predict
            predictions, labels, _ = trainer.predict(test_dataset)
            predictions = np.argmax(predictions, axis=1)

            # Evaluasi
            st.success("Evaluasi Model")
            st.write("**Accuracy:**", accuracy_score(Y_test, predictions) * 100)
            st.write("**Recall:**", recall_score(Y_test, predictions, average='macro') * 100)
            st.write("**Precision:**", precision_score(Y_test, predictions, average='macro') * 100)
            st.write("**F1 Score:**", f1_score(Y_test, predictions, average='macro') * 100)

            st.write("=========================================")
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(Y_test, predictions, ax=ax)
            st.pyplot(fig)

            st.write("=========================================")
            st.text('Classification Report:\n' + classification_report(Y_test, predictions, zero_division=0))
            st.write("=========================================")

        except Exception as e:
            st.error(f"Terjadi error: {e}")

if __name__ == "__main__":
    main()
