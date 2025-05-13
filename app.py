import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from imblearn.over_sampling import SMOTE  # Import SMOTE

# Fungsi utama
def main():
    st.set_page_config(layout="wide")
    st.title("📊 Analisis Sentimen Ulasan Aplikasi JMO Mobile")

    menu = st.sidebar.selectbox("Menu", ["Upload Data", "Processing & Labeling", "Visualisasi Data", "Evaluasi Model"])

    if "data" not in st.session_state:
        st.session_state.data = None
    if "data_labeled" not in st.session_state:
        st.session_state.data_labeled = None

    # Menu 1: Upload Data
    if menu == "Upload Data":
        st.header("Upload Data CSV")
        uploaded_file = st.file_uploader("Upload file CSV hasil scraping (kolom: content, star)", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.success("✅ Data berhasil di-upload.")
            st.write("📄 Preview Data:")
            st.dataframe(df)

    # Menu 2: Processing & Labeling
    elif menu == "Processing & Labeling":
        st.header("Processing & Labeling")
        if st.session_state.data is None:
            st.warning("⚠️ Silakan upload data terlebih dahulu.")
        else:
            if st.button("Mulai Processing & Labeling"):
                df = st.session_state.data.copy()
                df = df.rename(columns={'content': 'text_clean'})
                df['sentiment'] = df['star'].apply(lambda x: 'positive' if x > 3 else ('negative' if x < 3 else 'neutral'))
                st.session_state.data_labeled = df
                st.success("✅ Processing & Labeling selesai.")
                st.write("📄 Data Terproses:")
                st.dataframe(df[['text_clean', 'sentiment']].head())

    # Menu 3: Visualisasi Data
    elif menu == "Visualisasi Data":
        st.header("Visualisasi Data Sentimen")
        if st.session_state.data_labeled is None:
            st.warning("⚠️ Silakan lakukan processing terlebih dahulu.")
        else:
            if st.button("Tampilkan Visualisasi"):
                df = st.session_state.data_labeled

                # Tabel jumlah sentimen
                st.subheader("Jumlah Sentimen")
                count_df = df['sentiment'].value_counts().reset_index()
                count_df.columns = ['Sentimen', 'Jumlah']
                st.dataframe(count_df)

                # Pie chart
                fig1, ax1 = plt.subplots()
                ax1.pie(count_df['Jumlah'], labels=count_df['Sentimen'], autopct='%1.1f%%', startangle=90)
                ax1.axis('equal')
                st.pyplot(fig1)

                # Wordcloud per sentimen
                for sentimen in ['positive', 'neutral', 'negative']:
                    text = " ".join(df[df['sentiment'] == sentimen]['text_clean'].astype(str).tolist())
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                    st.subheader(f"Wordcloud Sentimen: {sentimen.capitalize()}")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)

    # Menu 4: Evaluasi Model
    elif menu == "Evaluasi Model":
        st.header("Evaluasi Model BERT")
        if st.session_state.data_labeled is None:
            st.warning("⚠️ Silakan lakukan processing terlebih dahulu.")
        else:
            if st.button("Mulai Training & Evaluasi Model BERT"):
                df = st.session_state.data_labeled.copy()

                # Label numerik
                def score_sentiment(label):
                    return 0 if label == 'positive' else (1 if label == 'negative' else 2)

                df['label'] = df['sentiment'].apply(score_sentiment)
                X = df['text_clean']
                y = df['label']

                # Split data tanpa oversampling
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.3,
                    stratify=y,
                    random_state=42
                )

                # SMOTE untuk menyeimbangkan kelas
                smote = SMOTE(random_state=42)
                X_train_res, y_train_res = smote.fit_resample(pd.DataFrame(X_train), y_train)
                X_train_res = X_train_res[0].values  # Menyesuaikan format X_train yang sebelumnya list

                # Tokenizer & model
                tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
                model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p1', num_labels=3)

                # Encoding
                train_enc = tokenizer(list(X_train_res), truncation=True, padding=True, max_length=128)
                test_enc = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

                class SentimentDataset(Dataset):
                    def __init__(self, encodings, labels):
                        self.encodings = encodings
                        self.labels = labels

                    def __getitem__(self, idx):
                        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                        item['labels'] = torch.tensor(self.labels[idx])
                        return item

                    def __len__(self):
                        return len(self.labels)

                train_dataset = SentimentDataset(train_enc, y_train_res)
                test_dataset = SentimentDataset(test_enc, y_test)

                # Training arguments
                training_args = TrainingArguments(
                    output_dir='./results',
                    num_train_epochs=3,
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=8,
                    warmup_steps=500,
                    weight_decay=0.01,
                    logging_dir='./logs',
                    logging_steps=10,
                    eval_strategy="epoch",
                    report_to="none"
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset
                )

                trainer.train()

                # Evaluasi
                preds, labels, _ = trainer.predict(test_dataset)
                y_pred = np.argmax(preds, axis=1)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
                rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

                st.success("✅ Evaluasi selesai.")
                st.write(f"**Akurasi:** {acc:.2%}")
                st.write(f"**Precision:** {prec:.2%}")
                st.write(f"**Recall:** {rec:.2%}")
                st.write(f"**F1 Score:** {f1:.2%}")

                # Confusion matrix
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
                st.pyplot(fig)

                # Classification report
                st.subheader("Classification Report")
                st.text(classification_report(y_test, y_pred, zero_division=0))

# Jalankan aplikasi
if __name__ == '__main__':
    main()
