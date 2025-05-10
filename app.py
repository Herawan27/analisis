import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split

# Set style
sns.set_style("whitegrid")

st.title("📊 Aplikasi Analisis Sentimen Menggunakan BERT")

# Upload file untuk seluruh aplikasi
file = st.file_uploader("Unggah file CSV", type=["csv"])

if file:
    # Membaca data dari file CSV
    df = pd.read_csv(file)
    st.subheader("🔍 Data yang Diunggah:")
    st.write(df)

    # === Processing dan Labeling ===
    st.header("⚙️ Processing dan Labeling Berdasarkan Rating (Star)")

    if st.button("🔧 Proses dan Labeling"):
        # Preprocessing
        df["content"] = df["content"].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)

        # Labeling berdasarkan rating
        def label_sentiment(score):
            if score >= 4:
                return "positif"
            elif score == 3:
                return "netral"
            else:
                return "negatif"
        
        df['label'] = df['star'].apply(label_sentiment)
        st.success("✅ Labeling selesai berdasarkan kolom 'star'")
        st.write(df)

        # Simpan hasil
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download File Berlabel", csv, "labeled_data.csv", "text/csv")

    # === Visualisasi Data ===
    st.header("📈 Visualisasi Sentimen")

    if "label" in df.columns and "content" in df.columns:
        st.subheader("📊 Distribusi Sentimen:")

        sentiment_counts = df["label"].value_counts().reindex(["positif", "netral", "negatif"]).fillna(0).astype(int)
        total = sentiment_counts.sum()
        sentiment_percent = (sentiment_counts / total * 100).round(2)

        distribusi_df = pd.DataFrame({
            "Jumlah": sentiment_counts,
            "Persentase (%)": sentiment_percent
        })
        distribusi_df.loc["Total"] = [total, 100.0]
        st.dataframe(distribusi_df)

        fig, ax = plt.subplots()
        ax.pie(sentiment_percent, labels=sentiment_percent.index, autopct='%1.1f%%',
               colors=["green", "orange", "red"], startangle=140)
        ax.axis('equal')
        st.subheader("📎 Diagram Lingkaran Sentimen")
        st.pyplot(fig)

        st.subheader("☁️ WordCloud per Sentimen")
        col1, col2, col3 = st.columns(3)
        for label, col, color in zip(["positif", "netral", "negatif"], [col1, col2, col3], ["Greens", "Oranges", "Reds"]):
            with col:
                text_data = " ".join(df[df["label"] == label]["content"].astype(str))
                if text_data.strip():
                    wordcloud = WordCloud(width=300, height=300, background_color="white", colormap=color).generate(text_data)
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.markdown(f"**{label.capitalize()}**")
                    st.pyplot(fig)
                else:
                    st.info(f"Tidak ada data untuk label: {label}")

        st.subheader('🔍 Contoh Ulasan dengan "would" atau "could"')
        keywords = df[df['content'].str.contains(r'\\b(would|could)\\b', case=False, na=False)]
        if not keywords.empty:
            for label in ["positif", "netral", "negatif"]:
                filtered = keywords[keywords["label"] == label]
                if not filtered.empty:
                    st.markdown(f"**{label.capitalize()}** ({len(filtered)} ditemukan):")
                    st.write(filtered[["content"]].head(3))
        else:
            st.info('Tidak ditemukan ulasan dengan kata "would" atau "could".')

    # === Evaluasi Model ===
    st.header("🧠 Evaluasi Model BERT")

    if "label" in df.columns and "content" in df.columns:
        # Label encoding: map 'positif' -> 1, 'negatif' -> 0, and 'netral' -> 2
        label_map = {'positif': 1, 'netral': 2, 'negatif': 0}
        df['label'] = df['label'].map(label_map)

        # Ensure that 'label' column contains integers
        assert df['label'].dtype == 'int64', "Labels should be integer values"

        st.write("Distribusi Data Sebelum Oversampling:")
        st.bar_chart(df["label"].value_counts())

        # Oversampling agar data seimbang
        label_counts = df['label'].value_counts()
        max_count = label_counts.max()

        df_oversampled = pd.DataFrame()
        for label in label_counts.index:
            df_label = df[df['label'] == label]
            df_label_upsampled = resample(df_label, replace=True, n_samples=max_count, random_state=42)
            df_oversampled = pd.concat([df_oversampled, df_label_upsampled])

        df_balanced = df_oversampled.sample(frac=1, random_state=42).reset_index(drop=True)

        st.write("Distribusi Data Setelah Oversampling:")
        st.bar_chart(df_balanced["label"].value_counts())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df_balanced["content"], df_balanced["label"], test_size=0.2, random_state=42)

        # Tokenizer dan encoding
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_encodings = tokenizer(list(X_train.astype(str)), truncation=True, padding=True, max_length=128)
        test_encodings = tokenizer(list(X_test.astype(str)), truncation=True, padding=True, max_length=128)

        class SentimentDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])  # Ensure labels are integers
                return item

        train_dataset = SentimentDataset(train_encodings, list(y_train))
        test_dataset = SentimentDataset(test_encodings, list(y_test))

        # Load pre-trained model
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # 3 classes: positive, neutral, negative

        # Set TrainingArguments
        training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="steps",  # Bisa ganti jadi "steps" atau "epoch"
)


        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        with st.spinner("Melatih model BERT..."):
            trainer.train()

        st.success("✅ Model telah dilatih!")

        # Evaluasi
        preds = trainer.predict(test_dataset)
        pred_labels = preds.predictions.argmax(axis=1)

        st.subheader("📊 Hasil Evaluasi:")
        st.text(classification_report(y_test, pred_labels))
        st.metric("🎯 Akurasi", f"{accuracy_score(y_test, pred_labels):.2f}")