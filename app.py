import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score
from transformers import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set style visual
sns.set_style("whitegrid")

st.title("📊 Aplikasi Analisis Sentimen Menggunakan BERT")

file = st.file_uploader("Unggah file CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("🔍 Data yang Diunggah:")
    st.write(df)

    # === Processing dan Labeling ===
    st.header("⚙️ Processing dan Labeling Berdasarkan Rating (Star)")

    if st.button("🔧 Proses dan Labeling"):
        df["content"] = df["content"].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)

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

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download File Berlabel", csv, "labeled_data.csv", "text/csv")

    # === Visualisasi ===
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
        keywords = df[df['content'].str.contains(r'\b(would|could)\b', case=False, na=False)]
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
        label_map = {'positif': 1, 'netral': 2, 'negatif': 0}
        df['label'] = df['label'].map(label_map)

        st.write("Distribusi Data Sebelum Oversampling:")
        st.bar_chart(df["label"].value_counts())

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

        X_train, X_test, y_train, y_test = train_test_split(
            df_balanced["content"], df_balanced["label"], test_size=0.2, random_state=42)

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
                item['labels'] = torch.tensor(self.labels[idx])
                return item

        train_dataset = SentimentDataset(train_encodings, list(y_train))
        test_dataset = SentimentDataset(test_encodings, list(y_test))

        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

        # Deteksi device (GPU jika ada)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        # Optimizer
        optimizer = AdamW(model.parameters(), lr=5e-5)

        st.info("⏳ Melatih model BERT secara manual...")

        # Training loop manual
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        model.train()
        for epoch in range(3):  # num_train_epochs
            progress = st.progress(0, text=f"Epoch {epoch + 1}/3")
            for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                progress.progress((i + 1) / len(train_loader))

        st.success("✅ Model telah dilatih!")

        # Evaluasi
        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=64)
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())

        # Tampilkan hasil
        st.subheader("📊 Hasil Evaluasi Manual:")
        st.text(classification_report(all_labels, all_preds))
        st.metric("🎯 Akurasi", f"{accuracy_score(all_labels, all_preds):.2f}")
