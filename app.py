import streamlit as st
import joblib
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import os

# === Пути ===
MODEL_DIR = "/content/model_files/"

# === Загрузка моделей ===
@st.cache_resource
def load_models():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer_tfidf.pkl"))
    stacked_model = joblib.load(os.path.join(MODEL_DIR, "stacked_model_tfidf.pkl"))
    final_model = joblib.load(os.path.join(MODEL_DIR, "final_sentiment_classifier.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return vectorizer, stacked_model, final_model, label_encoder

@st.cache_resource
def load_bert():
    MODEL_NAME = "DeepPavlov/rubert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    bert_model = BertModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    bert_model.eval()
    return tokenizer, bert_model, device

vectorizer, stacked_model, final_model, label_encoder = load_models()
tokenizer, bert_model, device = load_bert()

# === Эмбеддинги BERT ===
def get_bert_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    tokens = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = bert_model(**tokens)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings.cpu().numpy()

# === Предсказание ===
def predict_sentiment(texts):
    input_is_str = isinstance(texts, str)
    if input_is_str:
        texts = [texts]

    X_tfidf = vectorizer.transform(texts)
    X_stack = stacked_model.predict_proba(X_tfidf)
    X_bert = get_bert_embeddings(texts)
    X_final = np.hstack([X_stack, X_bert])
    preds = final_model.predict(X_final)
    probs = final_model.predict_proba(X_final)

    pred_labels = label_encoder.inverse_transform(preds)
    confidence_values = np.max(probs, axis=1) * 100
    levels = ["Низкая" if conf < 65 else "Средняя" if conf < 85 else "Высокая" for conf in confidence_values]

    if input_is_str:
        return pred_labels[0], confidence_values[0], levels[0]
    else:
        return pred_labels, confidence_values, levels

# === Загрузка данных ===
@st.cache_data
def load_test_data():
    try:
        df = pd.read_csv("test.csv")
        if "review_text" in df.columns and "date" in df.columns and "group_name" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors='coerce')
            return df
        else:
            st.error("Ожидается наличие колонок 'review_text', 'date', 'group_name' в test.csv.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Ошибка при загрузке test.csv: {e}")
        return pd.DataFrame()

df_test = load_test_data()

# === Интерфейс ===
st.set_page_config(page_title="Анализ отзывов", layout="wide")
st.title("🧠 Анализ тональности + Кластеризация")
st.markdown("**Модель:** TF-IDF + Stacking + BERT")

# === Ввод и анализ одного отзыва ===
st.markdown("---")
st.header("📝 Анализ одного отзыва")

with st.form("sentiment_form"):
    text_input = st.text_area("Введите отзыв:", height=150, key="review_input")
    col1, col2 = st.columns([1, 1])
    with col1:
        analyze_button = st.form_submit_button("Анализировать")
    with col2:
        clear_button = st.form_submit_button("Очистить")

if clear_button:
    st.session_state.pop("review_input", None)
    st.session_state.pop("sentiment_result", None)

if analyze_button:
    if text_input.strip() == "":
        st.warning("Введите текст.")
    else:
        with st.spinner("Анализируем..."):
            sentiment, confidence, level = predict_sentiment(text_input)
            result = {
                "sentiment": sentiment,
                "confidence": confidence,
                "level": level
            }
            st.session_state["sentiment_result"] = result

if "sentiment_result" in st.session_state:
    result = st.session_state["sentiment_result"]
    st.markdown(f"### 🔍 Тональность: **{result['sentiment']}**")
    st.markdown(f"**Уверенность:** {result['level']} ({result['confidence']:.2f}%)")
    st.progress(int(result['confidence']))

    if st.button("🗑️ Скрыть результат анализа"):
        st.session_state.pop("sentiment_result", None)

# === Фильтрация по дате и организации ===
st.markdown("---")
st.header("📅 Фильтр по дате и организации")

if not df_test.empty:
    min_date = df_test["date"].min().date()
    max_date = df_test["date"].max().date()
    unique_groups = df_test["group_name"].unique().tolist()
else:
    min_date = max_date = None
    unique_groups = []

date_range = st.date_input("Выберите диапазон дат", value=(min_date, max_date) if min_date and max_date else ())

selected_group = st.selectbox("Выберите организацию", ["Все организации"] + unique_groups)

# === Применение фильтров ===
filtered_df = df_test.copy()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[(filtered_df["date"].dt.date >= start_date) & (filtered_df["date"].dt.date <= end_date)]

if selected_group != "Все организации":
    filtered_df = filtered_df[filtered_df["group_name"] == selected_group]

# === Анализ выборки ===
st.markdown("---")
st.header("📊 Анализ выборки")

num_samples = st.slider(
    "Количество отзывов для анализа",
    min_value=10,
    max_value=len(filtered_df) if not filtered_df.empty else 10,
    value=min(50, len(filtered_df)) if not filtered_df.empty else 10
)

if not filtered_df.empty and st.button("📅 Сделать предсказания"):
    samples = filtered_df["review_text"].sample(min(num_samples, len(filtered_df))).tolist()
    labels, confidences, _ = predict_sentiment(samples)
    results = []

    X_tfidf = vectorizer.transform(samples)
    X_stack = stacked_model.predict_proba(X_tfidf)
    X_bert = get_bert_embeddings(samples)
    X_final = np.hstack([X_stack, X_bert])
    probs = final_model.predict_proba(X_final)

    try:
        pos_class_idx = list(label_encoder.classes_).index("positive")
    except ValueError:
        try:
            pos_class_idx = list(label_encoder.classes_).index("Positive")
        except ValueError:
            pos_class_idx = 0

    positive_probs = probs[:, pos_class_idx]

    for i, text in enumerate(samples):
        results.append({
            "Текст": text[:100] + "...",
            "Предсказанная метка": labels[i],
            "Уверенность (%)": round(confidences[i], 2),
            "Вероятность": round(positive_probs[i], 2)
        })

    df_results = pd.DataFrame(results)
    st.session_state["df_results"] = df_results
    st.session_state["show_results"] = True

if "df_results" in st.session_state and st.session_state.get("show_results", False):
    df_results = st.session_state["df_results"]

    st.dataframe(df_results)

    st.subheader("📈 Распределение вероятностей")
    st.markdown(
        "График ниже показывает **распределение вероятностей принадлежности отзывов к положительному классу**. "
        "Если пики ближе к 0.5, модель часто сомневается; если ближе к 0 или 1 — она уверена."
    )
    plt.figure(figsize=(6, 4))
    sns.histplot(df_results["Вероятность"], bins=10, kde=True, color="green")
    plt.xlabel("Вероятность положительного класса")
    plt.title("Распределение вероятностей")
    st.pyplot(plt)

    st.subheader("📊 Распределение по тональности")
    tone_counts = df_results["Предсказанная метка"].value_counts().reset_index()
    tone_counts.columns = ["Тональность", "Количество"]

    color_map = {
        "positive": "#2CA02C",
        "negative": "#D62728",
        "neutral": "#FF7F0E",
        "Positive": "#2CA02C",
        "Negative": "#D62728",
        "Neutral": "#FF7F0E"
    }

    fig_tone = px.bar(
        tone_counts,
        x="Тональность",
        y="Количество",
        color="Тональность",
        title="Количество отзывов по каждой тональности",
        color_discrete_map=color_map
    )
    st.plotly_chart(fig_tone)

    st.subheader("🔍 Низкая уверенность")
    low_conf_df = df_results[df_results["Уверенность (%)"] < 65]
    if not low_conf_df.empty:
        st.markdown(f"Найдено {len(low_conf_df)} неуверенных отзывов:")
        st.dataframe(low_conf_df)
        st.download_button(
            label="📅 Скачать неуверенные",
            data=low_conf_df.to_csv(index=False).encode("utf-8"),
            file_name="low_conf_reviews.csv",
            mime="text/csv"
        )
    else:
        st.success("Все предсказания выше 65% уверенности!")

    st.download_button(
        label="📁 Скачать все результаты",
        data=df_results.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv"
    )

    if st.button("🗑️ Скрыть результаты выборки"):
        st.session_state["show_results"] = False

# === Визуализация кластеров (PCA) ===
st.markdown("---")
st.header("📊 Визуализация кластеров (PCA)")

if not filtered_df.empty:
    if st.button("🖼️ Построить 2D-визуализацию"):
        texts = filtered_df["review_text"].sample(min(300, len(filtered_df))).tolist()
        labels, _, _ = predict_sentiment(texts)

        embeddings = get_bert_embeddings(texts)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)

        df_viz = pd.DataFrame({
            "X": reduced[:, 0],
            "Y": reduced[:, 1],
            "Тональность": labels,
            "Текст": [t[:80] + "..." for t in texts]
        })

        fig = px.scatter(
            df_viz,
            x="X",
            y="Y",
            color="Тональность",
            color_discrete_map=color_map,
            hover_data=["Текст"],
            title="Кластеризация отзывов (PCA 2D)",
            labels={"X": "Первая компонента", "Y": "Вторая компонента"}
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))

        st.session_state["pca_fig"] = fig
        st.session_state["pca_explained"] = sum(pca.explained_variance_ratio_)

    if "pca_fig" in st.session_state:
        st.plotly_chart(st.session_state["pca_fig"], key="pca_plot")
        st.markdown(f"🔍 PCA объясняет: {st.session_state['pca_explained']:.1%} дисперсии")
        st.markdown(
            "Данный график отображает **отзывы в 2D-пространстве**, уменьшенном с помощью PCA.\n"
            "Точки сгруппированы по предсказанной тональности. Он помогает **визуально понять** различие между отзывами."
        )

# === Облака слов по кластерам ===
st.markdown("---")
st.header("☁️ Облака слов по кластерам")

if not filtered_df.empty and st.button("🎨 Построить облака слов"):
    texts = filtered_df["review_text"].sample(min(300, len(filtered_df))).tolist()
    labels, _, _ = predict_sentiment(texts)

    for i, tone in enumerate(set(labels)):
        tone_texts = [texts[j] for j in range(len(texts)) if labels[j] == tone]
        full_text = " ".join(tone_texts).lower()

        if full_text.strip():
            words = [word for word in full_text.split() if len(word) > 3 and word.isalpha()]
            word_freq = Counter(words).most_common(5)

            wc = WordCloud(width=800, height=300, background_color="white", colormap="viridis").generate(full_text)

            st.markdown(f"#### 🌟 {tone}")
            st.markdown("**Топ-5 слов:**")
            for word, count in word_freq:
                st.markdown(f"- **{word.capitalize()}** — {count} раз")

            plt.figure(figsize=(10, 4))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt, clear_figure=True)
        else:
            st.info(f"🚫 Нет данных для кластера: {tone}")

