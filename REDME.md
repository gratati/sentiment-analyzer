# Анализ отзывов с кластеризацией и визуализацией

🧠 **Приложение для анализа тональности отзывов с использованием BERT и кластеризации методом KMeans**

---

## 🧩 Описание

Это веб-приложение на **Streamlit**, которое:
- Анализирует **тональность отзывов** (позитивная/негативная)
- Позволяет фильтровать отзывы по **дате**
- Делает **кластеризацию** по смыслу с помощью **BERT + KMeans**
- Визуализирует кластеры и строит **облака слов**
- Поддерживает анализ по **организациям** (по группам)

---

## 🛠️ Технологии

- **Streamlit** — интерфейс
- **BERT (DeepPavlov/rubert-base-cased)** — для получения эмбеддингов
- **KMeans** — кластеризация
- **PCA** — визуализация кластеров
- **WordCloud** — облака слов
- **scikit-learn** — классификация и кластеризация
- **transformers** — работа с BERT

---

## 📦 Установка и запуск

### 1. Клонируй репозиторий

```bash
git clone https://github.com/gratati/sentiment-analyzer.git
cd sentiment-analyzer

#2. Установи зависимости
pip install -r requirements.txt

#3. Убедись, что у тебя есть модели

Положи модели в папку model_files/:

vectorizer_tfidf.pkl
stacked_model_tfidf.pkl
final_sentiment_classifier.pkl
label_encoder.pkl
#4. Запусти приложение
streamlit run app.py

🚀 Деплой

Можно запустить на Streamlit Cloud или через Docker.

 📁 Структура проекта

 sentiment-analyzer/
├── app.py                  # Основное приложение
├── requirements.txt        # Зависимости
├── test.csv                # Тестовые данные
├── model_files/            # Модели ML
│   ├── vectorizer_tfidf.pkl
│   └── ...
