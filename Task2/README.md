# Task 2 — Natural Language Processing Text Classifier (Sentiment Analysis)

**Difficulty:** Intermediate  
**Dataset:** `sentiment_reviews.csv` (10 original benchmark reviews + 90 augmented reviews = 100 samples)  
**Track:** Datheon Machine Learning & Data Science Internship

---

## 🎯 Objective
Construct an end-to-end natural language processing (NLP) text classification pipeline to ingest customer reviews, clean and tokenize textual data, convert text into numerical TF-IDF feature matrices, and train supervised classifiers (Multinomial Naive Bayes & Logistic Regression) to identify sentiment polarity (`positive`, `negative`, `neutral`).

---

## 📋 Methodology & Pipeline Workflow

1. **Data Ingestion**: Ingest customer review records from `sentiment_reviews.csv`.
2. **Text Preprocessing**:
   - Stripping HTML markup & tags (`re.sub`)
   - Case folding / lowercasing
   - URL & non-alphabetical punctuation removal
   - Word tokenization (`nltk.tokenize.word_tokenize`)
   - Stopword filtering using NLTK English stopwords
   - Morphological lemmatization (`WordNetLemmatizer`)
3. **Exploratory Data Analysis (EDA)**:
   - Sentiment label distribution analysis
   - Post-cleaning review token length histograms
4. **Feature Extraction**:
   - `TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)`
   - Stratified 80/20 train/test split preserving class proportions
5. **Model Training & Benchmarking**:
   - **Multinomial Naive Bayes** (`MultinomialNB(alpha=0.5)`)
   - **Logistic Regression** (`LogisticRegression(max_iter=1000, solver='lbfgs')`)
   - 5-Fold Stratified Cross-Validation (`StratifiedKFold`)
6. **Evaluation & Diagnostics**:
   - Precision, Recall, F1-Score (weighted & macro), Accuracy
   - Normalized confusion matrix heatmaps
   - Top predictive TF-IDF feature log-probabilities per sentiment class
7. **Interactive Live Prediction Helper**:
   - Implemented `predict_sentiment(text, model, verbose=True)` with probability confidence bar visualizations for custom real-time inputs.

---

## 📊 Benchmark Results

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | 5-Fold CV F1 (Mean ± Std) |
|---|---|---|---|---|---|
| **Multinomial Naive Bayes** | 0.6500 | 0.6291 | 0.6500 | 0.6291 | 0.5848 ± 0.0750 |
| **Logistic Regression** | **0.6500** | **0.6600** | **0.6500** | **0.6171** | **0.6074 ± 0.0788** |

---

## 🖼️ Generated Visualizations

| Visualization | Description |
|---|---|
| `eda_distribution.png` | Class frequency balance & token length distribution |
| `confusion_matrices.png` | Normalized confusion matrix heatmaps for Naive Bayes and Logistic Regression |
| `metrics_comparison.png` | Side-by-side benchmark comparison bar charts across metrics |
| `top_features.png` | Top 15 influential TF-IDF n-grams per sentiment class |

---

## 💻 Interactive Live Test Helper (`predict_sentiment`)

```python
predict_sentiment("This product is absolutely amazing! I love everything about it.")
# Result : POSITIVE [+] (67.1% confident)
# positive   [################.........] 67.09%
# negative   [######...................] 27.85%
# neutral    [#........................] 5.06%
```

---

## 📂 Folder Structure

```
Task2/
├── sentiment_pipeline.ipynb    # Main Jupyter Notebook
├── sentiment_reviews.csv       # Raw dataset
├── eda_distribution.png        # Class & token distribution plot
├── confusion_matrices.png      # Confusion matrix heatmaps
├── metrics_comparison.png      # Performance benchmark comparison
├── top_features.png            # Top TF-IDF keywords per class
└── README.md                   # Task documentation
```

## 🚀 How to Run

```bash
pip install scikit-learn pandas seaborn matplotlib nltk jupyter
jupyter notebook sentiment_pipeline.ipynb
```
