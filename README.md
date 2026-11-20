# 🎓 Datheon Machine Learning & Data Science Internship

**Intern:** Merna Muhammad  
**Program:** Datheon Technical Internship — Machine Learning & Data Science Track  
**Submission:** Verified Certificate of Completion

---

## 📁 Repository Structure

| Task | Title | Status | Tech Stack |
|------|-------|--------|------------|
| [Task 1](./Task1_Classification_Pipeline/) | Multi-Class Benchmark Classification Pipeline (Iris / Wine) | ✅ Complete | Python · Scikit-Learn · Pandas · Seaborn · Matplotlib |
| [Task 2](./Task2/) | NLP Sentiment Analysis Classification Pipeline | ✅ Complete | Python · NLTK · Scikit-Learn · TF-IDF · Naive Bayes · Seaborn |
| [Task 3](./Task3/) | *(Coming Soon)* | 🔄 In Progress | — |

---

## 🗂️ Task Summaries

### ✅ Task 1 — Multi-Class Benchmark Classification Pipeline

> **Difficulty:** Beginner  
> **Dataset:** Iris (CSV) · Wine (sklearn)

Built a full end-to-end supervised classification pipeline covering:
- Exploratory Data Analysis with Seaborn pairplots
- Feature scaling with `StandardScaler`
- Stratified 80/20 train/test split
- `LogisticRegression` and `DecisionTreeClassifier` training
- Benchmark metrics: Accuracy · Precision · Recall · F1-Score
- Confusion matrix heatmaps
- PCA 2-D decision boundary visualization
- Decision Tree structure diagrams
- Feature importance (Gini) charts

**Results:**

| Dataset | Model               | Accuracy | F1-Score |
|---------|---------------------|----------|----------|
| Iris    | Logistic Regression | 0.9333   | 0.9333   |
| Iris    | Decision Tree       | 0.9333   | 0.9333   |
| Wine    | Logistic Regression | **0.9722** | **0.9722** |
| Wine    | Decision Tree       | 0.9444   | 0.9453   |

📂 [View Task 1 Details & Deliverables →](./Task1_Classification_Pipeline/)

---

### ✅ Task 2 — NLP Text Classification & Sentiment Pipeline

> **Difficulty:** Intermediate  
> **Dataset:** `sentiment_reviews.csv` (Customer & Movie Reviews)

Constructed a natural language processing text classifier covering:
- Text cleaning (HTML tag removal, lowercase, URL and punctuation stripping)
- Tokenization, stopword removal, and WordNet lemmatization with NLTK
- Numerical feature extraction using `TfidfVectorizer(max_features=5000, ngram_range=(1,2))`
- Supervised classification with `MultinomialNB` and `LogisticRegression`
- 5-Fold Stratified Cross-Validation benchmarking
- Visual diagnostics: class/token distributions, normalized confusion matrices, benchmark metrics, top TF-IDF features
- Interactive `predict_sentiment()` testing helper for custom user phrases with real-time probability confidence bars

**Results:**

| Model | Accuracy | Precision | Recall | F1-Score | 5-Fold CV F1 |
|-------|----------|-----------|--------|----------|--------------|
| Multinomial Naive Bayes | 0.6500 | 0.6291 | 0.6500 | 0.6291 | 0.5848 ± 0.0750 |
| Logistic Regression | **0.6500** | **0.6600** | **0.6500** | **0.6171** | **0.6074 ± 0.0788** |

📂 [View Task 2 Details & Deliverables →](./Task2/)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-3.9-green)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-4C72B0)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-11557c)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/mernamuhammeed/Datheon_Machine_Learning_Data_Science_Internship.git
cd Datheon_Machine_Learning_Data_Science_Internship

# Install dependencies
pip install scikit-learn pandas seaborn matplotlib nltk jupyter

# Launch notebooks
jupyter notebook Task1_Classification_Pipeline/classification_pipeline.ipynb
jupyter notebook Task2/sentiment_pipeline.ipynb
```

---

*Submitted to the Datheon Technical Directorate for certificate verification.*
