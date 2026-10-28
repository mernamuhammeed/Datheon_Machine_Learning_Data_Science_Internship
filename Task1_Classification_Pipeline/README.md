# Task 1 — Multi-Class Benchmark Classification Pipeline (Iris / Wine)

**Difficulty:** Beginner  
**Datasets:** Iris (user-supplied CSV) · Wine (sklearn built-in)

---

## 🎯 Objective

Build end-to-end classification pipelines with Scikit-Learn, covering data normalization, model benchmarking, and visual evaluation.

## 📋 Steps Implemented

1. **Load datasets** — Iris from `iris.csv`, Wine from `sklearn.datasets`
2. **EDA** — Seaborn pairplots with KDE diagonals for each class
3. **Preprocessing** — `StandardScaler` normalization + stratified 80/20 split
4. **Model training** — `LogisticRegression` and `DecisionTreeClassifier`
5. **Benchmarking** — Accuracy, Precision, Recall, F1-Score (weighted)
6. **Confusion matrices** — Normalized heatmaps for all model × dataset combos
7. **Decision boundaries** — PCA 2-D projection with mesh contour shading
8. **Tree structure** — Full `plot_tree` diagram (max_depth=4)
9. **Feature importances** — Gini importance horizontal bar charts

## 📊 Results

| Dataset | Model               | Accuracy | Precision | Recall | F1-Score |
|---------|---------------------|----------|-----------|--------|----------|
| Iris    | Logistic Regression | 0.9333   | 0.9333    | 0.9333 | 0.9333   |
| Iris    | Decision Tree       | 0.9333   | 0.9333    | 0.9333 | 0.9333   |
| Wine    | Logistic Regression | **0.9722** | **0.9741** | **0.9722** | **0.9722** |
| Wine    | Decision Tree       | 0.9444   | 0.9508    | 0.9444 | 0.9453   |

> **Winner:** Logistic Regression edges out Decision Tree on Wine (97.2% vs 94.4%), while both tie on Iris at 93.3%.

## 🖼️ Generated Visualizations

| File | Description |
|------|-------------|
| `iris_pairplot.png` | Seaborn pairplot — Iris features by class |
| `wine_pairplot.png` | Seaborn pairplot — top 5 Wine features |
| `confusion_matrices.png` | 2×2 normalized confusion matrix heatmaps |
| `metrics_comparison.png` | Side-by-side metric bar charts |
| `iris_decision_boundaries.png` | PCA 2-D decision boundaries — Iris |
| `wine_decision_boundaries.png` | PCA 2-D decision boundaries — Wine |
| `iris_decision_tree.png` | Decision Tree structure — Iris |
| `wine_decision_tree.png` | Decision Tree structure — Wine |
| `feature_importances.png` | Gini feature importances — both datasets |

## 📂 Files

```
Task1_Classification_Pipeline/
├── classification_pipeline.ipynb   # Main notebook (fully executed)
├── iris.csv                        # Iris dataset
├── iris_pairplot.png
├── wine_pairplot.png
├── confusion_matrices.png
├── metrics_comparison.png
├── iris_decision_boundaries.png
├── wine_decision_boundaries.png
├── iris_decision_tree.png
├── wine_decision_tree.png
└── feature_importances.png
```

## ▶️ Run

```bash
pip install scikit-learn pandas seaborn matplotlib jupyter
jupyter notebook classification_pipeline.ipynb
```
