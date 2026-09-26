# Classification Models — Iris Flower Classification

A comparison of three classic classification algorithms — Decision Tree, Logistic Regression (with cross-validation), and Support Vector Classifier — on the classic Iris flower dataset, plus a hyperparameter tuning pass on the best-performing model.

## Contents

| File | Description |
|---|---|
| `Classify Iris Flowers.ipynb` | Notebook containing data loading, model fitting, evaluation, and hyperparameter tuning. |
| `README.md` | This file. |

## Data

* **Iris dataset**, loaded via `sklearn.datasets.load_iris` (Fisher's classic dataset).
* 150 samples, 3 balanced classes (Setosa, Versicolor, Virginica), 50 samples each.
* 4 features: sepal length, sepal width, petal length, petal width.
* No missing values.

## How It Works

1. **Data prep**: Features (`X`) and target (`y`) are split 50/50 into train and test sets (`random_state=0`).
2. **Models trained and evaluated** (accuracy, confusion matrix, precision/recall/F1 per class):
   * **Decision Tree Classifier**
   * **Logistic Regression with cross-validation** (`LogisticRegressionCV`)
   * **Support Vector Classifier** (`SVC`)
3. **Model selection**: The Decision Tree is selected as the best performer, on both accuracy and F1 score.
4. **Hyperparameter tuning**: `RandomizedSearchCV` is used to search over `criterion`, `splitter`, `min_samples_leaf`, and `min_samples_split` for the Decision Tree.

## Results

| Model | Test Accuracy | Weighted F1 |
|---|---|---|
| **Decision Tree** | **96.0%** | **0.95** |
| Support Vector Classifier | 94.7% | 0.95 |
| Logistic Regression (CV) | 93.3% | 0.93 |

All three models perfectly separate the Setosa class (precision/recall/F1 = 1.00) — consistent with Setosa being linearly separable from the other two species. Most misclassifications occur between Versicolor and Virginica, which are known to overlap in feature space.

**Hyperparameter tuning**: `RandomizedSearchCV` found a best cross-validated training score of 96.0%, but the tuned model's test accuracy (93.3%) was lower than the original untuned Decision Tree's (96.0%). The original default Decision Tree was kept as the final model.

## Requirements

* Python 3
* `scikit-learn`

Install dependencies:
```
pip install scikit-learn
```

## Usage

1. Open and run the notebook:
   ```
   jupyter notebook "Classify Iris Flowers.ipynb"
   ```
2. The Iris dataset loads directly from scikit-learn — no external data files needed.
3. Run the cells in order to reproduce model training, evaluation, and tuning.

## Notes & Limitations

* The 50/50 train/test split is unusually large for the test set — with only 150 samples total, this leaves just 75 training examples, which makes results more sensitive to the specific split (`random_state=0`) than a more typical 70/30 or 80/20 split would be.
* Hyperparameter tuning did not improve on the default Decision Tree here, likely because the dataset is small and simple enough that the default settings already fit it well — the tuned model's cross-validated training score was similar, but it happened to generalize slightly worse on this particular test split.
* No k-fold cross-validation is used for the final model comparison (aside from `LogisticRegressionCV`'s internal CV) — a single train/test split means the reported scores could shift somewhat with a different `random_state`.
