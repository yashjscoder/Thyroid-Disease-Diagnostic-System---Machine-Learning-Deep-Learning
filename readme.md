# Thyroid Disease Classification Model

## 1. Project Overview

This project focuses on building and evaluating machine learning classification models to predict thyroid disease status based on a publicly available medical dataset from the UCI Machine Learning Repository (Garavan Institute, Sydney). The goal is to establish a robust prediction baseline and identify the most critical clinical features driving the classification.

## 2. Technical Stack

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python (3.x) | Core programming language. |
| **Data Handling** | Pandas, NumPy | Data loading, manipulation, and numerical operations. |
| **Modeling** | Scikit-learn (sklearn) | Implementing Decision Tree and K-Nearest Neighbors classifiers. |
| **Visualization** | Matplotlib, Seaborn | Generating model comparison and feature importance plots. |
| **Environment** | Jupyter Notebook/IDE | Interactive development and execution. |

## 3. Data Source and Initial State

* **Dataset:** `thyroid.csv` (UCI Thyroid Disease Data Set)
* **Data Size (Raw):** 9,172 patient records.
* **Data Challenges:**
    * Missing values encoded as `?`.
    * Categorical features encoded as non-numeric strings (`t`/`f`, `F`/`M`, etc.).
    * Unlabeled columns (due to missing header).
    * Highly sparse/redundant measurement flag columns.

## 4. Machine Learning Pipeline

The project followed a standard, sequential machine learning pipeline, focusing heavily on robust data preparation to ensure clean inputs for the models. 

### 4.1. Data Cleaning and Preprocessing

| Step | Action Taken | Rationale |
| :--- | :--- | :--- |
| **Standardization** | Replaced all `?` values with `np.nan`. | Standardizing missing values for proper handling. |
| **Binary Encoding** | Converted boolean columns (`t`/`f`) to numerical (`1`/`0`). | Preparing categorical flags for machine learning algorithms. |
| **Sex Encoding & Imputation** | Encoded `F` as `2`, `M` as `1`. Imputed missing `sex` values using the **MODE** (most frequent category). | Correct statistical approach for imputing categorical features. |
| **Column Dropping** | Removed the six `*_measured` flag columns and the highly sparse `TBG` column. | The `*_measured` columns are redundant after imputation, and `TBG` had over 95% missing data. |
| **Feature/Target Split** | Separated the cleaned data into **Features (X)** and **Target (y)**. | Prepared data for supervised learning. |
| **Train/Test Split** | Split `X` and `y` into training and testing sets (e.g., 70/30 split). | Ensuring model performance is evaluated on unseen data. |

### 4.2. Model Training

Two different classification algorithms were chosen for baseline comparison:

1.  **Decision Tree Classifier:** Trained with a maximum depth of `3` to prioritize interpretability and prevent overfitting.
2.  **K-Nearest Neighbors (K-NN):** Trained with a default number of neighbors (often `k=5`) to provide a distance-based classification baseline.

## 5. Current Analysis and Results

### 5.1. Baseline Model Accuracy

| Model Algorithm | Hyperparameters | Accuracy Score (on Test Set) |
| :--- | :--- | :--- |
| **Decision Tree** | `max_depth=3` | **67.96%** |
| **K-Nearest Neighbors** | Default `k` | **61.91%** |

**Conclusion:** The Decision Tree performs better than the baseline K-NN model, indicating that a simple rule-based structure is effective on this dataset.

### 5.2. Feature Importance Analysis (Decision Tree)

The Decision Tree model provided significant insights into the predictive power of the features. 

| Rank | Feature Name | Relative Importance Score |
| :--- | :--- | :--- |
| **1** | **`psych`** | **0.495** |
| **2** | **`T3`** | **0.300** |
| **3** | **`on_thyroxine`** | **0.202** |
| 4+ | *All Others* | $\mathbf{\approx 0.000}$ |

**Crucial Finding: Zero Importance ($\mathbf{0.000}$) Features**

Features such as `age`, `sex`, `sick`, `pregnant`, and notably, **`on_antithyroid_medication`**, were assigned a score of $0.000$ by the Decision Tree. This does **not** mean they are medically irrelevant, but rather that the model found them useless for its classification task.

**Confirmed Reason for `on_antithyroid_medication` (and similar flags): Data Sparsity.**

* **Dataset Analysis:** The raw data for `on_antithyroid_medication` shows that **98.74%** of records are 'f' (False/No).
* **Model Impact:** The Decision Tree avoids this feature because it cannot find a balanced, effective split. Splitting $98.74\%$ of the data away from $1.26\%$ provides negligible gain in purity compared to splitting on highly variable features like `T3` or `psych`.

## 6. Next Steps (Model Optimization)

The next phase of the project is dedicated to hyperparameter tuning and model refinement to improve the current accuracy scores.

* **K-NN Tuning:** Identify the optimal number of neighbors ($k$) using a validation curve to maximize accuracy and close the gap with the Decision Tree.
* **Decision Tree Tuning:** Test various `max_depth` values to potentially increase performance without sacrificing generalization (avoiding overfitting).
* **Model Selection:** Retrain the best-performing algorithm (Decision Tree or tuned K-NN) with its optimal hyperparameters for the final result.