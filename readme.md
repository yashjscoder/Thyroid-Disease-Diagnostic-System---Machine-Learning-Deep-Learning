# Thyroid Disease Classification Model

### https://thyroid-ai-diagnostic.streamlit.app/

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

## 2.B Screenshots

![alt text](<Screenshot 2025-12-28 042326.png>)![alt text](<Screenshot 2025-12-28 042408.png>) ![alt text](<Screenshot 2025-12-28 042425.png>) ![alt text](<Screenshot 2025-12-28 042414.png>) 

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


---

## 7. Phase 2: Advanced Modeling & Deployment

After establishing the baseline, the project was scaled up using a gradient boosting framework and a production-ready interface.

### 7.1. Model Upgrade (XGBoost)

To improve upon the 67% accuracy of the Decision Tree, the model was upgraded to **XGBoost (Extreme Gradient Boosting)**.

* **New Algorithm:** XGBoost Classifier.
* **Feature Engineering:** Expanded feature set to 20 clinical markers, including scaled numerical values for TSH, T3, TT4, T4U, and FTI.
* **Result:** The model achieved significantly higher precision and recall compared to the baseline models.

### 7.2. Interactive Web Application

A professional-grade web interface was developed to allow healthcare providers to interact with the model in real-time.

| Feature | Description |
| --- | --- |
| **Framework** | **Streamlit** (Python-based web framework). |
| **Real-time Inputs** | 12+ clinical inputs including age, sex, and laboratory blood levels (TSH, T3, etc.). |
| **Instant Diagnosis** | Automated prediction with a visual confidence score (Probability percentage). |
| **Report Generation** | Integration with **ReportLab** to generate a downloadable PDF Medical Report for patients. |

### 7.3. Project Structure

The final deployment includes the following core files:

* `main.py`: The Streamlit web application logic.
* `thyroid_model.json`: The trained XGBoost model "brain."
* `scaler.joblib`: The numerical scaler used to normalize patient data.
* `requirements.txt`: List of dependencies (XGBoost, Streamlit, ReportLab, etc.).

## 8. Usage

To run the diagnostic system locally:

1. Install dependencies: `pip install -r requirements.txt`
2. Launch the app: `streamlit run main.py`

---

### **What I added and why:**

* **XGBoost Section:** It explains why you moved away from the simple Decision Tree (to get better results).
* **Web Application Table:** It highlights your skills in **Full-Stack Data Science** (Model + UI + PDF generation).
* **Structure & Usage:** This tells anyone visiting your GitHub exactly how to use your project.

This is the perfect way to wrap up the project. Adding **Phase 3** shows that you didn't just build a model that gives a "Yes/No" answer, but one that is **transparent and explainable**.

Add this section to the very bottom of your `README.md`. It explains the "Feature Importance" chart and why it makes your app professional.

---

## 9. Phase 3: Explainable AI (XAI) & Transparency

The final phase focused on transforming the "Black Box" model into an **Explainable AI** system. This ensures that every diagnosis can be audited and understood by a medical professional.

### 9.1. Dynamic Feature Importance Mapping

The application now generates a real-time **Feature Importance Chart** for every diagnosis using the XGBoost `gain` metric.

| Metric | Description |
| --- | --- |
| **Feature Gain** | Measures the "Information Value" each clinical marker contributes to the final decision. |
| **Top Drivers** | Automatically identifies and ranks the top 10 markers (e.g., TSH, T3, Psych) influencing the current result. |
| **Visualization** | Built using `Matplotlib` and `Seaborn` for high-clarity data storytelling. |

### 9.2. How to Interpret the Diagnostic Chart

The chart serves as a "Reasoning Map" for the AI's decision:

* **The X-Axis (Gain):** Represents how much accuracy was gained by looking at that feature. Longer bars indicate that the AI relied heavily on that specific laboratory value.
* **The Y-Axis (Clinical Markers):** Lists the patient's data points. If **TSH** is at the top, it confirms the AI is following standard medical protocols where TSH is the primary indicator of thyroid health.
* **The "Why" Factor:** This chart answers the patient's question: *"Why did the AI give me this result?"* It shows exactly which part of their blood work or medical history triggered the diagnosis.

### 9.3. Technical Implementation

* **Logic:** Integrated XGBoost’s `get_score(importance_type='gain')` directly into the Streamlit UI.
* **Library Update:** Added `matplotlib` and `seaborn` to the production environment for on-the-fly rendering.

---

### **Summary of your Project Evolution:**

* **Phase 1:** Baseline models (Decision Tree/K-NN) and data cleaning.
* **Phase 2:** Upgrading to XGBoost, creating the Web App, and PDF reporting.
* **Phase 3:** Implementing **Explainable AI (XAI)** to show diagnostic reasoning.

---

