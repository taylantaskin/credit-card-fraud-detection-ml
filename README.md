# Credit Card Fraud Detection - End-to-End Machine Learning Project

**Türkiye Yapay Zeka Akademisi - Makine Öğrenmesi Final Ödevi**

## Project Goal

This project builds a complete machine learning pipeline to detect fraudulent credit card 
transactions. The goal is to classify each transaction as **Fraud (1)** or **Not Fraud (0)** 
using transaction details such as amount, authentication method, merchant category, and 
behavioral signals (velocity, device type, foreign transaction flags, etc).

## Dataset

- **Name:** Credit Card Fraud Detection 2026
- **Size:** 20,000 transactions, 26 original features
- **Target column:** `is_fraud` (binary classification)
- **Class distribution:** Highly imbalanced - only 1.7% of transactions are fraudulent

Unlike classic anonymized fraud datasets (e.g. PCA-transformed features like V1-V28), this 
dataset uses readable, real-world features (merchant category, device type, authentication 
method), making it well suited for exploratory analysis and explainability.

## Project Pipeline

The notebook follows a complete 17-step machine learning workflow:

1. Project setup and library imports
2. Data loading and problem definition
3. Target variable definition (binary classification)
4. Initial data inspection
5. Missing value check
6. Categorical variable identification
7. Outlier analysis
8. Feature scaling (StandardScaler)
9. Feature engineering (2 new features: `risk_flag_count`, `balance_to_amount_ratio`)
10. Feature selection (correlation-based)
11. Train / validation / test split (70/15/15, stratified)
12. Training 4 models: Logistic Regression, Decision Tree, Random Forest, KNN
13. Model comparison using precision, recall, F1-score, and confusion matrix
14. Hyperparameter tuning (GridSearchCV) on the top 2 models
15. Final evaluation on the held-out test set
16. Results interpretation
17. Explainability (Logistic Regression coefficient analysis)

## How to Run

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook hsd-veri-bilimi-ve-makine-renmesi.ipynb
4. Run all cells in order. The dataset file should be placed in the same directory as the 
   notebook, or update the `file_path` variable at the top of Step 2 to point to your local 
   copy of the CSV file.

## Results

The final model is a **tuned Logistic Regression** with `class_weight='balanced'`, evaluated 
on the test set:

| Metric | Score |
|---|---|
| Accuracy | 89.4% |
| Precision | 11.5% |
| Recall | 78.4% |
| F1-Score | 0.20 |

We prioritized **recall over precision**: in fraud detection, missing real fraud (a false 
negative) is usually far more costly than a false alarm, which can be quickly reviewed and 
dismissed by a human analyst. The final model catches 40 out of 51 fraud cases in the test set.

**Most important features:** authentication method (especially "No Authentication"), our 
engineered `risk_flag_count` feature, whether the merchant is new, merchant risk score, and 
transaction velocity were the strongest fraud indicators.

**Limitations:** the dataset is heavily imbalanced, precision is low (many false alarms), and 
the test set contains relatively few fraud cases (51), so results may vary somewhat with 
different data splits. More advanced techniques (e.g. SMOTE, ensemble methods) could likely 
improve results further.

## Files in this Repository

- `hsd-veri-bilimi-ve-makine-renmesi.ipynb` — main Jupyter notebook with the full pipeline
- `requirements.txt` — Python libraries needed to run the notebook
- `README.md` — this file
