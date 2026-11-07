<h1 align="center">💳 Fraud Detection using a Neural Network</h1>

<p align="center">
  <b>Kaggle Playground Series – Season 6, Episode 11</b><br>
  Predicting fraudulent transactions with a Deep Neural Network (TensorFlow / Keras)
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/Validation%20AUC-0.9403-brightgreen">
</p>

---

## 📌 Overview

This project builds an end-to-end **binary classifier** that predicts whether a card transaction is fraudulent (`is_fraud`). The data comes from the Kaggle competition
[Playground Series S6E11](https://www.kaggle.com/competitions/playground-series-s6e11/data), and submissions are judged on **ROC AUC**.

**Pipeline:** Data download → EDA → Preprocessing → Neural network → Evaluation → Submission

---

## 📂 Dataset

| File | Rows | Description |
| :--- | :--- | :--- |
| `train.csv` | 699,960 | Labelled transactions (26 columns) |
| `test.csv` | 299,984 | Unlabelled transactions to predict |
| `sample_submission.csv` | 299,984 | Required submission format |

**Features (25 + target):**

- **Numerical:** `amount_usd`, `hours_since_last_txn`, `txn_count_last_24h`, `distance_from_home_km`, `card_age_months`, `customer_age`, `account_balance_usd`, `cvv_retry_count`, `velocity_score`, `time_of_day_hour`, `day_of_week`, `merchant_risk_score`, `prior_disputes`
- **Categorical:** `merchant_category`, `card_type`, `auth_method`, `channel`, `device_type`
- **Boolean:** `is_foreign_transaction`, `is_new_merchant`, `used_vpn`, `ip_country_mismatch`, `billing_shipping_mismatch`, `is_ai_generated_scam_attempt`
- **Target:** `is_fraud` (0 = legitimate, 1 = fraud)

The data has **no missing values** and is **highly imbalanced** (fraud is the rare class).

### Categorical feature distributions

---

## ⚙️ Preprocessing

All steps are wrapped in a single scikit-learn `ColumnTransformer`:

| Step | Applied to | Why |
| :--- | :--- | :--- |
| Drop | `id` | Identifier, not a feature |
| `StandardScaler` | Numerical columns | Mean 0, variance 1 for stable NN training |
| `OneHotEncoder` | Categorical + boolean columns | Avoids implying a fake order between categories |

After encoding, each transaction becomes a vector of **50 input features**.
Data is split **80 / 20** (stratified, `random_state=42`) into train and validation sets.

---

## 🧠 Neural Network

A feed-forward (fully connected) network that gets narrower layer by layer to learn a compact representation of fraud patterns.



| Component | Choice | Purpose |
| :--- | :--- | :--- |
| Hidden layers | 256 → 128 → 64 units, ReLU | Learn non-linear feature interactions |
| Initializer | He Normal | Suited to ReLU activations |
| Batch Normalization | After every hidden layer | Faster, more stable training |
| Dropout | 0.4 → 0.3 → 0.2 | Reduces overfitting |
| Output | 1 unit, Sigmoid | Fraud probability between 0 and 1 |
| Loss | Binary Cross-Entropy | Standard for binary classification |
| Optimizer | Adam, learning rate 0.001 | Adaptive gradient descent |
| Class weights | `balanced` (≈ 0.51 for class 0, ≈ 29.94 for class 1) | Penalises missed fraud far more than false alarms |
| Early stopping | Monitor `val_auc`, patience 5, restore best weights | Stops when validation AUC stops improving |
| Training | Batch size 2048, up to 50 epochs | Efficient on ~700k rows |

---

## 📈 Results

Training stopped early at **epoch 16** (best weights restored).

| Metric | Score |
| :---: | :---: |
| **Validation ROC AUC** | **0.9403** |

---

## 🚀 How to Run

1. Clone the repository and open `Neural_Network_Fraud_Detection.ipynb` (Kaggle, Colab, or local Jupyter).
2. Install the requirements:
   Download train, test, sample file from kaggle website - https://www.kaggle.com/competitions/playground-series-s6e11/data
   or
   Log in to Kaggle when prompted (`kagglehub.login()`); the notebook downloads the competition data automatically.
5. Run all cells. The final cell writes **`NN2_submission.csv`** (`id`, `is_fraud` probability) ready for submission.

---
