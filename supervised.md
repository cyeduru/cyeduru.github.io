---
layout: default
title: "Supervised Learning"
parent: "AI & ML"
nav_order: 1
---

# Supervised Learning

Supervised Learning is a category of Machine Learning where the model is trained using **labeled data**.  
Each training example consists of:
- **Input features (X)**
- **Correct output label (Y)**

The goal is to learn a function `f(X) → Y` that can accurately predict outputs for unseen data.

---

## 🔹 Types of Supervised Learning
1. **Classification** – Output is categorical  
   - Example: Fraud / Not Fraud
2. **Regression** – Output is continuous  
   - Example: House price prediction

---

## Logistic Regression


::contentReference[oaicite:0]{index=0}


### 📌 What is Logistic Regression?
Logistic Regression is a **classification algorithm** used to predict the **probability** of a binary outcome (0 or 1).

Despite its name, it is **not a regression algorithm** — it is used for **classification**.

---

### 📐 Mathematical Concept
Logistic Regression uses the **Sigmoid (Logistic) Function**:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Where:
\[
z = wX + b
\]

- Converts any real number into a value between **0 and 1**
- Output is interpreted as **probability**

---

### 🧠 How Logistic Regression Works
1. Compute linear combination of inputs  
2. Apply sigmoid function  
3. Set a threshold (commonly 0.5)
4. Classify output as:
   - `1` if probability ≥ threshold
   - `0` otherwise

---

### 📊 Cost Function
Uses **Log Loss (Binary Cross-Entropy)**:

\[
Loss = -[y \log(p) + (1-y)\log(1-p)]
\]

Minimized using **Gradient Descent**

---

### ✅ Advantages
- Simple and easy to implement
- Fast training
- Works well for linearly separable data
- Outputs probabilistic interpretation
- Good baseline model

---

### ❌ Disadvantages
- Assumes linear decision boundary
- Poor performance on complex relationships
- Sensitive to outliers
- Requires feature engineering
- Not suitable for non-linear data without transformation

---

### 📍 Use Cases
- Fraud detection
- Credit scoring
- Medical diagnosis
- Spam detection
- Customer churn prediction

---

## XGBoost (Extreme Gradient Boosting)


::contentReference[oaicite:1]{index=1}


### 📌 What is XGBoost?
XGBoost is an **advanced ensemble learning algorithm** based on **Gradient Boosting**.  
It builds **multiple decision trees sequentially**, where each new tree corrects the errors of the previous ones.

---

### 🧠 Core Concept: Boosting
- Weak learners (shallow trees)
- Each tree focuses on **previous mistakes**
- Predictions are **added together**

Final Prediction:
\[
\hat{y} = \sum_{i=1}^{n} tree_i(x)
\]

---

### ⚙️ Key Innovations in XGBoost
- Regularization (L1 & L2)
- Parallel tree construction
- Handling missing values
- Tree pruning
- Shrinkage (learning rate)
- Column & row subsampling

---

### 📐 Objective Function
\[
Objective = Loss + Regularization
\]

Regularization helps prevent **overfitting**

---

### 🌲 Important Hyperparameters
- `n_estimators` – number of trees
- `max_depth` – tree depth
- `learning_rate` – contribution of each tree
- `subsample` – row sampling
- `colsample_bytree` – feature sampling
- `lambda`, `alpha` – regularization terms

---

### ✅ Advantages
- Excellent performance on tabular data
- Handles non-linear relationships
- Robust to outliers
- Built-in feature importance
- Works with missing values
- Highly scalable and fast

---

### ❌ Disadvantages
- Computationally expensive
- Requires careful hyperparameter tuning
- Less interpretable than linear models
- Can overfit if not tuned properly

---

### 📍 Use Cases
- Fraud detection
- Credit risk modeling
- Customer churn
- Ranking & recommendation systems
- Kaggle competitions

---

## Logistic Regression vs XGBoost

| Feature | Logistic Regression | XGBoost |
|------|------------------|--------|
| Model Type | Linear | Non-Linear |
| Interpretability | High | Medium |
| Speed | Very Fast | Slower |
| Handles Complexity | ❌ | ✅ |
| Feature Engineering | Required | Less Required |
| Overfitting Control | Limited | Strong |

---

## 📌 When to Use What?
- **Use Logistic Regression**:
  - Simple problems
  - Need explainability
  - Regulatory environments (AML, Credit)

- **Use XGBoost**:
  - Complex relationships
  - High accuracy required
  - Large structured datasets

---

## 📚 Summary
- Logistic Regression is a **baseline classifier**
- XGBoost is a **state-of-the-art ensemble method**
- Start simple → move complex as needed

---

➡️ *Next: Decision Trees & Random Forests*
