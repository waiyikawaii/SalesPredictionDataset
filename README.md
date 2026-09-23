# Sales Prediction Dataset
Linear regression project using the Advertising Sales dataset. Analyses how TV, Radio and Newspaper spending affects sales, builds a multiple linear regression model, compares raw and normalized data, and predicts sales for different ad budgets.

# Sales Prediction – Advertising Expenditure Analysis

A machine learning project using the Advertising dataset to analyse how TV, Radio and Newspaper spending affects sales. It uses linear regression to build a sales prediction model and compare different model settings.

---

## 📌 Overview

This project analyses the relationship between advertising spending and product sales using **linear regression**.

It covers data exploration, correlation analysis, multiple linear regression, model comparison between raw and normalized data, and sales prediction using new advertising budgets.

---

## 🎯 Problem Statement

This project aims to:

* Find the **average TV advertising spend**.
* Analyse the **correlation between Radio spending and Sales**.
* Identify which **advertising medium has the strongest relationship with Sales**.
* Build a **multiple linear regression model** using TV, Radio and Newspaper.
* Predict sales for a **new advertising budget**.
* Compare model performance using **raw and normalized data**.
* Analyse the model using **only Radio and Newspaper** as predictors.

---

## 📂 Dataset

* **Title:** Advertising Dataset
* **Format:** CSV
* **Rows:** 200
* **Columns:** 4

### Main Attributes

| Attribute   | Description                    |
| ----------- | ------------------------------ |
| `TV`        | Advertising spend on TV        |
| `Radio`     | Advertising spend on Radio     |
| `Newspaper` | Advertising spend on Newspaper |
| `Sales`     | Product sales                  |

---

## ❓ Case Questions

1. What is the average amount spent on TV advertising?
2. What is the correlation between Radio spending and Sales?
3. Which advertising medium has the strongest relationship with Sales?
4. Build a multiple linear regression model using TV, Radio and Newspaper and compare predicted vs. actual sales.
5. Predict sales for **$200 TV, $40 Radio and $50 Newspaper**.
6. How does model performance change after **normalization**?
7. What happens when using **only Radio and Newspaper** as predictors?

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** pandas, NumPy, matplotlib, seaborn, scikit-learn
* **Environment:** Jupyter Notebook / Python script

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python "Sales_Prediction.py"
```
