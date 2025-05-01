# 🦠 COVID-19 Data Analysis & Custom Decision Tree Classifier

This project implements a custom Decision Tree classifier in Python to analyze a COVID-19 dataset. The goal is to compare the performance of the custom model to the Scikit-learn implementation **without using one-hot encoding**, by **manually handling multi-label categorical features**.

---

## 📊 Dataset

- The dataset used is the **COVID-19 Dataset** from Kaggle.
- It includes patient-level data and various categorical features.

> 🔗 [Kaggle Dataset Link](https://www.kaggle.com/datasets/meirnizri/covid19-dataset)

---

## 🧪 Task Description

- Implement a **Decision Tree classification algorithm** from scratch in Python.
- Handle **multi-label categorical features manually** without one-hot encoding.
- Compare the results with Scikit-learn’s `DecisionTreeClassifier`.
- Evaluate performance using **F1-score**.

---

## 🛠️ Implementation Details

- Manual implementation of decision tree splitting based on categorical feature frequency and purity.
- Avoids one-hot encoding to simulate realistic limitations.
- Calculates F1-score for both the custom model and Scikit-learn’s version.
- Compares and prints results side-by-side.

---

## 🧑‍💻 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/ARASwithH/Covid_Data_Analysis.git
cd Covid_Data_Analysis
```
