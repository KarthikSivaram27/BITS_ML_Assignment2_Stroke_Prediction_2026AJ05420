# Stroke Prediction - ML Assignment 2

**Author:** Karthik Sivaram  
**BITS ID:** 2026AJ05420  
**GitHub Repo:** [BITS_ML_Assignment2_Stroke_Prediction_2026AJ05420](https://github.com/KarthikSivaram27/BITS_ML_Assignment2_Stroke_Prediction_2026AJ05420)  
**Live App:** [Streamlit Link](https://bitsmlassignment2strokeprediction2026aj05420-9o4uljeqstuqvdaaq.streamlit.app/)

## Problem Statement
The goal of this project is to predict the likelihood of a patient suffering a stroke based on various health indicators such as age, hypertension, heart disease, and glucose levels. Five different classification models were implemented to determine the most reliable predictor.

## Dataset Description
- **Source:** Healthcare Dataset (Kaggle)
- **Instances:** > 500
- **Features:** 12+ (including age, gender, hypertension, heart disease, avg glucose level, bmi, smoking status, etc.)
- **Target:** `stroke` (Binary Classification: 0 = No, 1 = Yes)

## Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 75.15% | 0.839 | 14.08% | 80.00% | 23.95% | 0.264 |
| Decision Tree | 92.27% | 0.533 | 12.82% | 10.00% | 11.24% | 0.073 |
| KNN | 94.81% | 0.614 | 20.00% | 2.00% | 3.64% | 0.049 |
| Naive Bayes | 86.79% | 0.803 | 16.54% | 42.00% | 23.73% | 0.203 |
| Support Vector Machine | 73.58% | 0.786 | 11.81% | 68.00% | 20.12% | 0.201 |

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | **Winner.** Achieved the highest Recall (80.00%) and MCC (0.264). In medical diagnostics, maximizing Recall is critical to minimize False Negatives (missing a stroke), which would have catastrophic clinical consequences. |
| Decision Tree | High accuracy but low recall; tends to overfit the majority class (non-stroke patients). |
| KNN | Highest accuracy (94.81%) but lowest recall (2.00%), meaning it fails to identify almost all actual stroke cases. |
| Naive Bayes | Balanced performance with a strong AUC (0.803), but significantly lower recall than Logistic Regression. |
| Support Vector Machine | Good recall (68.00%) but lower overall MCC and accuracy compared to the winning model. |
| **Overall Winner** | **Logistic Regression** (Selected for its superior ability to capture positive stroke cases). |

