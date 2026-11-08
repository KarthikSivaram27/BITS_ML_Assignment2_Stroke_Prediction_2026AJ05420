# Stroke Prediction - ML Assignment 2

## Problem Statement
The goal of this project is to predict the likelihood of a patient suffering a stroke based on various health indicators such as age, hypertension, heart disease, and glucose levels. Five different classification models were implemented to determine the most reliable predictor.

## Dataset Description
- **Source:** Healthcare Dataset (Kaggle)
- **Instances:** > 500
- **Features:** 12+ (including age, gender, hypertension, heart disease, avg glucose level, bmi, smoking status, etc.)
- **Target:** `stroke` (Binary Classification: 0 = No, 1 = Yes)

## GitHub Repository Link
[https://github.com/KarthikSivaram27/BITS_ML_Assignment2_Stroke_Prediction_2026AJ05420](https://github.com/KarthikSivaram27/BITS_ML_Assignment2_Stroke_Prediction_2026AJ05420)

## Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 75.15% | 0.839 | 14.08% | 80.00% | 23.95% | 0.264 |
| Decision Tree | 92.27% | 0.523 | 10.81% | 8.00% | 9.20% | 0.0634 |
| KNN | 94.81% | 0.614 | 20.00% | 2.00% | 3.64% | 0.0491 |
| Naive Bayes | 86.79% | 0.803 | 16.54% | 42.00% | 23.73% | 0.203 |
| Support Vector Machine | 73.58% | 0.786 | 11.81% | 68.00% | 20.12% | 0.201 |

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Achieved the highest Recall (80.00%) and MCC (0.264) after balancing class weights, making it the most reliable for detecting stroke cases. |
| Decision Tree | Moderate accuracy but struggled with recall, showing a tendency to overfit the majority class. |
| KNN | High accuracy but very low recall (2.00%), indicating it failed to identify most stroke patients. |
| Naive Bayes | Provided balanced performance with a strong AUC (0.803) and reasonable recall (42.00%). |
| Support Vector Machine | Good recall (68.00%) but lower overall accuracy and MCC compared to Logistic Regression. |
| **Overall, Winner for your dataset?** | **Logistic Regression** |

