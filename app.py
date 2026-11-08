import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(page_title="Stroke Prediction App", layout="wide")

st.title("🧠 Stroke Prediction ML Application")
st.markdown("""
This app allows you to upload a test dataset and evaluate different Machine Learning models
trained on healthcare data to predict the likelihood of a stroke.
""")

# Load Preprocessing Artifacts
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('model/scaler.pkl')
    label_encoders = joblib.load('model/label_encoders.pkl')
    imputer = joblib.load('model/imputer.pkl')
    models = {
        'Logistic Regression': joblib.load('model/logistic_regression_model.pkl'),
        'Decision Tree': joblib.load('model/decision_tree_model.pkl'),
        'KNN': joblib.load('model/knn_model.pkl'),
        'Naive Bayes': joblib.load('model/naive_bayes_model.pkl'),
        'SVM': joblib.load('model/svm_model.pkl')
    }
    return scaler, label_encoders, imputer, models

try:
    scaler, label_encoders, imputer, models = load_artifacts()
    st.success("Models, Scaler, Encoders, and Imputer loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.info("Please ensure you have run the `model_training.ipynb` notebook first to generate the .pkl files in the /model folder.")
    st.stop()

# 1. Dataset Upload
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.write(df_test.head())

    # 2. Model Selection
    selected_model_name = st.sidebar.selectbox("Select ML Model", list(models.keys()))
    selected_model = models[selected_model_name]

    # Data Preprocessing for Prediction
    try:
        # Drop ID if present
        if 'id' in df_test.columns:
            df_processed = df_test.drop('id', axis=1)
        else:
            df_processed = df_test.copy()

        # 1. Handle Missing Values (Using Training Imputer for consistency)
        if 'bmi' in df_processed.columns:
            df_processed['bmi'] = pd.to_numeric(df_processed['bmi'], errors='coerce')
            # Transform using the saved imputer (expects 2D array)
            bmi_values = df_processed[['bmi']]
            df_processed['bmi'] = imputer.transform(bmi_values)

        # 2. Handle categorical columns (ensure they match training encoding)
        for col, le in label_encoders.items():
            if col in df_processed.columns:
                # Handle unseen labels by mapping them to the first known class or a default
                df_processed[col] = df_processed[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)

        # Separate features and target
        if 'stroke' in df_processed.columns:
            X_test = df_processed.drop('stroke', axis=1)
            y_test = df_processed['stroke']
        else:
            X_test = df_processed
            y_test = None

        # Scale features
        X_test_scaled = scaler.transform(X_test)

        # Predictions
        y_pred = selected_model.predict(X_test_scaled)

        # Handle probabilities for AUC
        if hasattr(selected_model, "predict_proba"):
            y_prob = selected_model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = None

        # 3. Display Evaluation Metrics
        if y_test is not None:
            st.subheader(f"Performance Metrics: {selected_model_name}")

            col1, col2, col3 = st.columns(3)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"

            col1.metric("Accuracy", f"{acc:.2%}")
            col2.metric("Precision", f"{prec:.2%}")
            col3.metric("Recall", f"{rec:.2%}")

            col4, col5, col6 = st.columns(3)
            col4.metric("F1 Score", f"{f1:.2%}")
            col5.metric("MCC", f"{mcc:.2f}")
            col6.metric("AUC Score", f"{auc:.2%}" if isinstance(auc, float) else auc)

            # 4. Confusion Matrix & Report
            st.markdown("---")
            st.subheader("Detailed Classification Report")

            c1, c2 = st.columns(2)

            with c1:
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))

            with c2:
                st.text("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)

        else:
            st.warning("The uploaded CSV does not contain a 'stroke' column. Displaying predictions only.")
            st.write("Predictions for the uploaded data:")
            st.write(pd.DataFrame({'Prediction': y_pred}))

    except Exception as e:
        st.error(f"Preprocessing Error: {e}")
        st.info("Ensure your uploaded CSV has the same columns and encoding as the training dataset.")

else:
    st.info("Please upload a CSV file from the sidebar to begin.")
