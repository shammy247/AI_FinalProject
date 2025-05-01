import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import shap
from PIL import Image

# Load model and assets
model = joblib.load("random_forest_model.pkl")
explainer = shap.TreeExplainer(model)
validation_report = joblib.load("validation_report.pkl")
test_report = joblib.load("test_report.pkl")
val_cm = joblib.load("val_cm.pkl")
test_cm = joblib.load("test_cm.pkl")
learning_img = Image.open("learning_curves.png")

feature_names = ['Age', 'Gender', 'BMI', 'SBP', 'DBP', 'FPG',
                 'Chol', 'Tri', 'HDL', 'LDL', 'ALT', 'BUN',
                 'CCR', 'FFPG', 'smoking', 'drinking', 'family_history']

# Page config
st.set_page_config(page_title="Diabetes Predictor", layout="wide")

# Sidebar inputs
st.sidebar.header("🩺 Input Parameters")
age = st.sidebar.slider("Age", 18, 100, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 22.0)
sbp = st.sidebar.slider("SBP", 80, 200, 120)
dbp = st.sidebar.slider("DBP", 40, 130, 80)
fpg = st.sidebar.slider("FPG", 50.0, 300.0, 90.0)
chol = st.sidebar.slider("Cholesterol", 50.0, 400.0, 180.0)
tri = st.sidebar.slider("Triglycerides", 30.0, 300.0, 100.0)
hdl = st.sidebar.slider("HDL", 20.0, 100.0, 50.0)
ldl = st.sidebar.slider("LDL", 30.0, 300.0, 100.0)
alt = st.sidebar.slider("ALT", 5.0, 150.0, 20.0)
bun = st.sidebar.slider("BUN", 3.0, 30.0, 12.0)
ccr = st.sidebar.slider("CCR", 30.0, 150.0, 80.0)
ffpg = st.sidebar.slider("Follow-up FPG", 50.0, 300.0, 100.0)
smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
drinking = st.sidebar.selectbox("Drinking", ["No", "Yes"])
family_history = st.sidebar.selectbox("Family History", ["No", "Yes"])

# Main title
st.title("🧠 Diabetes Probability Predictor")
st.write("Enter the values in the sidebar and click Predict.")

# Process inputs
gender_enc = 1 if gender == "Male" else 0
smoking_enc = 1.0 if smoking == "Yes" else 0.0
drinking_enc = 1.0 if drinking == "Yes" else 0.0
family_enc = 1 if family_history == "Yes" else 0

features = np.array([[age, gender_enc, bmi, sbp, dbp, fpg,
                      chol, tri, hdl, ldl, alt, bun, ccr, ffpg,
                      smoking_enc, drinking_enc, family_enc]])

# Prediction button
if st.sidebar.button("🔍 Predict"):
    # Model prediction
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🩺 Prediction", value="Diabetic" if pred == 1 else "Not Diabetic")
    with col2:
        st.metric(label="📈 Probability", value=f"{prob*100:.2f}%")

    # Patient Risk Category
    st.markdown("### 🏷 Risk Category")
    if prob < 0.2:
        st.success("Low Risk")
        st.write("Your calculated probability indicates a low risk of diabetes.")
    elif prob < 0.5:
        st.warning("Moderate Risk")
        st.write("Your calculated probability indicates a moderate risk. Consider lifestyle changes and regular check-ups.")
    else:
        st.error("High Risk")
        st.write("Your calculated probability indicates a high risk of diabetes. Please consult a healthcare professional.")

    # Input summary
    st.subheader("🔎 Input Summary")
    input_df = pd.DataFrame(features, columns=feature_names)
    st.table(input_df)

    # Health tip
    st.markdown("### 💡 Health Tip")
    if pred == 1:
        st.warning("Consider visiting a healthcare professional. Maintain a balanced diet, regular exercise, and monitor your glucose levels.")
    else:
        st.info("Great job! Keep up your healthy lifestyle and continue routine monitoring.")

    # SHAP explanation in expander
    with st.expander("🔬 SHAP Feature Explanation"):
        try:
            features_df = pd.DataFrame(features, columns=feature_names)
            shap_values = explainer.shap_values(features_df)
            arr = np.array(shap_values[1]) if isinstance(shap_values, list) else np.array(shap_values)
            if arr.ndim == 3: arr = arr[0,:,1]
            values = arr[0] if arr.ndim == 2 else arr
            shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP': values})
            shap_df = shap_df.sort_values(by='SHAP', key=abs, ascending=True)
            fig, ax = plt.subplots(figsize=(6,5))
            ax.barh(shap_df['Feature'], shap_df['SHAP'], color='skyblue')
            ax.set_title('Feature Contributions')
            st.pyplot(fig)
            st.markdown("*Interpretation:* Features with positive SHAP values pushed the prediction towards 'Diabetic', while negative values pushed towards 'Not Diabetic'. The longer the bar, the stronger the influence.")
            top_features = shap_df.sort_values(by='SHAP', key=abs, ascending=False).head(3)
            st.markdown(f"**Top Influential Features for this Prediction:** {', '.join(top_features['Feature'])}")
        except Exception as e:
            st.error(f"SHAP error: {e}")

    # Global insights in expanders
    with st.expander("📊 Global Feature Importance"):
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
        imp_df = imp_df.sort_values(by='Importance', ascending=False)
        fig = px.bar(data_frame=imp_df, x='Importance', y='Feature', orientation='h',
                     title='Global Feature Importance')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig)
        st.markdown("*Interpretation:* This chart shows how much each feature contributes on average to model decisions across all samples. Higher importance means the feature is more influential.")

    with st.expander("📈 Model Performance & Context"):
        st.subheader("Classification Reports")
        st.json({"Validation": validation_report, "Test": test_report})
        st.markdown(
            """
            *Interpretation:*
            - The *Validation Report* shows precision, recall, and F1-score on the held-out validation set, indicating how well the model generalizes during tuning.
            - The *Test Report* shows the final performance on unseen data, reflecting real-world expected accuracy.
            """
        )
        c1, c2 = st.columns(2)
        with c1:
            st.write("*Validation Set*")
            fig_v, ax_v = plt.subplots()
            sns.heatmap(val_cm, annot=True, fmt='d', cmap='Greens', ax=ax_v)
            st.pyplot(fig_v)
        with c2:
            st.write("*Test Set*")
            fig_t, ax_t = plt.subplots()
            sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=ax_t)
            st.pyplot(fig_t)

        st.image(learning_img, use_column_width=True)

        # Learning Curve Interpretation
        st.markdown("### 📉 Learning Curves Interpretation")
        st.markdown("""
        The learning curves below demonstrate how model performance improves with increasing training data:

        - **Accuracy**: Training accuracy remains very high (near 100%) throughout, while test accuracy steadily increases from ~70% to ~95%.  
          📌 *This reflects improved generalization and reduced overfitting as the dataset grows.*

        - **F1-Score**: Training F1-score stays consistent and high, and test F1-score shows a gradual upward trend.  
          📌 *This shows that the model is increasingly balancing precision and recall — important in imbalanced datasets like diabetes.*

        - **Recall**: Initially lower on the test set (~80%), recall improves significantly with more data, converging toward training recall (~95%).  
          📌 *This is crucial for diabetes detection, where minimizing false negatives (missed diabetic cases) is a priority.*

        ➡️ **Conclusion**: The model generalizes better with more training data, and its ability to correctly identify diabetic patients improves steadily.
        """)

        st.markdown("""
        ### Clinical Context & References
        - *Data Source*: This dataset is from kaggle. It is the output of chinese study conducted in 2016.
        - *Model*: Random Forest Classifier trained on 1,000+ records.
        - *Guidelines*: FPG ≥ 126 mg/dL indicates diabetes according to ADA.

        ### Privacy & Disclaimer
        - This tool is for informational purposes only; not a substitute for medical advice.
        -The model has been trained on limited data therefore make sure to consult a medical professional to confirm your results.
    
        """)
