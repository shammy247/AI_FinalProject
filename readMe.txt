# Diabetes Prediction Model

The Google Colab notebook builds and evaluates a machine learning model to predict the likelihood of diabetes using patient medical data.

##Models Implementation
-The data was already cleaned (https://www.kaggle.com/datasets/pkdarabi/diabetes-dataset-with-18-features/data
And this is the dataset on kaggle)
-Data visualization and pre-processing was done
-Decision tree was the first model to be Implemented
-Then random forest
-Then balanced and better tuned random forest
-Lastly the data was balanced using SMOTE and trained using random forest

## Features
- Built with **Random Forest Classifier**
- SHAP values for feature importance
- Interactive Streamlit app integration
- Performance evaluation (Accuracy, Recall, F1-Score and confusion metrics)
- Handles imbalanced data uding balanced random forest and the use of SMOTE and focuses on recall for diabetic class

## 📂 Dataset
- **Source**: We found the dataset on kaggle. It is from a chinese research conducted in 2016
             The dataset had over 3000 entries and 17 features.
- **Features**: Age, BMI, Blood Pressure, FPG, Cholesterol, Family History, etc.
- **Target**: Diabetes status (1 = Diabetic, 0 = Not Diabetic). The output is a probability of being diabetic.

## 🛠️ Technologies Used
- Python, Scikit-learn, Pandas, Seaborn, SHAP, Streamlit
- Google Colab for training and visualization

## 🚀 How to Use
1. Run each cell from top to bottom.
2. Train the model and evaluate results.
3. Download the trained model using `joblib`.
4. Deploy using `Streamlit` by uploading model files.

## 📊 Evaluation Metrics
- Accuracy: 94–95%
- Recall for diabetic class: > 85%
- SHAP interpretation for individual predictions

## 📁 Files
- `random_forest_model.pkl`: Trained model
- `test_report.pkl`: Test performance report
- `val_cm.pkl` and `test_cm.pkl`: Confusion matrices
- `learning_curves.png`: Training performance visual

## ⚠️ Disclaimer
This notebook is for educational purposes only. It should not be used for real medical diagnosis.

