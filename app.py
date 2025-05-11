import streamlit as st
import numpy as np
import joblib

# Simple CSS Styling
st.markdown("""
    <style>
        /* Styling for all input boxes and labels */
        .stInput, .stSelectbox, .stNumberInput, .stTextInput, .stCheckbox, .stRadio {
            font-size: 30px;  /* Set font size for all input boxes */
            font-weight: bold;  /* Make input text bold */
            padding: 10px;  /* Add padding to input boxes */
        }

        /* Styling for all labels */
        label {
            font-size: 30px;  /* Set font size for all labels */
            font-weight: bold;  /* Make label text bold */
            color: #023e8a;  /* Label color */
        }

        /* Styling for the prediction result box */
        .stButton > button {
            background-color: #06d6a0;
            color: white;
            font-weight: bold;
            font-size: 18px;
            padding: 0.6rem 1.2rem;
            border-radius: 10px;
            transition: background-color 0.3s ease;
        }
        

        .stButton > button:hover {
           # background-color: ;
        }

            
        /* Success and error boxes */
        .success-box {
            background-color: #d8f3dc;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #06d6a0;
            font-weight: bold;
            color: #065f46;
        }

        .error-box {
            background-color: #fde2e4;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #ef476f;
            font-weight: bold;
            color: #9b2226;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
scaler = joblib.load("model/scaler.pkl")
models = {
    "Logistic Regression": joblib.load("model/logistic_model.pkl"),
    "Decision Tree": joblib.load("model/decision_tree_model.pkl"),
    "Random Forest": joblib.load("model/random_forest_model.pkl")
}

st.title("💓 Heart Disease Prediction App")

# Sidebar info
st.sidebar.title("📘 About")
st.sidebar.info("""This app predicts the likelihood of heart disease based on health data. Choose a model from the dropdown and enter the information to get started!""")

st.sidebar.markdown("**Models Available:**")
for m in models.keys():
    st.sidebar.markdown(f"✅ {m}")

# Select Model
model_name = st.selectbox("🧠 Choose a model", list(models.keys()))
model = models[model_name]

# Input Fields (with emojis)
age = st.number_input("Age", 18, 100, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120? (fbs)", [0, 1])
restecg = st.selectbox("Rest ECG (restecg)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST segment", [0, 1, 2])
ca = st.selectbox("Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Predict Button
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])
input_scaled = scaler.transform(input_data)

if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.markdown('<div class="error-box">⚠️ Likely to have heart disease.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">✅ Unlikely to have heart disease.</div>', unsafe_allow_html=True)
