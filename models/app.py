import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# Load Deep Learning Models
diabetes_model = tf.keras.models.load_model('./models/diabetes_model.h5')
heart_model = tf.keras.models.load_model('./models/heart_model.h5')
parkinsons_model = tf.keras.models.load_model('./models/parkinsons_model.h5')

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction"],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

st.title("Multiple Disease Prediction System")

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.subheader("Diabetes Prediction")
    Pregnancies = st.number_input('Number of Pregnancies')
    Glucose = st.number_input('Glucose Level')
    BloodPressure = st.number_input('Blood Pressure')
    SkinThickness = st.number_input('Skin Thickness')
    Insulin = st.number_input('Insulin Level')
    BMI = st.number_input('BMI Value')
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function')
    Age = st.number_input('Age')

    if st.button('Predict Diabetes'):
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        prediction = diabetes_model.predict(input_data)
        result = 'Diabetic' if prediction > 0.5 else 'Not Diabetic'
        st.success(f"The person is {result}")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.subheader("Heart Disease Prediction")
    age = st.number_input("Age")
    sex = st.selectbox("Sex", ['Male', 'Female'])
    cp = st.number_input("Chest Pain Type")
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Cholesterol")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['No', 'Yes'])
    restecg = st.number_input("Resting ECG")
    thalach = st.number_input("Maximum Heart Rate")
    exang = st.selectbox("Exercise Induced Angina", ['No', 'Yes'])
    oldpeak = st.number_input("ST Depression")
    slope = st.number_input("Slope")
    ca = st.number_input("Major Vessels")
    thal = st.number_input("Thalassemia")

    if st.button('Predict Heart Disease'):
        input_data = np.array([[age, int(sex == 'Male'), cp, trestbps, chol, int(fbs == 'Yes'), restecg, thalach, int(exang == 'Yes'), oldpeak, slope, ca, thal]])
        prediction = heart_model.predict(input_data)
        result = 'Heart Disease Detected' if prediction > 0.5 else 'No Heart Disease'
        st.success(f"The person is {result}")

# Parkinson's Prediction Page
if selected == "Parkinson's Prediction":
    st.subheader("Parkinson's Disease Prediction")
    fo = st.number_input('MDVP:Fo(Hz)')
    fhi = st.number_input('MDVP:Fhi(Hz)')
    flo = st.number_input('MDVP:Flo(Hz)')
    Jitter_percent = st.number_input('MDVP:Jitter(%)')
    RAP = st.number_input('MDVP:RAP')
    HNR = st.number_input('HNR')
    spread1 = st.number_input('spread1')
    spread2 = st.number_input('spread2')
    D2 = st.number_input('D2')
    PPE = st.number_input('PPE')

    if st.button("Predict Parkinson's"):
        input_data = np.array([[fo, fhi, flo, Jitter_percent, RAP, HNR, spread1, spread2, D2, PPE]])
        prediction = parkinsons_model.predict(input_data)
        result = "Parkinson's Detected" if prediction > 0.5 else "No Parkinson's"
        st.success(f"The person is {result}")

# Footer
st.markdown("""
<style>
.footer {
    text-align: center;
    margin-top: 50px;
    padding: 10px;
    background-color: #f1f1f1;
}
</style>
<div class="footer" style="text-align: center; margin-top: 50px; padding: 10px; background: inherit; color: inherit;">
    Developed by <b>Team Kitty<3</b>
</div>


""", unsafe_allow_html=True)

