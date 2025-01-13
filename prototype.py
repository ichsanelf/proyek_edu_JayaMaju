import streamlit as st
import gdown
import joblib
import numpy as np

#drive url for model & scaler
model_url = "https://drive.google.com/uc?id=1mdrcso2Rh814y1uoS-FYCEqt7t7Qv0NK"
scaler_url = "https://drive.google.com/uc?id=1EyvMUwf0Ld9xQKQNWZWFKZoYCPPEgH79"

#download model from drive
model_output = "RandomForest_model.joblib"
gdown.download(model_url, model_output, quiet=False)

scaler_output = "scaler.joblib"
gdown.download(scaler_url, scaler_output, quiet=False)

#load model and scaler
model = joblib.load(model_output)
scaler = joblib.load(scaler_output)


st.markdown( """ <style> .title { text-align: center; } </style> """, unsafe_allow_html=True )
st.markdown('<h1 class="title">Jaya Jaya Maju</h1>', unsafe_allow_html=True)
st.markdown('<h1 class="title">Prediksi Status Siswa</h1>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    debtor = st.selectbox("Debtor?", ["Yes", "No"])
with col2:
    tuition = st.selectbox("Tuition Fees Up to Date?", ["Yes", "No"])

col1, col2 = st.columns(2)
with col1:
    scholar = st.selectbox("Scholarship Holder?", ["Yes", "No"])
with col2:
    age = st.number_input("Age", min_value=17, max_value=80, step=1)

col1, col2 = st.columns(2)
with col1:
    curricular_1st_sem_approved = st.number_input("Curricular First Semester Approved", min_value=0, max_value=30, step=1)
with col2:
    curricular_1st_sem_grade = st.number_input("Curricular First Semester Grade", min_value=0, max_value=30, step=1)

col1, col2 = st.columns(2)
with col1:
    curricular_2nd_sem_approved = st.number_input("Curricular Second Semester Approved", min_value=0, max_value=30, step=1)
with col2:
    curricular_2nd_sem_grade = st.number_input("Curricular Secod Semester Grade", min_value=0, max_value=30, step=1)

def map_YesNo(value):
    return 1 if value == "Yes" else 0

debtor_input = map_YesNo(debtor)
tuition_input = map_YesNo(tuition)
scholar_input = map_YesNo(scholar)

if st.button("Predict"):    
    input_data = np.array([[debtor_input, tuition_input, scholar_input, age,
                            curricular_1st_sem_approved, curricular_1st_sem_grade,
                            curricular_2nd_sem_approved, curricular_2nd_sem_grade]])    
    
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)   
    st.write("Prediction :", prediction)




