import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('model/RandomForest_model.joblib')
st.markdown( """ <style> .title { text-align: center; } </style> """, unsafe_allow_html=True )

st.markdown('<h1 class="title">Jaya Jaya Maju</h1>', unsafe_allow_html=True)
st.markdown('<h1 class="title">Prediksi Status Siswa</h1>', unsafe_allow_html=True)

curricular_1st_sem_approved = st.number_input("Curricular First Semester Approved", min_value=0, max_value=30, step=1)
curricular_1st_sem_grade = st.number_input("Curricular First Semester Grade", min_value=0, max_value=30, step=1)
curricular_2nd_sem_approved = st.number_input("Curricular Second Semester Approved", min_value=0, max_value=30, step=1)
curricular_2nd_sem_grade = st.number_input("Curricular Secod Semester Grade", min_value=0, max_value=30, step=1)

if st.button("Predict"):    
    input_data = np.array([[curricular_1st_sem_approved, curricular_1st_sem_grade,
                            curricular_2nd_sem_approved, curricular_2nd_sem_grade]])    
   
    prediction = model.predict(input_data)   
    st.write("Prediction :", prediction)




