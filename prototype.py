import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('model/RandomForest_model.joblib')

# Encoding mappings
yes_no_encoding = {"Yes": 1, "No": 0}
#prediction_mapping = {2: "Graduated", 1: "Enrolled", 0: "Dropout"}

# Streamlit app
st.title("Simple Prediction App")

# Define input fields for the features
#admission_grade = st.number_input("Admission_grade", min_value=0, max_value=200, step=1, key="enter number")
#tuition = st.selectbox("Tuition Fees is up to date?", ["Yes", "No"])
curricular_1st_sem_approved = st.number_input("Curricular_units_1st_sem_approved", min_value=0, max_value=30, step=1)
curricular_1st_sem_grade = st.number_input("Curricular_units_1st_sem_grade", min_value=0, max_value=30, step=1)
curricular_2nd_sem_approved = st.number_input("Curricular_units_2nd_sem_approved", min_value=0, max_value=30, step=1)
curricular_2nd_sem_grade = st.number_input("Curricular_units_2nd_sem_grade", min_value=0, max_value=30, step=1)

#tuition_encoded = yes_no_encoding[tuition]
# Create a prediction button
if st.button("Predict"):
    # Create a numpy array of the inputs
    input_data = np.array([[
                            curricular_1st_sem_approved, curricular_1st_sem_grade,
                            curricular_2nd_sem_approved, curricular_2nd_sem_grade]])
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_data)  
    
    # Display the prediction
    st.write("Prediction:", prediction)


