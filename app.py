import streamlit as st
import requests

# Define FastAPI endpoint URL
api_url = "https://drug-adverse.streamlit.app/predict/"  # Replace with your FastAPI endpoint URL

# Streamlit app
st.title('Adverse Event Prediction')

# Input fields
age = st.number_input('Age at Onset', min_value=0, max_value=150, value=30)
sex = st.selectbox('Sex', ['Male', 'Female'])

drug_composition = st.text_input('Drug Composition', 'Drug A,Drug B')
drug_indication = st.text_input('Drug Indication', 'Indication X,Indication Y')

if sex == 'Male':
    patientsex = 0
else:
    patientsex = 1

# Predict button
if st.button('Predict'):
    # Prepare data for API request
    data = {
        "patientonsetage": age,
        "patientsex": patientsex,
        "drug_composition": drug_composition,
        "drug_indication": drug_indication
    }

    # Make API request
    try:
        response = requests.get(api_url, params=data)
        prediction = response.json()["prediction"]

        # Display prediction
        st.success(f'Prediction: {prediction}')
    except Exception as e:
        st.error(f'Error: {e}')
