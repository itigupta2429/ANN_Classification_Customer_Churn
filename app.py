import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import pickle

#Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the scaler and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('one_hot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

## Streamlit app
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox("Geography", onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider("Age", min_value=18, max_value=100)
credit_score = st.number_input("Credit Score")
balance = st.number_input("Balance")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure (Years)", min_value=0, max_value=10)
number_of_products = st.slider("Number of Products", min_value=1, max_value=4)
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare the input data with correct feature order
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]
# Display the prediction probability
st.write(f"Prediction Probability: {prediction_proba:.2f}")

# Display the prediction
if prediction[0][0] > 0.5:
    st.success("The customer is likely to churn.")
else:
    st.success("The customer is likely to stay loyal.")

