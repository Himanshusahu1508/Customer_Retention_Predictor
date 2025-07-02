import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load the trained scaler and model
sc = joblib.load('scaler.pkl')
model = tf.keras.models.load_model('ann_model.h5')

st.title("Bank Customer Churn Prediction")
st.write("""
Enter the details of the customer below and find out whether they are likely to leave the bank or not.
""")

# Geography input
geography = st.selectbox("Geography", ("France", "Spain", "Germany"))

# Credit Score
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)

# Gender
gender = st.selectbox("Gender", ("Male", "Female"))

# Age
age = st.number_input("Age", min_value=18, max_value=100, value=40)

# Tenure
tenure = st.slider("Tenure (years at bank)", min_value=0, max_value=10, value=3)

# Balance
balance = st.number_input("Account Balance ($)", min_value=0.0, value=60000.0, step=1000.0, format="%.2f")

# Number of Products
num_products = st.selectbox("Number of Products", (1, 2, 3, 4))

# Has credit card
has_cr_card = st.radio("Has Credit Card?", ("Yes", "No"))

# Is active member
is_active_member = st.radio("Is Active Member?", ("Yes", "No"))

# Estimated Salary
estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")

if st.button("Predict"):
    # One-hot encode geography manually (same as your training encoding order: France, Germany, Spain)
    if geography == "France":
        geo_encoded = [1, 0, 0]
    elif geography == "Germany":
        geo_encoded = [0, 1, 0]
    else:  # Spain
        geo_encoded = [0, 0, 1]

    # Label encode gender (Male: 1, Female: 0) as in your code
    gender_encoded = 1 if gender == "Male" else 0

    # Convert has credit card and active member to binary
    has_cr_card_bin = 1 if has_cr_card == "Yes" else 0
    is_active_member_bin = 1 if is_active_member == "Yes" else 0

    # Build the feature vector
    features = geo_encoded + [
        credit_score,
        gender_encoded,
        age,
        tenure,
        balance,
        num_products,
        has_cr_card_bin,
        is_active_member_bin,
        estimated_salary
    ]

    # Reshape to 2D array and scale
    X_input = sc.transform(np.array([features]))

    # Make prediction
    prediction = model.predict(X_input)
    churn = prediction[0][0] > 0.5

    st.write("---")
    if churn:
        st.error("⚠️ The model predicts this customer is likely to leave the bank!")
    else:
        st.success("✅ The model predicts this customer will stay with the bank!")
