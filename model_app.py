import streamlit as st
import joblib

model = joblib.load("regression.joblib")

size = st.number_input("Size")
nb_rooms = st.number_input("Number of rooms")
garden = st.selectbox("Garden", ["Yes", "No"])

if st.button("Predict"):
    prediction = model.predict([[size, nb_rooms, garden]])
    st.write(f"The price of the house is {prediction[0]}")

