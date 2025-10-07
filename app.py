import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("house_price_model.pkl")

st.set_page_config(page_title="🏡 Housing Price Predictor", page_icon="🏠")
st.title("🏡 Housing Price Prediction App")
st.write("Enter the house details below to predict the price:")

# Input fields
area = st.number_input("Area (sq.ft)", min_value=0.0, step=1.0)
bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=0, step=1)
stories = st.number_input("Number of Stories", min_value=0, step=1)
mainroad = st.selectbox("Main Road", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
parking = st.number_input("Parking Spaces", min_value=0, step=1)
prefarea = st.selectbox("Preferred Area", ["yes", "no"])
furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# Encode categorical variables manually (same order as training)
def encode_inputs(mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus):
    def encode_yes_no(value):
        return 1 if value == "yes" else 0

    encoded = [
        encode_yes_no(mainroad),
        encode_yes_no(guestroom),
        encode_yes_no(basement),
        encode_yes_no(hotwaterheating),
        encode_yes_no(airconditioning),
        encode_yes_no(prefarea),
    ]

    # Encode furnishing status manually (same label order)
    furnishing_map = {"furnished": 0, "semi-furnished": 1, "unfurnished": 2}
    encoded.append(furnishing_map[furnishingstatus])

    return encoded

if st.button("Predict Price"):
    encoded = encode_inputs(mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus)
    
    # Combine numerical + categorical in the same order as training
    features = np.array([[area, bedrooms, bathrooms, stories, parking] + encoded])
    
    prediction = model.predict(features)[0]
    st.success(f"🏠 Estimated House Price: ₹{prediction:,.2f}")
