import streamlit as st
import numpy as np
import joblib

# =========================================================
# 🎯 PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="🏡 House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =========================================================
# 🏫 SIDEBAR INFO
# =========================================================
st.sidebar.title("👨‍💻 Developer Info")
st.sidebar.markdown("""
**Name:** Abiraj Jatheeskumar  
**University:** Uva Wellassa University of Sri Lanka  
**Department:** Faculty of Applied Sciences – Department of Computing  
**Project:** 🏡 *House Price Prediction System*  
""")

st.sidebar.markdown("---")
st.sidebar.caption("📧 Contact: abirajjatheeskumar@gmail.com")

# =========================================================
# 📦 LOAD TRAINED MODEL
# =========================================================
model = joblib.load("house_price_model.pkl")

# =========================================================
# 🏡 MAIN PAGE
# =========================================================
st.title("🏡 House Price Prediction App")
st.write("Welcome to the **House Price Predictor**, built with Python and Streamlit.  Provide your house details below to estimate its market value.")

# ---------------------------
# Input fields
# ---------------------------
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

# ---------------------------
# Encode categorical inputs
# ---------------------------
def encode_inputs(mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus):
    def yes_no(val): return 1 if val == "yes" else 0
    encoded = [
        yes_no(mainroad),
        yes_no(guestroom),
        yes_no(basement),
        yes_no(hotwaterheating),
        yes_no(airconditioning),
        yes_no(prefarea),
    ]
    furnishing_map = {"furnished": 0, "semi-furnished": 1, "unfurnished": 2}
    encoded.append(furnishing_map[furnishingstatus])
    return encoded

# =========================================================
# 🔮 PREDICTION SECTION
# =========================================================
if st.button("🔮 Predict Price"):
    encoded = encode_inputs(mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus)
    features = np.array([[area, bedrooms, bathrooms, stories, parking] + encoded])
    prediction = model.predict(features)[0]
    st.success(f"🏠 Estimated House Price: ₹{prediction:,.2f}")

# =========================================================
# 🪄 FOOTER
# =========================================================
st.markdown("---")
st.caption("Developed by **Abiraj Jatheeskumar** | Uva Wellassa University of Sri Lanka © 2025")
