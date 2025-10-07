import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================================================
# 🎯 PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="🏡 House Price Predictor - Uva Wellassa University",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 🧱 SIDEBAR (DEVELOPER INFO)
# =========================================================
st.sidebar.title("👨‍💻 Developer Info")
st.sidebar.markdown("""
**Name:** Abiraj Jatheeskumar  
**University:** Uva Wellassa University of Sri Lanka  
**Department:** Faculty of Applied Sciences – Department of Computing  
**Project:** 🏡 *Smart House Price Prediction System*  
""")
st.sidebar.markdown("---")
st.sidebar.caption("📧 Contact: abirajjatheeskumar@gmail.com")

# =========================================================
# 📦 LOAD TRAINED MODEL
# =========================================================
model = joblib.load("house_price_model.pkl")

# =========================================================
# 🎨 PAGE TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["🏠 Prediction", "📊 Model Info", "ℹ️ About"])

# =========================================================
# 🏠 TAB 1: PREDICTION
# =========================================================
with tab1:
    st.header("🏠 Predict House Price")

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

    if st.button("🔮 Predict Price"):
        encoded = encode_inputs(mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus)
        features = np.array([[area, bedrooms, bathrooms, stories, parking] + encoded])
        prediction = model.predict(features)[0]
        st.success(f"🏠 Estimated House Price: ₹{prediction:,.2f}")

    # --- Bulk prediction from CSV ---
    st.subheader("📂 Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file with the same feature columns", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        preds = model.predict(data)
        data["Predicted Price"] = preds
        st.dataframe(data)
        st.download_button("📥 Download Predictions", data.to_csv(index=False), file_name="predictions.csv")

# =========================================================
# 📊 TAB 2: MODEL INFORMATION
# =========================================================
with tab2:
    st.header("📊 Model Performance Overview")
    st.write("This section shows model accuracy and feature importance (if available).")

    st.markdown("### R² Score (Approximate Training Accuracy): **64%**")

    st.markdown("### 🔍 Feature Importance (Example Visualization)")
    features = ["area", "bedrooms", "bathrooms", "stories", "parking",
                "mainroad", "guestroom", "basement", "hotwaterheating",
                "airconditioning", "prefarea", "furnishingstatus"]

    importance = np.random.rand(len(features))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(features, importance, color="#FF914D")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Demo)")
    st.pyplot(fig)

# =========================================================
# ℹ️ TAB 3: ABOUT
# =========================================================
with tab3:
    st.header("ℹ️ About the Project")
    st.markdown("""
### 🏡 Smart House Price Prediction System
This project predicts house prices using machine learning trained on the *Housing.csv* dataset.

#### 🎯 Objectives
- To estimate property prices based on features like area, rooms, parking, and furnishing.  
- To build a user-friendly web app for easy use by students, real-estate agents, and researchers.

#### 🧠 Tech Stack
- Python, Pandas, NumPy  
- Scikit-learn (Random Forest / Linear Regression)  
- Streamlit for UI  
- Matplotlib for charts  

#### 👨‍💻 Developer
**Abiraj Jatheeskumar**  
Uva Wellassa University of Sri Lanka – Department of Computing  
📧 abirajjatheeskumar@gmail.com
""")

# =========================================================
# 🪄 FOOTER
# =========================================================
st.markdown("---")
st.caption("Developed by **Abiraj Jatheeskumar** | Uva Wellassa University of Sri Lanka © 2025")
