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
# 🎨 CUSTOM CSS
# =========================================================
st.markdown("""
    <style>
        body {
            background-color: #F9FAFB;
        }
        .stButton>button {
            color: white;
            background-color: #FF914D;
            border-radius: 10px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

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
st.sidebar.caption("📧 Contact: abiraj30@gmail.com")
# =========================================================
# 🌟 APP HEADER
# =========================================================
st.markdown("<h1 style='text-align:center; color:#FF914D;'>🏡 Smart House Price Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:gray;'>Uva Wellassa University of Sri Lanka</h4>", unsafe_allow_html=True)


# =========================================================
# 📦 LOAD TRAINED MODEL
# =========================================================
model, feature_scaler, target_scaler = joblib.load("house_price_model.pkl")

# Display model info in sidebar
st.sidebar.info(f"✅ Model Loaded: {type(model).__name__}")

# =========================================================
# 🧹 PREPROCESSING FUNCTION
# =========================================================
EXPECTED_COLS = [
    "area","bedrooms","bathrooms","stories",
    "mainroad","guestroom","basement","hotwaterheating",
    "airconditioning","parking","prefarea","furnishingstatus"
]

def preprocess_bulk(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        st.error(f"CSV is missing columns: {missing}")
        st.stop()

    for c in ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea","furnishingstatus"]:
        df[c] = df[c].astype(str).str.strip().str.lower()

    yn_cols = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
    for c in yn_cols:
        df[c] = df[c].map({"yes": 1, "no": 0})

    furnish_map = {"furnished": 0, "semi-furnished": 1, "unfurnished": 2}
    df["furnishingstatus"] = df["furnishingstatus"].map(furnish_map)

    return df[EXPECTED_COLS]

# =========================================================
# 🎨 PAGE TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["🏠 Prediction", "📊 Model Info", "ℹ️ About"])


# =========================================================
# 🏠 TAB 1: PREDICTION
# =========================================================
with tab1:
    st.header("🏠 Predict House Price")

    # =========================================================
    # 💱 CURRENCY SELECTION
    # =========================================================
    currency_option = st.selectbox(
        "💱 Select Currency",
        ("INR (₹)", "LKR (Rs.)", "USD ($)")
    )
    st.caption("💡 This currency will be used for both single and bulk predictions.")



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
        furnish_map = {"furnished": 0, "semi-furnished": 1, "unfurnished": 2}
        encoded.append(furnish_map[furnishingstatus])
        return encoded

    if st.button("🔮 Predict Price"):
        with st.spinner("Calculating... Please wait ⏳"):
            encoded = encode_inputs(mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus)
            features = np.array([[area, bedrooms, bathrooms, stories, parking] + encoded])
            features_scaled = feature_scaler.transform(features)
            prediction_scaled = model.predict(features_scaled)[0]
            prediction_inr = target_scaler.inverse_transform([[prediction_scaled]])[0][0]

            # Currency conversion
            if currency_option == "LKR (Rs.)":
                prediction = prediction_inr * 3.7
                symbol = "Rs."
            elif currency_option == "USD ($)":
                prediction = prediction_inr / 83
                symbol = "$"
            else:
                prediction = prediction_inr
                symbol = "₹"

            st.success(f"🏠 Estimated House Price: {symbol}{prediction:,.2f}")

    # --- Bulk prediction from CSV ---
    st.subheader("📂 Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file with the same feature columns", type=["csv"])
    st.caption("📘 Example CSV columns: area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus")

    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
        X = preprocess_bulk(raw)
        X_scaled = feature_scaler.transform(X)
        y_scaled = model.predict(X_scaled)
        y_pred = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

        # Currency conversion
        if currency_option == "LKR (Rs.)":
            y_pred *= 3.7
            symbol = "Rs."
        elif currency_option == "USD ($)":
            y_pred /= 83
            symbol = "$"
        else:
            symbol = "₹"

        out = raw.copy()
        out[f"Predicted Price ({symbol})"] = np.round(y_pred, 2)
        st.dataframe(out.style.background_gradient(cmap='Blues'))
        st.success(f"✅ Bulk predictions generated successfully in {symbol} currency.")
        st.download_button("📥 Download Predictions", out.to_csv(index=False), file_name="predictions.csv")

# =========================================================
# 📊 TAB 2: MODEL INFORMATION
# =========================================================
with tab2:
    st.header("📊 Model Performance Overview")
    st.write("This section shows model accuracy and feature importance (if available).")

    st.markdown("### 🔍 Feature Importance (Model-Derived Visualization)")
    if hasattr(model, "feature_importances_"):
        features = EXPECTED_COLS
        importance = model.feature_importances_
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(features, importance, color="#FF914D")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    else:
        st.warning("Feature importance not available for this model.")
    st.info("💡 Model trained using RandomForestRegressor with approximately 64% accuracy on test data.")


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
- Scikit-learn (Random Forest Regressor)  
- Streamlit for UI  
- Matplotlib for charts  

#### 👨‍💻 Developer
**Abiraj Jatheeskumar**  
Uva Wellassa University of Sri Lanka – Department of Computing  
📧 abiraj30@gmail.com
""")

# =========================================================
# 🪄 FOOTER
# =========================================================
st.markdown("---")
st.caption("Developed by **Abiraj Jatheeskumar** | Uva Wellassa University of Sri Lanka © 2025")
