
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Configure the Streamlit page
st.set_page_config(page_title="House Price Predictor", page_icon="🏡")

# App title and description
st.title("🏡 House Price Prediction")
st.markdown("Enter the property details to predict its estimated price.")
model = joblib.load("models/trained_pipe_knn.sav")
# Input form (collapsible section)
with st.expander("📝 Enter Property Details"):
    lot_area = st.number_input("📏 Lot Area (LotArea)", min_value=500, max_value=20000, value=5000)
    basement = st.number_input("🏗️ Basement Area (TotalBsmtSF)", min_value=0, max_value=3000, value=800)
    bedrooms = st.slider("🛏️ Number of Bedrooms (BedroomAbvGr)", 1, 10, 3)
    garage = st.slider("🚗 Number of Garage Spaces (GarageCars)", 0, 5, 1)

# Create input DataFrame for the model
input_df = pd.DataFrame([{
    'LotArea': lot_area,
    'TotalBsmtSF': basement,
    'BedroomAbvGr': bedrooms,
    'GarageCars': garage
}])


# Prediction button
if st.button("📊 Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"💰 Estimated House Price: {prediction[0]:,.0f} Toman")
    import matplotlib.pyplot as plt

# Show model performance
st.subheader("📈 Model Performance")

# Load test vs predicted values
eval_df = pd.read_csv("models/evaluation.csv")
r2 = r2_score(eval_df["actual"], eval_df["predicted"])

st.write(f"**R² Score on Test Set:** {r2:.2f}")

# Error analysis chart
fig, ax = plt.subplots()
ax.scatter(eval_df["actual"], eval_df["predicted"])
ax.plot([eval_df["actual"].min(), eval_df["actual"].max()],
        [eval_df["actual"].min(), eval_df["actual"].max()],
        'r--')
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("🔍 Error Analysis: Actual vs Predicted")
st.pyplot(fig)
