import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocess import prepare_input

# Load model
# Load model
with open("model/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# App title
st.title("Diamond Price Predictor")

# Sidebar inputs
st.sidebar.header("Input Features")
carat = st.sidebar.slider("Carat", 0.2, 5.0, 1.0)
depth = st.sidebar.slider("Depth", 50.0, 70.0, 61.5)
table = st.sidebar.slider("Table", 50.0, 70.0, 57.0)

# Predict button
if st.sidebar.button("Predict"):
    input_df = prepare_input(carat, depth, table)
    prediction = model.predict(input_df)[0]
    st.subheader("Predicted Price (USD)")
    st.success(f"${prediction:,.2f}")

    # Visualize Input
    st.subheader("Input Data Overview")
    st.write(input_df)

    # Sample Visualization
    st.subheader("Carat vs. Price Distribution (Sample Data)")
    sample_data = pd.read_csv("dataset/Diamonds Prices2022.csv")
    sns.scatterplot(x='carat', y='price', data=sample_data.sample(500), alpha=0.5)
    plt.axvline(carat, color='red', linestyle='--')
    st.pyplot(plt)
