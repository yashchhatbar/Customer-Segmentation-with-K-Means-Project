# Streamlit app for AI-Enhanced Customer Segmentation using K-Means + Gemini AI
# Run locally: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import google.generativeai as genai


# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_PATH = "Mall_Customers_cleaned.csv"

st.set_page_config(page_title="AI Customer Segmentation", layout="centered")
st.title("AI-Enhanced Customer Segmentation (K-Means + Gemini)")


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head(10))


# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
st.sidebar.header("Clustering Settings")
k = st.sidebar.slider("Number of Clusters (k)", 2, 8, 5)
run_button = st.sidebar.button("Run Clustering")


# -----------------------------
# RUN CLUSTERING
# -----------------------------
if run_button:

    features = ["Annual Income (k$)", "Spending Score (1-100)"]

    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    st.subheader("Clustered Data")
    st.dataframe(df.head(20))


    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    st.subheader("AI-Style Cluster Names (Placeholder)")

cluster_names = {}

for c in sorted(df["Cluster"].unique()):
    c_int = int(c)  # 🔥 FIX 1: numpy.int32 → int
    sub = df[df["Cluster"] == c_int]

    income = float(sub["Annual Income (k$)"].mean())      # 🔥 FIX 2
    spend = float(sub["Spending Score (1-100)"].mean())   # 🔥 FIX 3

    if income > df["Annual Income (k$)"].mean() and spend > df["Spending Score (1-100)"].mean():
        cluster_names[str(c_int)] = "Premium High-Spenders"
    elif income <= df["Annual Income (k$)"].mean() and spend > df["Spending Score (1-100)"].mean():
        cluster_names[str(c_int)] = "Low Income - High Spend"
    elif income > df["Annual Income (k$)"].mean() and spend <= df["Spending Score (1-100)"].mean():
        cluster_names[str(c_int)] = "High Income - Low Spend"
    else:
        cluster_names[str(c_int)] = "Budget / Moderate Segment"

st.json(cluster_names)

    # -----------------------------
    # CLUSTER SUMMARIES
    # -----------------------------
    cluster_summaries = []

    for c in sorted(df["Cluster"].unique()):
        c = int(c)
        sub = df[df["Cluster"] == c]

        cluster_summaries.append({
            "cluster_id": c,
            "customers": int(len(sub)),
            "avg_income": round(float(sub["Annual Income (k$)"].mean()), 2),
            "avg_spending": round(float(sub["Spending Score (1-100)"].mean()), 2)
        })

    st.subheader("Cluster Statistics")
    st.json(cluster_summaries)


# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("This project uses K-Means clustering and Google Gemini AI to convert data insights into natural language explanations.")
