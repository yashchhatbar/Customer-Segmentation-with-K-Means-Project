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
# GEMINI CONFIG
# -----------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


def generate_gemini_insights(cluster_summaries):
    """
    Sends cluster summaries to Gemini AI and returns
    business-friendly insights.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
You are a senior business data analyst.

Given the following customer segmentation results,
provide clear and simple insights for each cluster.

For every cluster, explain:
1. Customer behavior
2. Business or marketing strategy

Cluster data:
{cluster_summaries}

Use bullet points.
Keep the explanation concise and professional.
"""

    response = model.generate_content(prompt)
    return response.text


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
    st.subheader("Customer Clusters")

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"]
    )
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    st.pyplot(fig)

    st.subheader("Cluster Sizes")
    st.bar_chart(df["Cluster"].value_counts().sort_index())


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
    # GEMINI AI INSIGHTS
    # -----------------------------
    st.subheader("🤖 Gemini AI – Business Insights")

    with st.spinner("Gemini AI is generating insights..."):
        insights = generate_gemini_insights(cluster_summaries)

    st.markdown(insights)


# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("This project uses K-Means clustering and Google Gemini AI to convert data insights into natural language explanations.")
