# Streamlit app for AI-Enhanced Customer Segmentation (K-Means)
# Run with: streamlit run streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

DATA_PATH = "Mall_Customers_cleaned.csv"

st.title('AI-Enhanced Customer Segmentation (K-Means)')

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.subheader('Dataset (first 10 rows)')
st.dataframe(df.head(10))

features = ['Annual Income (k$)', 'Spending Score (1-100)']

st.sidebar.subheader('Clustering settings')
k = st.sidebar.slider('Number of clusters (k)', 2, 8, 5)
run_button = st.sidebar.button('Run Clustering')

if run_button:
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    st.subheader('Clustered Data (first 20 rows)')
    st.dataframe(df.head(20))

    st.subheader('Cluster Scatter Plot')
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'])
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    st.pyplot(fig)

    st.subheader('Cluster Sizes')
    st.bar_chart(df['Cluster'].value_counts().sort_index())

    st.subheader('AI-Style Cluster Names (placeholder)')
    # Simple heuristic naming based on cluster means
    names = {}
    for c in sorted(df['Cluster'].unique()):
        c = int(c)

        sub = df[df['Cluster'] == c]

        income = float(sub['Annual Income (k$)'].mean())
        spend = float(sub['Spending Score (1-100)'].mean())

        if income > float(df['Annual Income (k$)'].mean()) and spend > float(df['Spending Score (1-100)'].mean()):
            names[c] = 'Premium High-Spenders'
        elif income <= float(df['Annual Income (k$)'].mean()) and spend > float(df['Spending Score (1-100)'].mean()):
            names[c] = 'Low Income - High Spend'
        elif income > float(df['Annual Income (k$)'].mean()) and spend <= float(df['Spending Score (1-100)'].mean()):
            names[c] = 'High Income - Low Spend'
        else:
            names[c] = 'Budget / Moderate Segment'
    st.write(names)

    st.info('To integrate a real AI (LLM) for dynamic insights, wire up an API call to OpenAI/Gemini and send the cluster summaries for natural-language explanations.')

st.markdown('---')
st.write('Files generated along with this app: cleaned CSV, PDF report, cluster plots.')
