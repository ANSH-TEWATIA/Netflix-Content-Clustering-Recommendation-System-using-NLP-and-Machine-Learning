import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("clustered_netflix.csv")

# Load model (optional if needed)
kmeans = pickle.load(open("kmeans.pkl", "rb"))

similarity = pickle.load(open("similarity.pkl", "rb"))

# -------------------------------
# UI Title
# -------------------------------
st.title("🎬 Netflix Recommendation System")

# -------------------------------
# Dropdown
# -------------------------------
selected_show = st.selectbox("Choose a show", df['title'].unique())

# -------------------------------
# Recommendation Logic
# -------------------------------
def recommend(title):
    idx = df[df['title'] == title].index[0]

   
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_5 = [i[0] for i in scores[1:6]]
    return df['title'].iloc[top_5]

# -------------------------------
# Button
# -------------------------------
if st.button("Recommend"):
    recommendations = recommend(selected_show)

    st.subheader("Top Recommendations:")
    for rec in recommendations:
        st.write(rec)