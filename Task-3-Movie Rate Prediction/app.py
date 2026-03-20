import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt


# ======================
# Load model
# ======================

with open("movie_rating_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
le_genre = data["le_genre"]
le_dir = data["le_dir"]
le_a1 = data["le_a1"]
le_a2 = data["le_a2"]
le_a3 = data["le_a3"]


# ======================
# Page config
# ======================

st.set_page_config(
    page_title="Movie Rating Predictor",
    layout="centered"
)


# ======================
# CSS UI
# ======================

st.markdown("""
<style>

.stApp {
    background-image: url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba");
    background-size: cover;
}


/* Main Title */

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: gold;
    margin-bottom: 10px;
}


/* Highlight box */

.subbox {
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: white;
    background: rgba(0,0,0,0.7);
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 20px;
}


/* labels */

label {
    color: white !important;
    font-size: 16px !important;
    font-weight: bold;
}


/* button */

.stButton>button {
    width: 100%;
    background-color: gold;
    color: black;
    font-size: 18px;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# ======================
# Title
# ======================

st.markdown(
    '<div class="title">🎬 Movie Rating Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subbox">Enter Movie Details</div>',
    unsafe_allow_html=True
)


# ======================
# Inputs
# ======================

genre = st.selectbox("Genre", le_genre.classes_)

director = st.selectbox("Director", le_dir.classes_)

actor1 = st.selectbox("Actor 1", le_a1.classes_)

actor2 = st.selectbox("Actor 2", le_a2.classes_)

actor3 = st.selectbox("Actor 3", le_a3.classes_)

year = st.slider("Year", 1980, 2025, 2015)

votes = st.slider("Votes", 100, 1000000, 5000)


# ======================
# Prediction
# ======================

if st.button("Predict Rating"):

    g = le_genre.transform([genre])[0]
    d = le_dir.transform([director])[0]
    a1 = le_a1.transform([actor1])[0]
    a2 = le_a2.transform([actor2])[0]
    a3 = le_a3.transform([actor3])[0]

    X = np.array([[g, d, a1, a2, a3, year, votes]])

    pred = model.predict(X)[0]

    st.success(f"⭐ Predicted Rating: {round(pred,2)} / 10")


    # Graph

    fig, ax = plt.subplots()

    ax.bar(["Rating"], [pred])

    ax.set_ylim(0,10)

    ax.set_title("Predicted Rating")

    st.pyplot(fig)