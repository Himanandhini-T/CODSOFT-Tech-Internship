# app.py

import streamlit as st
import pandas as pd
import pickle


# ------------------------
# Page config
# ------------------------

st.set_page_config(
    page_title="Titanic Predictor",
    page_icon="🚢",
    layout="centered"
)


# ------------------------
# Custom CSS (UI Design)
# ------------------------

st.markdown(
    """
    <style>

    body {
        background-color: #0e1117;
    }

    .main-title {
        font-size:40px;
        font-weight:bold;
        text-align:center;
        color:#00d4ff;
    }

    .sub-text {
        text-align:center;
        color:gray;
        margin-bottom:20px;
    }

    .stButton>button {
        background-color:#00d4ff;
        color:black;
        font-size:18px;
        border-radius:10px;
        height:3em;
        width:100%;
    }

    .result-box {
        padding:20px;
        border-radius:10px;
        text-align:center;
        font-size:24px;
        font-weight:bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------
# Load model
# ------------------------

model = pickle.load(open("titanic_model.pkl", "rb"))


# ------------------------
# Title
# ------------------------

st.markdown(
    "<div class='main-title'>🚢 Titanic Survival Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-text'>Enter passenger details to predict survival</div>",
    unsafe_allow_html=True
)


# ------------------------
# Input form
# ------------------------

col1, col2 = st.columns(2)

with col1:

    pclass = st.selectbox("Passenger Class", [1, 2, 3])

    sex = st.selectbox("Sex", ["Male", "Female"])

    age = st.slider("Age", 1, 80, 25)

with col2:

    sibsp = st.slider("Siblings / Spouse", 0, 5, 0)

    parch = st.slider("Parents / Children", 0, 5, 0)

    fare = st.number_input("Fare", 0.0, 500.0, 50.0)


embarked = st.selectbox("Embarked", ["S", "C", "Q"])


# ------------------------
# Encode input
# ------------------------

sex = 1 if sex == "Male" else 0

embarked_map = {"S": 2, "C": 0, "Q": 1}

embarked = embarked_map[embarked]


input_data = pd.DataFrame(
    [[pclass, sex, age, sibsp, parch, fare, embarked]],
    columns=[
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ],
)


# ------------------------
# Predict button
# ------------------------

if st.button("Predict Survival"):

    result = model.predict(input_data)[0]

    if result == 1:

        st.markdown(
            "<div class='result-box' style='background:#00ff9c;'>✅ Survived</div>",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            "<div class='result-box' style='background:#ff4b4b;'>❌ Did Not Survive</div>",
            unsafe_allow_html=True
        )