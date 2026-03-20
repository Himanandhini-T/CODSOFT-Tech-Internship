import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Transaction Fraud Detector",
    layout="centered"
)

# -----------------------
# Title
# -----------------------

st.title("💳 Transaction Fraud Detection")
st.write(
    "Classify transactions as Fraudulent or Genuine using Machine Learning"
)

# -----------------------
# Load model
# -----------------------

model = pickle.load(open("fraud_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# -----------------------
# Upload file
# -----------------------

st.subheader("Upload Transaction Data")

file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)


if file is not None:

    df = pd.read_csv(file)

    st.write("Preview")

    st.dataframe(df.head())


    # remove target column if exists
    if "Class" in df.columns:
        X = df.drop("Class", axis=1)
    else:
        X = df.copy()


    if st.button("Detect Fraud"):

        X_scaled = scaler.transform(X)

        preds = model.predict(X_scaled)


        results = []

        for p in preds:
            if p == 1:
                results.append("Fraudulent")
            else:
                results.append("Genuine")


        df["Detection"] = results


        # -----------------------
        # Show result summary
        # -----------------------

        fraud = results.count("Fraudulent")
        genuine = results.count("Genuine")


        st.subheader("Detection Result")

        if fraud > 0:
            st.error(
                f"⚠ Fraudulent Transactions Detected: {fraud}"
            )
        else:
            st.success(
                "✅ All Transactions are Genuine"
            )


        st.info(
            f"Genuine: {genuine} | Fraudulent: {fraud}"
        )


        # -----------------------
        # Show table
        # -----------------------

        st.subheader("Detailed Output")

        st.dataframe(df)