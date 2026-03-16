import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("House Price Prediction in Kota Bandung")

st.write(
    "This app predicts house prices based on property characteristics using machine learning."
)

# ---------------- SIDEBAR INPUT ----------------

st.sidebar.header("Please input Property Features")

LT = st.sidebar.number_input("Land Area (LT)", min_value=1, max_value=507, value=100)
LB = st.sidebar.number_input("Building Area (LB)", min_value=1, max_value=556, value=150)
KT = st.sidebar.slider("Number of Bedrooms (KT)", min_value=1, max_value=8, value=3)

district = st.sidebar.selectbox(
    "Choose District",
    [
        "antapani","sukasari","cibeunying kidul","buahbatu","coblong",
        "mandalajati","arcamanik","bojongloa kaler","rancasari",
        "gedebage","cibeunying kaler","bandung kidul","kiaracondong",
        "ujungberung","sukajadi","regol","cidadap","lengkong",
        "cibiru","batununggal","cicendo","bandung kulon","andir",
        "panyileukan","astana anyar","bojongloa kidul","bandung wetan",
        "babakan ciparay","cinambo"
    ]
)

# ---------------- DISTRICT GROUPING ----------------

if district in [
    "kiaracondong","arcamanik","cinambo","astana anyar",
    "mandalajati","panyileukan","ujungberung","cibiru"
]:
    grade_district = "District1"

elif district in [
    "babakan ciparay","buahbatu","rancasari",
    "antapani","bojongloa kaler","bandung kulon"
]:
    grade_district = "District2"

elif district in [
    "cibeunying kaler","batununggal",
    "andir","cicendo","cibeunying kidul"
]:
    grade_district = "District3"

elif district in [
    "bojongloa kidul","lengkong","regol",
    "gedebage","sukajadi","bandung kidul","sukasari"
]:
    grade_district = "District4"

elif district in ["coblong","cidadap","bandung wetan"]:
    grade_district = "District5"

else:
    grade_district = "District3"


# ---------------- CREATE DATAFRAME ----------------

data_property = pd.DataFrame({
    "LB": [LB],
    "LT": [LT],
    "KT": [KT],
    "grade_district": [grade_district]
})

# convert category to numeric for LightGBM
data_property["grade_district"] = (
    data_property["grade_district"]
    .astype("category")
    .cat.codes
)

# ---------------- LOAD MODEL ----------------

with open("best_model_lgbm.sav", "rb") as f:
    model_loaded = pickle.load(f)

# bypass pipeline and use final LightGBM model
model = model_loaded.steps[-1][1]

# ---------------- PREDICTION ----------------

if st.sidebar.button("Predict Price"):

    prediction = model.predict(data_property)[0]

    st.subheader("Prediction Result")

    st.markdown(
        f"""
        <div style="background-color:white;padding:20px;border-radius:10px;width:fit-content">
            <h2 style="color:black;">
            Predicted Property Price: Rp {prediction:,.2f}
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
