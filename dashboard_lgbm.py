import streamlit as st
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------- CUSTOM TRANSFORMER ----------------

class HandlingOutliers(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        data = X.copy()
        columns_to_process = ["LB", "LT", "KT"]

        for column in columns_to_process:

            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            data[column] = data[column].clip(lower=lower, upper=upper)

        return data


# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="Bandung House Price Predictor")

st.title("House Price Prediction in Kota Bandung")

st.write(
    "Predict property price using a trained LightGBM regression model."
)


# ---------------- USER INPUT ----------------

st.sidebar.header("Property Features")

LT = st.sidebar.number_input("Land Area (LT)", 1, 507, 100)
LB = st.sidebar.number_input("Building Area (LB)", 1, 556, 150)
KT = st.sidebar.slider("Bedrooms (KT)", 1, 8, 3)


district = st.sidebar.selectbox(
    "District",
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


# ---------------- CREATE INPUT DATA ----------------

data_property = pd.DataFrame({
    "LB": [LB],
    "LT": [LT],
    "KT": [KT],
    "grade_district": [grade_district]
})


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    with open("best_model_lgbm.sav", "rb") as f:
        model = pickle.load(f)

    return model


model = load_model()


# ---------------- PREDICTION ----------------

if st.sidebar.button("Predict Price"):

    prediction = model.predict(data_property)[0]

    st.subheader("Prediction Result")

    st.markdown(
        f"""
        <div style="background-color:white;padding:20px;border-radius:10px">
        <h2 style="color:black">
        Predicted Property Price: Rp {prediction:,.2f}
        </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
