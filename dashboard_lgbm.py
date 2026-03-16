import streamlit as st
import pandas as pd
import pickle


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

def map_district(d):

    if d in [
        "kiaracondong","arcamanik","cinambo","astana anyar",
        "mandalajati","panyileukan","ujungberung","cibiru"
    ]:
        return "District1"

    elif d in [
        "babakan ciparay","buahbatu","rancasari",
        "antapani","bojongloa kaler","bandung kulon"
    ]:
        return "District2"

    elif d in [
        "cibeunying kaler","batununggal",
        "andir","cicendo","cibeunying kidul"
    ]:
        return "District3"

    elif d in [
        "bojongloa kidul","lengkong","regol",
        "gedebage","sukajadi","bandung kidul","sukasari"
    ]:
        return "District4"

    elif d in ["coblong","cidadap","bandung wetan"]:
        return "District5"

    return "District3"


grade_district = map_district(district)


# ---------------- CREATE MODEL INPUT ----------------

def create_features(LB, LT, KT, grade):

    df = pd.DataFrame({
        "LB":[LB],
        "LT":[LT],
        "KT":[KT],
        "District1":[0],
        "District2":[0],
        "District3":[0],
        "District4":[0]
    })

    if grade == "District1":
        df["District1"] = 1
    elif grade == "District2":
        df["District2"] = 1
    elif grade == "District3":
        df["District3"] = 1
    elif grade == "District4":
        df["District4"] = 1

    return df


data_property = create_features(LB, LT, KT, grade_district)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    with open("best_model_lgbm.sav", "rb") as f:
        pipeline = pickle.load(f)

    model = pipeline.steps[-1][1]   # extract LightGBM estimator

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
