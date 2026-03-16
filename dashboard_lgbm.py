
import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------
# Page configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="Bandung House Price Predictor",
    layout="centered"
)

st.title("🏡 Bandung House Price Prediction")


# ----------------------------------------------------------
# Load trained model (extract LightGBM only)
# ----------------------------------------------------------

@st.cache_resource
def load_model():

    pipeline = joblib.load("best_model_lgbm.sav")

    # last step of pipeline = trained LGBM model
    model = pipeline.steps[-1][1]

    return model


model = load_model()


# ----------------------------------------------------------
# Sidebar input
# ----------------------------------------------------------

st.sidebar.header("Property Features")

LB = st.sidebar.number_input(
    "Building Area (LB)",
    min_value=20,
    max_value=1000,
    value=120
)

LT = st.sidebar.number_input(
    "Land Area (LT)",
    min_value=20,
    max_value=1000,
    value=150
)

KT = st.sidebar.slider(
    "Bedrooms (KT)",
    1,
    10,
    3
)

district = st.sidebar.selectbox(
    "District",
    [
        "Antapani","Arcamanik","Andir","Astana Anyar",
        "Babakan Ciparay","Bandung Kidul","Bandung Kulon",
        "Bandung Wetan","Batununggal","Bojongloa Kaler",
        "Bojongloa Kidul","Buahbatu","Cibeunying Kidul",
        "Cibeunying Kaler","Cibiru","Cicendo","Cidadap",
        "Cinambo","Coblong","Gedebage","Kiaracondong",
        "Lengkong","Mandalajati","Panyileukan",
        "Rancasari","Regol","Sukajadi","Sukasari",
        "Ujungberung"
    ]
)


# ----------------------------------------------------------
# District grouping (same logic used during training)
# ----------------------------------------------------------

def map_district(d):

    district_1 = ["Arcamanik","Cinambo","Cibiru","Panyileukan","Ujungberung"]
    district_2 = ["Antapani","Mandalajati","Kiaracondong","Gedebage"]
    district_3 = ["Batununggal","Lengkong","Regol","Buahbatu","Rancasari"]
    district_4 = ["Coblong","Cidadap","Sukasari","Sukajadi"]

    if d in district_1:
        return "District1"
    elif d in district_2:
        return "District2"
    elif d in district_3:
        return "District3"
    elif d in district_4:
        return "District4"
    else:
        return "District5"


grade = map_district(district)


# ----------------------------------------------------------
# Create model input (manual encoding)
# ----------------------------------------------------------

def build_features():

    df = pd.DataFrame({
        "LB": [LB],
        "LT": [LT],
        "KT": [KT],
        "District1": [0],
        "District2": [0],
        "District3": [0],
        "District4": [0]
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


data_property = build_features()


# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

if st.sidebar.button("Predict Price"):

    try:

        prediction = model.predict(data_property)[0]

        st.subheader("Predicted House Price")
        st.success(f"Rp {prediction:,.0f}")

    except Exception as e:

        st.error("Prediction failed")
        st.exception(e)


# ----------------------------------------------------------
# Debug panel
# ----------------------------------------------------------

with st.expander("Debug Input Data"):
    st.write(data_property)

