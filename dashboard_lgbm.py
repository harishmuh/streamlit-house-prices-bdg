
import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================================
# CUSTOM TRANSFORMER REQUIRED FOR UNPICKLING
# ==========================================================

class HandlingOutliers(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X, y=None):

        data = X.copy()

        columns_to_process = ['LB', 'LT', 'KT']

        for column in columns_to_process:

            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            data[column] = data[column].clip(
                lower=lower_bound,
                upper=upper_bound
            )

        return data


# ==========================================================
# STREAMLIT PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Bandung House Price Predictor",
    layout="centered"
)

st.title("🏡 House Price Prediction - Bandung")


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    model = joblib.load("best_model_lgbm.sav")

    return model


model = load_model()


# ==========================================================
# USER INPUT
# ==========================================================

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


# ==========================================================
# DISTRICT GRADE MAPPING
# ==========================================================

def map_district_grade(district):

    district_1 = ["Arcamanik","Cinambo","Cibiru","Panyileukan","Ujungberung"]

    district_2 = ["Antapani","Mandalajati","Kiaracondong","Gedebage"]

    district_3 = ["Batununggal","Lengkong","Regol","Buahbatu","Rancasari"]

    district_4 = ["Coblong","Cidadap","Sukasari","Sukajadi"]

    district_5 = [
        "Andir","Cicendo","Astana Anyar","Babakan Ciparay",
        "Bojongloa Kaler","Bojongloa Kidul",
        "Bandung Kulon","Bandung Kidul","Bandung Wetan"
    ]

    if district in district_1:
        return 1
    elif district in district_2:
        return 2
    elif district in district_3:
        return 3
    elif district in district_4:
        return 4
    else:
        return 5


# ==========================================================
# BUILD INPUT DATAFRAME
# ==========================================================

def build_input_dataframe():

    grade = map_district_grade(district)

    df = pd.DataFrame({
        "Building Area": [LB],
        "Land Area": [LT],
        "Bedrooms": [KT],
        "Districts": [grade]
    })

    return df


input_df = build_input_dataframe()


# ==========================================================
# PREDICTION
# ==========================================================

if st.sidebar.button("Predict Price"):

    try:

        prediction = model.predict(input_df)[0]

        st.subheader("Predicted House Price")

        st.success(f"Rp {prediction:,.0f}")

    except Exception as e:

        st.error("Prediction failed.")
        st.exception(e)


# ==========================================================
# DEBUG PANEL (Optional but useful)
# ==========================================================

with st.expander("Debug Input Data"):
    st.write(input_df)
```
