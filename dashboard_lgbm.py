import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin


# ==========================================================
# 1. CUSTOM TRANSFORMER 
# ==========================================================
# REQUIRED FOR PICKLE

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
# 2. PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Bandung House Price Prediction",
    layout="centered"
)

st.title("🏡 Bandung House Price Predictor")


# ==========================================================
# 3. LOAD MODEL (CACHED)
# ==========================================================

@st.cache_resource
def load_model():

    model = joblib.load("best_model_lgbm.sav")

    return model


model = load_model()


# ==========================================================
# 4. USER INPUT
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
# 5. CREATE INPUT DATAFRAME
# ==========================================================

def build_input_dataframe():

    data = pd.DataFrame({
        "LB": [LB],
        "LT": [LT],
        "KT": [KT],
        "District": [district]
    })

    return data


input_df = build_input_dataframe()


# ==========================================================
# 6. PREDICTION
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
# 7. DEBUG (Optional)
# ==========================================================

with st.expander("Debug Input Data"):
    st.write(input_df)
