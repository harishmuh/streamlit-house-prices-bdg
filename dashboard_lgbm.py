import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle 
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Literal





st.title('House Price Prediction in Kota Bandung')



st.text('This app predicts house prices based on property characteristics using machine learning.')

# menampilkan data distrik kota bandung
data = {
    'District1' : ['kiaracondong','arcamanik','cinambo','astanya anyar','mandalajati','panyileukan','ujung berung','cibiru'],
    'District2' : ['babakan ciparay','buahbatu','rancasari','antapani','bojongloa kaler','bandung kulon','',''],
    'District3' : ['cibeunying kaler', 'batununggal', 'andir', 'cicendo', 'cibeunying kidul','','',''],
    'District4' : ['bojongloa kidul','lengkong','regol','gedebage','sukajadi','bandung kidul','sukasari',''],
    'Distrcit5' : ['coblong', 'cidadap', 'bandung wetan','','','','','']
}

df = pd.DataFrame(data)



st.sidebar.header('Please input Property Features')


# Input fitur dari pengguna
def create_user_input():
    
    # Input numerik
    LT = st.sidebar.number_input('Land Area (LT)', min_value=1, max_value=507, value=100)
    LB = st.sidebar.number_input('Building Area (LB)', min_value=1, max_value=556, value=150)
    KT = st.sidebar.slider('Number of Bedrooms (KT)', min_value=1, max_value=8, value=3)

    # Input kategorikal
    grade_district = st.sidebar.selectbox(
    'Choose District', 
    ['antapani', 'sukasari', 'cibeunying kidul', 'buahbatu', 'coblong',
     'mandalajati', 'arcamanik', 'bojongloa kaler', 'rancasari',
     'gedebage', 'cibeunying kaler', 'bandung kidul', 'kiaracondong',
     'ujungberung', 'sukajadi', 'regol', 'cidadap', 'lengkong',
     'cibiru', 'batununggal', 'cicendo', 'bandung kulon', 'andir',
     'panyileukan', 'astana anyar', 'bojongloa kidul', 'bandung wetan',
     'babakan ciparay', 'cinambo'])
    
    if grade_district in ['kiaracondong','arcamanik','cinambo','astana anyar','mandalajati','panyileukan','ujungberung','cibiru'] : 
        grade_district = 'District1'
    elif grade_district in ['babakan ciparay','buahbatu','rancasari','antapani','bojongloa kaler','bandung kulon'] : 
        grade_district = 'District2'
    elif grade_district in ['cibeunying kaler', 'batununggal', 'andir', 'cicendo', 'cibeunying kidul'] : 
        grade_district = 'District3'
    elif grade_district in ['bojongloa kidul','lengkong','regol','gedebage','sukajadi','bandung kidul','sukasari'] : 
        grade_district = 'District4'
    elif grade_district in ['coblong', 'cidadap', 'bandung wetan'] : 
        grade_district = 'District5'
    else :
        st.write('District Not Found')
    
    # Membuat DataFrame dari input
    data = pd.DataFrame({
        'LB': LB,
        'LT': LT,
        'KT': KT,
        'grade_district': grade_district
    }, index=['value'])

    return data

# Mendapatkan data input dari pengguna
data_property = create_user_input()

# Tampilkan data yang diinput
#st.subheader('Property Features')
#st.write(data_property)


# ---------- Membuat Prediksi dengan Model Regresi ------------------

# Definisi class HandlingOutliers
class HandlingOutliers(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self 
    
    def transform(self, X, y=None):
        data = X.copy()
        
        # Kolom yang akan diproses
        columns_to_process = ['LB', 'LT', 'KT']
        
        for column in columns_to_process:
            # Menggunakan IQR untuk setiap kolom
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Menentukan batas bawah dan batas atas
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Mengganti outliers dengan batas yang sesuai
            data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
        
        return data
    
    def set_output(self, transform: Literal['default', 'pandas']):
        return super().set_output(transform=transform)


# Load Model
with open('best_model_lgbm.sav', 'rb') as f:
    model_loaded = pickle.load(f)

# Prediksi menggunakan model regresi yang telah dimuat
predicted_price = model_loaded.predict(data_property)[0]

# Tampilkan hasil prediksi
st.subheader('Prediction Result:')
#st.write(f'Predicted Property Price: Rp {predicted_price:,.2f}')

# st.markdown(
#     f"""
#     <h2 style="font-size: 24px; font-weight: bold; ">Predicted Property Price: Rp {predicted_price:,.2f}</h2>
#     """,
#     unsafe_allow_html=True
# )

st.markdown(
    f"""
    <div style="background-color: white; padding: 15px; border-radius: 10px; width: fit-content;">
        <h2 style="font-size: 24px; font-weight: bold; color: black;">Predicted Property Price: Rp {predicted_price:,.2f}</h2>
    </div>
    """,
    unsafe_allow_html=True
)