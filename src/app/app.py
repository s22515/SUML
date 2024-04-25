import sys
sys.path.append('../')
from ml.model_creator_module import get_model_prediction
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import math


np.set_printoptions(precision=2)
st.title("Crabs prediction form")

sex_dixt = {
    "Male": 2,
    "Female" : 0,
    "Indeterminate" : 1
}

option = st.selectbox(
   "Crab sex",
   ("Female", "Male", "Indeterminate"),
   index=0,
   placeholder="Indeterminate",
)
length = st.number_input("Crab lenght", 0.1,2.5)

diameter = st.number_input("Crab diameter", 0.1,2.0)

height = st.number_input("Crab height", 0.1,3.0)

weight = st.number_input("Crab weight", 0.01,80.5)

shucked_weight = st.number_input("Crab shucked weight", 0.01,45.0)

viscara_weight = st.number_input("Crab viscara weight", 0.01,22.5)

shell_weight = st.number_input("Crab shell weight", 0.01,30.0)


if st.button('did'):
    volume = length * height * diameter
    weight_proportion = (shucked_weight + viscara_weight + shell_weight) / weight
    shucked_proportion = shucked_weight / weight
    viscera_proportion = viscara_weight / weight
    shell_proportion = shell_weight / weight
    shell_area = (diameter /2)**2 * math.pi

    scaler = joblib.load('../scaler.bin')
    scaled_len = scaler.transform([[length, diameter, height, weight, shucked_weight,
                                    viscara_weight, shell_weight, volume,weight_proportion,
                                    shell_proportion, viscera_proportion,shell_proportion,
                                    shell_area]])
    
    input_data = np.insert(scaled_len, 0, sex_dixt[option])

    prediction = get_model_prediction([input_data])
    st.write('Predicted age of crab:', round(prediction[0], 2))

