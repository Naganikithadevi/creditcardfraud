import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import load

from PIL import Image

# create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')

submit = st.button("Submit")
if submit:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    st.write(features)
    l = features.reshape(1,-1)
    st.write(l)
    # make prediction
    model = load('creditcardmodel.joblib')
    
    prediction = model.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.write(" Legitimate transaction")
    else:
        st.write(" fraud transaction")
