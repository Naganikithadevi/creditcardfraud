import numpy as np
import pandas as pd
import streamlit as st
import scikit_learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from PIL import Image

# create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")





submit = st.button("Submit")
@st.cache(allow_output_mutation=True)
def loading_model():
    fp="./model.h5"
    model_loader=load_model(fp)
    return model_loader

image = Image.open('your_image_url.jpg')
st.image(image)


if submit:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = model.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.write(" Legitimate transaction")
    else:
        st.write(" fraud transaction")
