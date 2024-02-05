import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import load,dump
from PIL import Image

model=load("creditcardmodel.joblib")



# Assuming X_train and y_train are defined before this point
# Make sure to replace 'your_image_url.jpg' with the actual path or URL of your image


st.title("Credit Card Fraud Detection Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")
input_df = st.text_input('Input All features')

submit = st.button("Submit")

# Load the image
image = Image.open('your_image_url.jpg')
st.image(image)

if submit:
    # Split the input string into a list of features
    input_df_lst = [float(x.strip()) for x in input_df.split(',')]
    features = np.array(input_df_lst).reshape(1, -1)
    # Make prediction
    prediction = model.predict(features)
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")

