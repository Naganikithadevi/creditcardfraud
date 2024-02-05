import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import load,dump
from PIL import Image

model = RandomForestClassifier(n_estimators=100, random_state=42)

dump(model, "creditcardmodel.joblib")

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

    # Check if the number of features is the same as expected
    if len(input_df_lst) != model.n_features_in_:
        st.error("Invalid number of features. Please provide the correct number of features.")
    else:
        # Reshape and convert to numpy array
        features = np.array(input_df_lst).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        if prediction[0] == 0:
            st.write("Legitimate transaction")
        else:
            st.write("Fraudulent transaction")

