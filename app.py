import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def load_assets():
    with open("insurance_cost.pkl","rb") as f:
        model = pickle.load(f)
    with open("model_columns.pkl","rb") as f:
        columns = pickle.load(f)
    with open("scaler.pkl","rb") as f:
        scaler = pickle.load(f)

    return model,columns,scaler


model,model_columns,scaler = load_assets() 

num_cols = ['age', 'bmi', 'children']
    
st.title("Insurance Cost Predictor")
st.write("Enter details below to estimate medical insurance charges.")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.selectbox("Number of Children", options=[0, 1, 2, 3, 4, 5])

with col2:
    sex = st.selectbox("Sex", options=['male', 'female'])
    smoker = st.selectbox("Smoker?", options=['yes', 'no'])
    region = st.selectbox("Region", options=['southwest', 'southeast', 'northwest', 'northeast'])
    
if st.button("Predict Cost"):
    input_dict = {
        'age' : age,
        'bmi' : bmi,
        'children' : children,
        'sex' : sex,
        'smoker' : smoker,
        'region' : region
    }
    
    input_df = pd.DataFrame([input_dict]) #<----- put in [] because df meant to be array of dict
    encoded_df = pd.get_dummies(input_df)
    final_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    #FIX: Use .transform() (DO NOT USE .fit_transform here)
    # This uses the mean/std learned from your training data
    final_df[num_cols] = scaler.transform(final_df[num_cols])
    
    log_pred = model.predict(final_df)
    pred = np.expm1(log_pred)  # Reversing the log transformation
    st.success(f"Estimated Insurance Cost: ${pred[0]:.2f}")
    