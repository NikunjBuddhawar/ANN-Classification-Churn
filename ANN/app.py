import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

model = tf.keras.models.load_model('model.h5')

with open('leg.pkl','rb') as file:
    leg = pickle.load(file)

with open('ohe.pkl','rb') as file:
    ohe = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

st.title('Customer Churn Predictor')

geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [leg.transform([gender])[0]], ## leg.transform([gender]) this basically gives you a 1d array so to get the value access the first element of the array using [0].
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

values = ohe.transform([[geography]]).toarray()
values_df = pd.DataFrame(values,columns = ohe.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data,values_df],axis =1)

ids = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(ids)

    answer = prediction[0][0]

    st.subheader("Prediction Result")
    st.write(f"Churn Probability: {answer:.2f}")

    if answer > 0.5:
        st.error('The customer is likely to churn.')
    else:
        st.success('The customer is not likely to churn.')



