from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
import joblib

#Loading up the Regression model we created
loaded_rf = joblib.load("./finalrf.joblib")

#Caching the model for faster loading
# @st.cache

def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe
    """
    volatile_acidity = st.sidebar.slider('volatile acidity', 0.00, 1.58, 0.25)
    citric_acid  = st.sidebar.slider('citric acid', 0.00, 1.00, 0.26)
    chlorides  = st.sidebar.slider('chlorides', 0.00, 1.00, 0.08)
    free_sulfur_dioxide = st.sidebar.slider('free sulfur dioxide', 1, 100, 14)
    sulphates = st.sidebar.slider('sulphates', 0.00, 2.00, 0.62)
    alcohol = st.sidebar.slider('alcohol', 8.0, 20.0, 10.2)
    
    features = {'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur_dioxide,
            'sulphates': sulphates,
            'alcohol': alcohol
            }
    data = pd.DataFrame(features,index=[0])

    return data

st.title('Red Wine Quality Prediction')

user_input_dta = get_user_input()
user_input_dta_col = ['volatile acidity', 'citric acid', 'chlorides', 'free sulfur dioxide', 'sulphates', 'alcohol']

st.image("./red_wine.jpeg")

st.subheader('Predicted wine quality')

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

temp = 0
for i in [col1, col2, col3, col4, col5, col6, col7]:
    if temp < 6:
        i.metric(user_input_dta_col[temp], user_input_dta[user_input_dta.columns[temp]])
        temp += 1
    else:
        i.metric('pred quality', loaded_rf.predict(user_input_dta)[0])

pred_proba = pd.DataFrame(loaded_rf.predict_proba(user_input_dta), columns=('Quality %d' % i for i in range(3,9)))

st.subheader('Probability of prediction for each quality group')

st.write(pred_proba)

