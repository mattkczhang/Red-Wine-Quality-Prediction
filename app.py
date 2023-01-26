from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt

#Loading up the Regression model, the original dataset
loaded_rf = joblib.load("./finalrf.joblib")
loaded_dta = pd.read_csv('winequality-red.csv')

if 'curr_rf' not in st.session_state:
    st.session_state.curr_rf = loaded_rf

#Caching the model for faster loading
# @st.cache

def reTrain(all_dta):
    X = all_dta[user_input_dta_col]
    y = all_dta['quality']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state=42)
    param = {'criterion':['gini','entropy'],
         'n_estimators': range(100,400,100),
         'min_samples_split' : range(1,5,1),
         'min_samples_leaf' : range(1,5,1)}
    clf = RandomForestClassifier(random_state = 42)
    grid = GridSearchCV(clf,param_grid = param, cv = 3, verbose = 2, n_jobs = -1)
    grid.fit(X_train, y_train)
    st.session_state.curr_rf = RandomForestClassifier(criterion = grid.best_params_['criterion'], 
                             min_samples_leaf = grid.best_params_['min_samples_leaf'],
                             min_samples_split = grid.best_params_['min_samples_split'],
                             n_estimators = grid.best_params_['n_estimators'],
                             random_state = 42)
    st.session_state.curr_rf.fit(X_train, y_train)
    st.session_state.acc = st.session_state.curr_rf.score(X_test, y_test)
    st.success('New model is trained based on the uploaded dataset!')

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

def upload_dta():
    if uploaded_file is not None:
        new_dta = pd.read_csv(uploaded_file)
        if all(item in new_dta.columns for item in user_input_dta_col):
            st.success('You have successfully uploaded a dataset!')
            new_dta = new_dta[user_input_dta_col+['quality']]
            st.caption('View the data')
            st.write(new_dta.head(3))
            all_dta = loaded_dta[user_input_dta_col+['quality']]
            all_dta = all_dta.append(new_dta, ignore_index=True)
            reTrain(all_dta)
        else:
            st.error('Some variables are missed in the uploaded dataset. Please check the file ^_^ ')
    else:
        st.session_state.curr_rf = loaded_rf
        st.session_state.acc = st.session_state.curr_rf.score(X_test, y_test)

user_input_dta = get_user_input()
user_input_dta_col = ['volatile acidity', 'citric acid', 'chlorides', 'free sulfur dioxide', 'sulphates', 'alcohol']

st.session_state.pred = st.session_state.curr_rf.predict(user_input_dta)[0]

X_train, X_test, y_train, y_test = train_test_split(loaded_dta[user_input_dta_col],loaded_dta['quality'],test_size = 0.30, random_state=42)

if 'acc' not in st.session_state:
    st.session_state.acc = st.session_state.curr_rf.score(X_test, y_test)

st.session_state.pred_proba = pd.DataFrame(st.session_state.curr_rf.predict_proba(user_input_dta), columns=('Quality %d' % i for i in range(3,9)))

importance = st.session_state.curr_rf.feature_importances_
feature_importance = pd.DataFrame({"features": st.session_state.curr_rf.feature_names_in_,
                                   "importance": importance})
feature_importance.sort_values('importance', ascending=False, inplace=True)

feature_importance_graph = plt.figure()
sns.barplot(data=feature_importance, x="importance", y="features")
plt.xticks(np.arange(0, 0.26, 0.02))



###########################
### Format the web page ###
###########################

st.title('Red Wine Quality Prediction')

st.image("./red_wine.jpeg")

# st.write(st.session_state)

# Section 1
st.subheader('Add in new data')

uploaded_file = st.file_uploader('Select the new dataset for model training')

retrain = st.button('Process')

if retrain:
    upload_dta()

# Section 2
st.subheader('Predicted wine quality')

st.write('Play with the sidebar on the left to predict the red wine quality')

m1, m2 = st.columns(2)
m1.metric('Predicted quality', st.session_state.pred)
m2.metric('Model accuracy', st.session_state.acc)

col1, col2, col3, col4, col5, col6 = st.columns(6)
temp = 0
for i in [col1, col2, col3, col4, col5, col6]:
    i.metric(user_input_dta_col[temp], user_input_dta[user_input_dta.columns[temp]])
    temp += 1

# Section 3
st.subheader('Probability of prediction for each quality group')

st.write(st.session_state.pred_proba)

# Section 4
st.subheader('Importance of each feature')

st.pyplot(feature_importance_graph)

