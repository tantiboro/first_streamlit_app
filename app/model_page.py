import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from PIL import Image
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer



def load_model():
    with open('random_forest.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model
dv, model = load_model()
customer = {
    
    'sector_name': "Sector Name",
    'currency_policy': 'Currency Policy',
    'loan_amount': 'Loan Amount',
}
def predict_single(customer, dv, model):
    # cat = customer[categorical + numerical].to_dict(orient='rows')
    # dv = DictVectorizer(sparse=False)
    # dv.fit(cat)
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]




def show_model_page():
    st.markdown("<h1 style='text-align: center; color: red;'>KIVA LOAN APPLICATION</h1>", unsafe_allow_html=True)

    # html_temp = """
    # <div style="background-color:tomato;padding:10px">
    
    # """

    # st.markdown(html_temp, unsafe_allow_html=True) 
    st.markdown("<h1 style='text-align: center; color: green;'>Funding application Process</h1>", unsafe_allow_html=True)
    #st.title("Funding application Process") 

    #st.markdown(html_temp, unsafe_allow_html=True)

    #st.subheader(""" We need some information to predict outcome of your loan""")
    st.markdown("<h2 style='text-align: center; color: blue;'>Please give me some information about your application and I will predict the outcome</h2>", unsafe_allow_html=True)
    sector_name = (
        "Agriculture",
        "Arts",
        "Clothing",
        "Construction",
        "Education",
        "Entertainment",
        "Food",
        "Health",
        "Manufacturing",
        "Personal Use",
        "Retail",
        "Services",
        "Transportation",
        "Wholesale"
    )

    currency_policy = ("shared","standard")

    sector_name = st.sidebar.selectbox("Sector Name", sector_name)
    customer['sector_name'] = sector_name
    currency_policy = st.sidebar.selectbox("Currency Policy", currency_policy)
    customer['currency_policy'] = currency_policy
    loan_amount = st.sidebar.number_input("Loan Needed:" )
    customer['loan_amount'] = loan_amount
    #prediction = predict_single(customer, dv, model)

    ok = st.sidebar.button("Submit your Application ")
    if ok:
        prediction = predict_single(customer,dv,model)
        
        if prediction >= 0.5:
            st.balloons()
            st.markdown("<h2 style='text-align: center; color: black;'>Good luck to the next step of the process</h2>", unsafe_allow_html=True)
            st.markdown("<h1 style='text-align: center; color: black;'>CONGRATULATIONS</h1>", unsafe_allow_html=True)
        else:
            #st.write("Do not be discouraged. Please redefine your application")
            st.markdown("<h2 style='text-align: center; color: red;'>Don't be discourage, Please redefine your application</h2>", unsafe_allow_html=True)
        #st.header( prediction)
        st.markdown("<h1 style='text-align: center; color: green;'>prediction</h1>", unsafe_allow_html=True)
    
    from PIL import Image
    image = Image.open('luke-chesser-unsplash.jpg')
    col1, col2, col3 = st.columns([6,8,6])

    with col1:
        st.markdown("<h2 style='text-align: center; color: blue;'>How Do I achieve my KPYs?</h2>", unsafe_allow_html=True)
        #st.subheader("How Do I my goals?")

    with col2:
        st.image(image, width=500)

    with col3:
        st.markdown("<h2 style='text-align: center; color: green;'>Machine learning or Artificial Intelligence may be your best Toolkit</h2>", unsafe_allow_html=True)
        #st.subheader("Machine learning or Artificial Intelligence may be your best Toolkit")
    #st.image(image, width=400, caption='Road To Unknown')
    


