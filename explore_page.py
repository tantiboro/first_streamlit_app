import plotly .express as px
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image

# Use the full page instead of a narrow central column
#st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: red;'>Analysis of Kiva Application Process</h1>", unsafe_allow_html=True)
@st.cache
def load_data():
    df = pd.read_csv("loan_2.csv")
    df.columns = [i.replace(' ', '_').lower() for i in df.columns]
    df = df[df.status != 'fundRaising']
    df['status'] = df.apply(lambda x: 1 if x.status == 'funded' else 0, axis=1)
    df = df[['loan_amount', 'sector_name', 'currency_policy', 'status']]
    df = df.dropna()
    return df
df = load_data()
st.markdown("<h1 style='text-align: center; color: purple;'>Welcome to our application that will help evaluate how much amount of loan you can get from the KIVA loan process</h1>", unsafe_allow_html=True)
df.head()
# st.header(
#     """
#      KIVA LOAN APPLICATION DATABASE
#     """
# )


def show_explore_page():
    header = st.container()
    dataset = st.container()
    features = st.container()
    
    st.markdown(
    """
    <style>
    .main {
    background-color: #F0F8FF;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    with header:
        st.title("KIVA Loan Application Process")
        st.text('The reduced dataset has 5 columns')

    image3 = Image.open('newplot.png')
    #st.image(image,caption='Bar Plot of the imbalanced dataset')
    #image2 = Image.open('barplot_balanced.png')
    col1, col2, col3 = st.columns([12,10, 12])
    with col2:
        st.write("   ")
    #imbalanced_barplot = Image.open(image)
    col1.header("Sector of Activity")
    col1.image(image3, width=400)

    grayscale = Image.open('funded_loan_ratio.png')
    col3.header("Funded versus  Expired Loans")
    col3.image(grayscale, width=400)

    with dataset:
        st.dataframe(df)

    with features:
        st.subheader("""
        95% of the loan applications have been funded. The ratio of funded vs non-funded is biased towards funded ones. We will reduced the size of funded to the size of non-funded for machine learning application
        Because of the imbalance of our target feature towards funded application,we have reduced the size of the funded loan for the training of our model
          """)
    original = Image.open('yearly_posted_before.png')
    #st.image(image,caption='Bar Plot of the imbalanced dataset')
    #image2 = Image.open('barplot_balanced.png')
    col1, col2, col3 = st.columns([10, 10, 10])

    #imbalanced_barplot = Image.open(image)
    col1.header("Before features adjustment")
    col1.image(original, width=400)
    with col2:
        st.write("")
    grayscale = Image.open('yearly_posted_after.png')
    col3.header("After features engineering for ML")
    col3.image(grayscale, width=400)
    #Data Set
    data1 = pd.read_csv('yearly_posted_after.csv')
    x=data1.posted_year
 
    y= data1.loan_amount


#The plot
    fig = px.line(        
        data1, #Data Frame
        x = "posted_year", #Columns from the data frame
        y = "loan_amount",
        title = "Line frame"
    )

    page_bg_img = '''
    
    <style>
    body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
    }
    </style>
    '''