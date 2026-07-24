import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Heart Disease Prediction",page_icon="❤️",layout="wide")

df = pd.read_csv("framingham.csv")
model = joblib.load("Heart_Disease_Prediction_Model.pkl")
scaler = joblib.load("scaler.pkl")

st.sidebar.title("❤️ Heart Disease Prediction")
page = st.sidebar.radio("Navigation",["🏠 Home","❤️ Prediction","📊 Dataset","📈 Visualization","ℹ️ About"])

if page=="🏠 Home":
    st.title("Heart Disease Prediction")
    st.write("Predict 10-year CHD risk using Machine Learning.")

elif page=="❤️ Prediction":
    c1,c2=st.columns(2)
    with c1:
        male=st.selectbox("Gender",[0,1],format_func=lambda x:"Female" if x==0 else "Male")
        age=st.number_input("Age",20,100,40)
        education=st.selectbox("Education",[1,2,3,4])
        currentSmoker=st.selectbox("Current Smoker",[0,1])
        cigsPerDay=st.number_input("Cigarettes Per Day",0,100,0)
        BPMeds=st.selectbox("BP Medication",[0,1])
        prevalentStroke=st.selectbox("Stroke",[0,1])
        prevalentHyp=st.selectbox("Hypertension",[0,1])
    with c2:
        diabetes=st.selectbox("Diabetes",[0,1])
        totChol=st.number_input("Total Cholesterol",100,700,200)
        sysBP=st.number_input("Systolic BP",80,250,120)
        diaBP=st.number_input("Diastolic BP",40,150,80)
        BMI=st.number_input("BMI",10.0,60.0,25.0)
        heartRate=st.number_input("Heart Rate",30,200,75)
        glucose=st.number_input("Glucose",40,400,90)
    if st.button("Predict"):
        X=[[male,age,education,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,
            prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose]]
        X=scaler.transform(X)
        pred=model.predict(X)[0]
        st.success("✅ Low Risk of Heart Disease" if pred==0 else "⚠️ High Risk of Heart Disease")

elif page=="📊 Dataset":
    st.dataframe(df.head())
    st.write(df.describe())

elif page=="📈 Visualization":
    fig,ax=plt.subplots(figsize=(10,8))
    sns.heatmap(df.corr(numeric_only=True),cmap="coolwarm",ax=ax)
    st.pyplot(fig)
    fig2,ax2=plt.subplots()
    sns.countplot(data=df,x="TenYearCHD",ax=ax2)
    st.pyplot(fig2)
else:
    st.write("Heart Disease Prediction using Logistic Regression.")
