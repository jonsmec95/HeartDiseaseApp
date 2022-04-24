import numpy as np
import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import altair as alt

from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/heart-disease.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
np.random.seed(42)


def userInputFeatures():
    # create sidebar inputs
    age = st.sidebar.number_input("Age:", 0, 100, 63, help="age in years")
    sex = st.sidebar.selectbox("Sex: ", options=["male", "female"], help="(0 = female, 1 = male)")
    cp = st.sidebar.selectbox("Chest Paint Type:",
                              options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"], help=
                              "chest pain type: "
                              "0.Typical angina: chest pain related decrease blood supply to the heart, "
                              "1.Atypical angina: chest pain not related to heart, "
                              "2.Non-anginal pain: typically esophageal spasms (non heart related), "
                              "3.Asymptomatic: chest pain not showing signs of disease")
    trestbps = st.sidebar.number_input("Resting Blood Pressure:", 50, 150, 145, help="resting blood pressure (in mm Hg "
                                                                                     "on admission to the hospital) anything above 130-140 is typically cause for concern")
    chol = st.sidebar.number_input("Serum Cholesterol in mg/dl:", 100, 300, 233, help="serum cholesterol in mg/d")
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["True", "False"],
                               help="(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)")
    restecg = st.sidebar.selectbox("Resting Electrocardiography results :",
                                   options=["Nothing to note", "ST-T Wave abnormality", "Possible or definite LVF"],
                                   help=
                                   "resting electrocardiography results")
    thalach = st.sidebar.number_input("Maximum heart rate achieved:", 120, 200, 150, help="maximum heart rate achieved")
    exang = st.sidebar.selectbox("Exercise Induced Angina:", options=["Yes", "No"],
                                 help="exercise induced angina (1 = yes; 0 = no)")
    oldpeak = st.sidebar.number_input("ST depression:", 0.1, 4.9, 2.3,
                                      help="ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more")
    slope = st.sidebar.selectbox("Slope:", options=["Upsloping", "Flatsloping", "Downsloping"],
                                 help="the slope of the peak exercise ST segment")
    ca = st.sidebar.number_input("Number of Colored Vessels colored by Fluoroscopy (0-3):", 0, 3,
                                 help="number of major vessels (0-3) colored by fluoroscopy")
    thal = st.sidebar.number_input("Thalium Stress Result (0-3):", 1, 3, help="thallium stress result")

    # convert string inputs
    # sex
    if sex == "male":
        sex_number = 1
    else:
        sex_number = 0

    # cp
    cp_number = 0
    if cp == "Typical angina":
        cp_number = 0
    elif cp == "Atypical angina":
        cp_number = 1
    elif cp == "Non-anginal pain":
        cp_number = 2
    elif cp == "Asymptomatic":
        cp_number = 3

    # fbs
    if fbs == "True":
        fbs_number = 1
    else:
        fbs_number = 0

    # restecg
    restecg_number = 0
    if restecg == "Nothing to note":
        restecg_number = 0
    elif restecg == "ST-T Wave abnormality":
        restecg_number = 1
    elif restecg == "Possible or definite LVF":
        restecg_number = 2

    # restecg
    exang_number = 1
    if exang == "Yes":
        exang_number = 1
    else:
        exang_number = 0

    slope_number = 0
    if slope == "Upsloping":
        slope_number = 0
    elif slope == "Flatsloping":
        slope_number = 1
    elif slope == "Downsloping":
        slope_number = 2

    data = {'age': age,
            'sex': sex_number,
            'cp': cp_number,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs_number,
            'restecg': restecg_number,
            'thalach': thalach,
            'exang': exang_number,
            'oldpeak': oldpeak,
            'slope': slope_number,
            'ca': ca,
            'thal': thal
            }

    features = pd.DataFrame(data, index=[0])
    return features


def featureImportanceChart(x, y):
    clf = LogisticRegression(C=0.20433597178569418,
                             solver="liblinear")
    clf.fit(x, y)
    feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
    feature_df = pd.DataFrame.from_dict(feature_dict, orient='index')

    # feature_df.T.plot.bar(title="Feature Importance", legend=False)
    # st.pyplot()

    st.bar_chart(data=feature_df)


def chestPainChart():
    chart_data = pd.DataFrame(
        df,
        columns=["cp", "age"]
    )

    c = alt.Chart(chart_data).mark_point().encode(
        x='age', y='cp')

    st.altair_chart(c, use_container_width=True)


def rocChart():
    st.line_chart(df)


st.write("""
# Heart Disease Predictor
This app predicts whether a patient has heart disease or not and the risk of them getting it based on their health 
parameters. 
""")

st.sidebar.header("Input Patient's Health Parameters:")
inputted_df = userInputFeatures()

st.subheader("Input Parameters:")
st.write(inputted_df)

loaded_model = pickle.load(open("gs_log_reg.pkl", "rb"))
preds = loaded_model.predict(inputted_df)
probability = loaded_model.predict_proba(inputted_df)

st.subheader("Predicted Heart Disease Result:")
patient_risk_status = ""
test_col1, test_col2 = st.columns(2)
test_col1.write("(0 = Negative, 1 = Positive)")
test_col1.write(preds)
test_col2.write(patient_risk_status)

st.subheader("Prediction Probability Result:")
col1, col2 = st.columns(2)

col1.write(probability)
col2.metric("Probability of Getting Heart Disease", "{:.0%}".format(probability[:, 1][0]))

st.header("About the Data and Test")
st.write("To make accurate predictions, 300 patients health attributes were studied to recognize patterns that lead to "
         "illness. Some of the key characteristics of the data are described below.")

st.subheader("**Feature Importance**")
st.write("To find which health attributes contributed the most to heart disease, we took the coefficient value of each "
         "attribute. We can see below the three strongest factors in heart disease are: **chest pain**, **the slope of "
         "the peak exercise ST segment**, and **the resting electrocardiography results**")
featureImportanceChart(X_train, y_train)

st.subheader("Chest Pain and Age")
st.write("This chart displays the correlation between Chest Pain and Age. The X axis represents Age. The Y axis "
         "represent the four chest pain types: **0.Typical angina, 1.Atypical angina, 2.Non-anginal pain, "
         "3.Asymptomatic**.  ")
chestPainChart()

st.subheader("Prediction Test Accuracy")
st.write("This graph show the area under the curve  which measure's the test's accuracy. With it, we can see  test "
         "results being 96% accurate. ")
st.set_option('deprecation.showPyplotGlobalUse', False)
plot_roc_curve(loaded_model, X_test, y_test)
st.pyplot()


