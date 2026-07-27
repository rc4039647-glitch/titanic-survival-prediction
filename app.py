import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("titanic_model.pkl")
features = joblib.load("features.pkl")

# Page Configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction")

st.markdown("""
Predict whether a passenger would have survived the Titanic disaster
using a Machine Learning Logistic Regression model.
""")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

age = st.slider(
    "Age",
    min_value=0,
    max_value=80,
    value=25
)

sibsp = st.number_input(
    "Number of Siblings / Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Number of Parents / Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    max_value=600.0,
    value=30.0
)

has_cabin = st.selectbox(
    "Cabin Available?",
    ["No", "Yes"]
)

title = st.selectbox(
    "Title",
    ["Mr", "Mrs", "Miss", "Master", "Rare"]
)

embarked = st.selectbox(
    "Embarked Port",
    ["C", "Q", "S"]
)

# -----------------------------
# Predict Button
# -----------------------------

if st.button("Predict"):

    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0

    sex = 0 if sex == "male" else 1
    has_cabin = 1 if has_cabin == "Yes" else 0

    data = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "has_cabin": has_cabin,
        "family_size": family_size,
        "is_alone": is_alone,

        "Title_Miss": 0,
        "Title_Mr": 0,
        "Title_Mrs": 0,
        "Title_Rare": 0,

        "Embarked_Q": 0,
        "Embarked_S": 0
    }

    # Title Encoding

    if title == "Miss":
        data["Title_Miss"] = 1

    elif title == "Mr":
        data["Title_Mr"] = 1

    elif title == "Mrs":
        data["Title_Mrs"] = 1

    elif title == "Rare":
        data["Title_Rare"] = 1

    # Embarked Encoding

    if embarked == "Q":
        data["Embarked_Q"] = 1

    elif embarked == "S":
        data["Embarked_S"] = 1

    input_df = pd.DataFrame([data])

    input_df = input_df.reindex(columns=features, fill_value=0)

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.success("✅ Passenger is likely to SURVIVE")
    else:
        st.error("❌ Passenger is likely to NOT SURVIVE")

    st.subheader("Survival Probability")

    st.progress(float(probability))

    st.write(f"**{probability*100:.2f}%**")

    st.subheader("Input Data")

    st.dataframe(input_df)