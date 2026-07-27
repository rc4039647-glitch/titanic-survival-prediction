# 🚢 Titanic Survival Prediction

## 📌 Project Overview

This project predicts whether a passenger would have survived the Titanic disaster using a Machine Learning model built with Logistic Regression.

The application is deployed using Streamlit and allows users to enter passenger details to receive a survival prediction along with the probability of survival.

---

## 📊 Dataset

- Dataset: Titanic Dataset
- Source: Kaggle

The dataset contains passenger information such as:

- Passenger Class
- Sex
- Age
- Fare
- Number of Siblings/Spouses
- Number of Parents/Children
- Cabin Availability
- Embarked Port

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

---

## 🧠 Machine Learning Model

Model Used:

- Logistic Regression

Feature Engineering:

- Missing value handling
- Title extraction from passenger names
- Family size calculation
- Is Alone feature
- Cabin availability feature
- One-Hot Encoding
- Train-Test Split

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🚀 How to Run the Project

1. Clone the repository.

2. Create a virtual environment.

3. Activate the virtual environment.

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
titanic_survival_prediction_project/
│
├── app.py
├── titanic.csv
├── titanic_model.pkl
├── features.pkl
├── requirements.txt
├── README.md
└── age_distribution.png
```

---

## 🎯 Features

- Interactive Streamlit Interface
- Logistic Regression Prediction
- Survival Probability
- User-Friendly Input Form
- Clean and Simple UI

---

## 📈 Future Improvements

- Add more machine learning models for comparison.
- Deploy the application online.
- Improve the user interface.
- Add visual analytics to the Streamlit app.

---

## 👩‍💻 Author

**Ritikaa Chaurasia**