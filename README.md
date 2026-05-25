# Explainable AI Credit Risk Scoring System

## Project Overview

Lecturer @uloko100

This project is an Explainable AI (XAI) Credit Risk Scoring System developed using Machine Learning techniques. The system predicts whether a customer represents a good or bad credit risk based on financial information.

The project uses the German Credit Dataset and applies Explainable AI methods such as SHAP and LIME to improve transparency and interpretability of predictions.

A Streamlit web application was also developed to allow users interact with the prediction system through a simple interface.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SHAP
* LIME
* Streamlit
* Matplotlib

---

# Machine Learning Model

The model used in this project is:

* Random Forest Classifier

The model achieved approximately:

* 77.5% Accuracy

---

# Features Used

The system uses several customer financial features including:

* Account Status
* Loan Duration
* Credit Amount
* Savings Status
* Employment Years
* Housing Information
* Age
* Credit History

---

# Explainable AI Techniques

## SHAP

SHAP was used to identify globally important features influencing model predictions.

## LIME

LIME was used to explain individual predictions and improve interpretability.

---

# Streamlit Deployment

The model was deployed using Streamlit to provide an interactive user interface.

Users can:

* Enter customer information
* Predict credit risk
* View model explanations

---

# Codebase Structure

```text
G5_CreditRisk/
│
├── app.py
├── dataset/
│   └── german_credit_data.csv
├── screenshots/
├── README.md
├── requirements.txt
└── report/
```

---

# Installation

## Clone Repository

```bash
git clone <your-github-repository-link>
```

## Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib shap lime streamlit
```

---

# Run the Application

```bash
python -m streamlit run app.py
```

---

# YouTube Demo Link

[Add YouTube Demo Link Here (https://youtube.com/@group5csc322?si=feWxkEnfiDVEYzEL)]

---

# Deployment Link

[[Add Streamlit Deployment Link Here](https://creditriskscoregit-68pjcfxjvjcckkm8eahxsp.streamlit.app/)]

---

# Screenshots

Add screenshots of:

* Streamlit Interface
* Prediction Results
* SHAP Graph
* Feature Importance Chart

---

# Contributors

Group 5

---

# Final Insight

This project demonstrates how Explainable AI can improve transparency, fairness, and trust in financial machine learning systems.
Responsible AI systems should not only make predictions but also explain why decisions are made.
