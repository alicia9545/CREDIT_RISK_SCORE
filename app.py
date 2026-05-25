import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Credit Risk AI System",
    page_icon="💳",
    layout="wide"
)

# --------------------------------
# CUSTOM STYLING
# --------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

h1 {
    color: #003366;
}

h2, h3 {
    color: #004080;
}

.stButton>button {
    background-color: #003366;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# SIDEBAR
# --------------------------------
st.sidebar.title("📘 About Project")

st.sidebar.info("""
Explainable AI Credit Risk Scoring System

Group 5 Project

Technologies Used:
- Random Forest
- SHAP
- Streamlit
- Scikit-learn
""")

# --------------------------------
# TITLE
# --------------------------------
st.title("💳 Explainable AI Credit Risk Scoring System")

st.write(
    "This application predicts whether a customer "
    "represents a good or bad credit risk."
)

# --------------------------------
# LOAD DATASET
# --------------------------------
data = pd.read_csv("german_credit_data.csv")

# --------------------------------
# ENCODE CATEGORICAL DATA
# --------------------------------
label_encoder = LabelEncoder()

object_columns = data.select_dtypes(include=['object']).columns

for column in object_columns:
    data[column] = label_encoder.fit_transform(
        data[column].astype(str)
    )

# --------------------------------
# FEATURES & TARGET
# --------------------------------
X = data.drop('target', axis=1)
y = data['target']

# --------------------------------
# TRAIN TEST SPLIT
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------
# TRAIN MODEL
# --------------------------------
model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------
# MODEL ACCURACY
# --------------------------------
accuracy = model.score(X_test, y_test)

st.metric(
    label="Model Accuracy",
    value=f"{accuracy:.2%}"
)

# --------------------------------
# FEATURE IMPORTANCE CHART
# --------------------------------
st.subheader("📊 Feature Importance")

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=True
)

fig, ax = plt.subplots(figsize=(8,6))

ax.barh(
    importance_df['Feature'][-10:],
    importance_df['Importance'][-10:]
)

ax.set_xlabel("Importance")
ax.set_ylabel("Features")

st.pyplot(fig)

# --------------------------------
# USER INPUT SECTION
# --------------------------------
st.subheader("📋 Customer Financial Information")

col1, col2 = st.columns(2)

with col1:

    status_account = st.selectbox(
        "Status Account",
        [0, 1, 2, 3]
    )

    month_duration = st.slider(
        "Loan Duration (Months)",
        1, 72, 12
    )

    credit_history = st.selectbox(
        "Credit History",
        [0, 1, 2, 3, 4]
    )

    purpose = st.selectbox(
        "Loan Purpose",
        [0, 1, 2, 3, 4, 5, 6, 7]
    )

    credit_amount = st.number_input(
        "Credit Amount",
        min_value=0,
        value=3000
    )

    status_savings = st.selectbox(
        "Savings Status",
        [0, 1, 2, 3, 4]
    )

    years_employment = st.selectbox(
        "Years Employment",
        [0, 1, 2, 3, 4]
    )

    payment_to_income_ratio = st.slider(
        "Payment To Income Ratio",
        1, 4, 2
    )

    status_and_sex = st.selectbox(
        "Status And Sex",
        [0, 1, 2, 3]
    )

    secondary_obligor = st.selectbox(
        "Secondary Obligor",
        [0, 1, 2]
    )

with col2:

    residence_since = st.slider(
        "Residence Since",
        1, 4, 2
    )

    collateral = st.selectbox(
        "Collateral",
        [0, 1, 2, 3]
    )

    age = st.slider(
        "Age",
        18, 75, 30
    )

    other_installment_plans = st.selectbox(
        "Other Installment Plans",
        [0, 1, 2]
    )

    housing = st.selectbox(
        "Housing",
        [0, 1, 2]
    )

    n_credits = st.slider(
        "Number of Credits",
        1, 10, 1
    )

    job = st.selectbox(
        "Job",
        [0, 1, 2, 3]
    )

    n_guarantors = st.slider(
        "Number of Guarantors",
        1, 4, 1
    )

    telephone = st.selectbox(
        "Telephone",
        [0, 1]
    )

    is_foreign_worker = st.selectbox(
        "Is Foreign Worker",
        [0, 1]
    )

# --------------------------------
# PREDICTION BUTTON
# --------------------------------
if st.button("🔍 Predict Risk"):

    input_data = pd.DataFrame([[
        status_account,
        month_duration,
        credit_history,
        purpose,
        credit_amount,
        status_savings,
        years_employment,
        payment_to_income_ratio,
        status_and_sex,
        secondary_obligor,
        residence_since,
        collateral,
        age,
        other_installment_plans,
        housing,
        n_credits,
        job,
        n_guarantors,
        telephone,
        is_foreign_worker
    ]], columns=X.columns)

    prediction = model.predict(input_data)

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Good Credit Risk")
    else:
        st.error("❌ Bad Credit Risk")

    # --------------------------------
# AI EXPLANATION
# --------------------------------
st.subheader("🧠 AI Model Explanation")

st.info(
    """
    The AI model uses important financial factors such as:
    
    - Credit Amount
    - Loan Duration
    - Account Status
    - Savings Status
    - Employment Years
    
    to determine whether a customer is a good or bad credit risk.
    """
)
# --------------------------------
# FOOTER
# --------------------------------
st.markdown("---")

st.caption(
    "Group 5 | Explainable AI Credit Risk Scoring Project"
)
