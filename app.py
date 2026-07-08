
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Loan Default Early Warning System", layout="wide")

# Load model and files
model = joblib.load("models/xgb_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")
grade_summary = pd.read_csv("data/grade_summary.csv")
dti_summary = pd.read_csv("data/dti_summary.csv")
feature_importance = pd.read_csv("data/feature_importance.csv")

# Title
st.title("🏦 Loan Default Early Warning System")
st.markdown("Enter borrower details to get an instant default risk assessment.")

# Sidebar — borrower input form
st.sidebar.header("Borrower Details")

grade = st.sidebar.selectbox("Loan Grade", [1, 2, 3, 4, 5, 6, 7], 
                              format_func=lambda x: ["A","B","C","D","E","F","G"][x-1])
sub_grade = st.sidebar.slider("Sub Grade", 1, 35, 10)
int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 30.0, 12.0)
dti = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 50.0, 15.0)
annual_inc = st.sidebar.number_input("Annual Income ($)", 10000, 500000, 60000, step=5000)
term = st.sidebar.selectbox("Loan Term", [36, 60])
small_business = st.sidebar.checkbox("Small Business Loan")

# Build input dictionary
input_data = {
    "grade": grade,
    "sub_grade": sub_grade,
    "int_rate": int_rate,
    "dti": dti,
    "annual_inc": annual_inc,
    "term": term,
    "purpose_small_business": 1 if small_business else 0
}

# Convert to DataFrame and align with training features
input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Predict
risk_prob = model.predict_proba(input_df)[:, 1][0]
risk_pct = round(risk_prob * 100, 2)

# Risk flag
if risk_prob >= 0.5:
    risk_flag = "🔴 HIGH RISK"
    color = "red"
elif risk_prob >= 0.3:
    risk_flag = "🟡 MEDIUM RISK"
    color = "orange"
else:
    risk_flag = "🟢 LOW RISK"
    color = "green"

# Main area — risk score output
col1, col2, col3 = st.columns(3)
col1.metric("Default Probability", f"{round(risk_pct,2)}%")
col2.metric("Risk Flag", risk_flag)
col3.metric("Loan Grade Selected", ["A","B","C","D","E","F","G"][grade-1])

st.markdown("---")

# Charts
st.subheader("📊 Key Risk Insights")
col4, col5 = st.columns(2)

with col4:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(grade_summary["grade"], grade_summary["default_rate"],
            color=["#2ecc71","#27ae60","#f39c12","#e67e22","#e74c3c","#c0392b","#922b21"])
    ax1.set_title("Default Rate by Loan Grade")
    ax1.set_xlabel("Grade")
    ax1.set_ylabel("Default Rate (%)")
    st.pyplot(fig1)

with col5:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(dti_summary["dti_bucket"], dti_summary["default_rate"],
            color=["#2ecc71","#f39c12","#e67e22","#e74c3c"])
    ax2.set_title("Default Rate by DTI Range")
    ax2.set_xlabel("DTI Bucket")
    ax2.set_ylabel("Default Rate (%)")
    ax2.tick_params(axis="x", rotation=15)
    st.pyplot(fig2)

st.subheader("🔍 Top Features Driving Default Risk")
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.barh(feature_importance["feature"].head(10), 
         feature_importance["importance"].head(10), color="#3498db")
ax3.invert_yaxis()
ax3.set_title("Feature Importance — XGBoost Model")
ax3.set_xlabel("Importance Score")
st.pyplot(fig3)

st.subheader("📋 Model Comparison")
model_results = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"],
    "ROC-AUC": [0.65, 0.65, 0.71, 0.73],
    "Recall (Default)": [0.64, 0.66, 0.46, 0.60],
    "Precision (Default)": [0.35, 0.34, 0.41, 0.38]
})
st.dataframe(model_results, use_container_width=True)

st.markdown("---")
st.caption("Built using XGBoost | Lending Club Dataset | 1.3M loans analyzed")
