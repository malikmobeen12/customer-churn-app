# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from utils import load_data

# Page config
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# Load data and model
@st.cache_data
def load_app_data():
    df = load_data()
    model = joblib.load('churn_model.pkl')
    return df, model

df, model = load_app_data()

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
    ["Home", "EDA", "Model Performance", "Predict Churn", "Insights"])

# Home Page
if page == "Home":
    st.title("📱 Customer Churn Prediction & Analytics")
    st.markdown("""
    ### Project Overview
    This dashboard analyzes customer churn patterns and predicts 
    likelihood of customers leaving a telecom service provider.
    
    **Dataset**: Telco Customer Churn (7,043 customers with 21 features)
    
    **Key Features**:
    - Exploratory data analysis
    - Machine learning model performance metrics
    - Churn prediction for individual customers
    - Actionable business insights
    
    **Models Used**:
    - Logistic Regression
    - Random Forest
    - XGBoost (Best Performing)
    """)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

# EDA Page
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    
    # Churn distribution
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df, ax=ax)
    ax.set_title("Churn vs Non-Churn Customers")
    st.pyplot(fig)
    
    # Demographic analysis
    st.subheader("Demographic Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(x='gender', hue='Churn', data=df, ax=ax)
        ax.set_title("Churn by Gender")
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x='SeniorCitizen', hue='Churn', data=df, ax=ax)
        ax.set_title("Churn by Senior Citizen Status")
        st.pyplot(fig)
    
    # Contract vs Churn
    st.subheader("Contract Impact on Churn")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Contract', hue='Churn', data=df, ax=ax)
    ax.set_title("Churn by Contract Type")
    st.pyplot(fig)
    
    # Tenure vs Churn
    st.subheader("Tenure Impact on Churn")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='tenure', hue='Churn', kde=True, 
                bins=30, multiple="stack", ax=ax)
    ax.set_title("Tenure Distribution by Churn Status")
    st.pyplot(fig)
    
    # Correlation matrix
    st.subheader("Correlation Matrix")
    numeric_df = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)

# Model Performance Page
elif page == "Model Performance":
    st.title("🤖 Model Performance")
    
    # Metrics table
    st.subheader("Evaluation Metrics")
    metrics = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Accuracy': [0.80, 0.79, 0.85],
        'Precision': [0.67, 0.63, 0.74],
        'Recall': [0.52, 0.50, 0.58],
        'F1 Score': [0.58, 0.56, 0.65],
        'ROC AUC': [0.84, 0.83, 0.89]
    })
    st.dataframe(metrics.style.highlight_max(color='lightgreen', axis=0))
    
    # Confusion matrix
    st.subheader("XGBoost Confusion Matrix")
    st.image("confusion_matrix.png", width=500)
    
    # ROC curves
    st.subheader("ROC Curves")
    st.image("roc_curve.png", width=600)

elif page == "Predict Churn":
    st.title("🔮 Predict Customer Churn")
    st.markdown("Enter customer details to predict churn probability")
    
    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", 
                ["Month-to-month", "One year", "Two years"])
            paperless = st.radio("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", 
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
            senior = st.radio("Senior Citizen", ["Yes", "No"])
            
        with col2:
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
            total = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
            internet = st.selectbox("Internet Service", 
                ["DSL", "Fiber optic", "No"])
            services = st.multiselect("Additional Services", 
                ["Online Security", "Online Backup", "Device Protection", 
                 "Tech Support", "Streaming TV", "Streaming Movies"])
        
        submit = st.form_submit_button("Predict Churn")
    
    if submit:
        # Compute derived features
        service_count = len(services)
        monthly_tenure_ratio = monthly / (tenure + 1)
        total_monthly_ratio = total / (monthly + 1)
        
        # Map SeniorCitizen to 0/1
        senior_citizen = 1 if senior == "Yes" else 0
        
        # Create TenureGroup (same bins as in model training)
        if tenure <= 12:
            tenure_group = '0-1Y'
        elif tenure <= 24:
            tenure_group = '1-2Y'
        elif tenure <= 48:
            tenure_group = '2-4Y'
        else:
            tenure_group = '4-6Y'
        
        # Prepare input data with all required features
        input_data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly],
            'TotalCharges': [total],
            'gender': ['Male'],   # default
            'SeniorCitizen': [senior_citizen],
            'Partner': ['No'],    # default
            'Dependents': ['No'], # default
            'PhoneService': ['Yes'], # default
            'MultipleLines': ['No'], # default
            'InternetService': [internet],
            'OnlineSecurity': ['Yes' if "Online Security" in services else 'No'],
            'OnlineBackup': ['Yes' if "Online Backup" in services else 'No'],
            'DeviceProtection': ['Yes' if "Device Protection" in services else 'No'],
            'TechSupport': ['Yes' if "Tech Support" in services else 'No'],
            'StreamingTV': ['Yes' if "Streaming TV" in services else 'No'],
            'StreamingMovies': ['Yes' if "Streaming Movies" in services else 'No'],
            'Contract': [contract],
            'PaperlessBilling': [paperless],
            'PaymentMethod': [payment],
            'ServiceCount': [service_count],
            'MonthlyTenureRatio': [monthly_tenure_ratio],
            'TotalMonthlyRatio': [total_monthly_ratio],
            'TenureGroup': [tenure_group]  # ADDED THIS LINE
        })
        
        # Make prediction
        proba = model.predict_proba(input_data)[0][1]
        prediction = "Churn" if proba > 0.5 else "Retain"
        
        # Display results
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        col1.metric("Churn Probability", f"{proba:.1%}")
        col2.metric("Prediction", prediction, 
                   delta="High Risk" if proba > 0.5 else "Low Risk")
        
        # Visual indicator
        st.progress(proba)
        if proba > 0.7:
            st.warning("⚠️ High churn risk! Recommend retention offers")
        elif proba > 0.5:
            st.warning("⚠️ Moderate churn risk")
        else:
            st.success("✅ Low churn risk")

# Insights Page
elif page == "Insights":
    st.title("💡 Business Insights & Recommendations")
    
    st.subheader("Key Findings")
    st.markdown("""
    - **Churn Rate**: 26.5% of customers churned
    - **High Risk Groups**:
        - Month-to-month contract customers (43% churn rate)
        - Fiber optic internet users (41% churn rate)
        - Electronic check payment users (34% churn rate)
    - **Protective Factors**:
        - Two-year contracts (11% churn rate)
        - Customers with tech support (15% churn rate)
        - Bank transfer/credit card users (16-19% churn rate)
    - **Tenure Impact**: 80% of churn occurs in first 20 months
    """)
    
    st.subheader("Actionable Recommendations")
    st.markdown("""
    1. **Contract Incentives**:
        - Offer discounts for switching to 1/2-year contracts
        - Create graduated discount programs for long-term customers
    
    2. **Payment Method Optimization**:
        - Promote automatic payment methods with $5 monthly discount
        - Charge processing fee for paper bills
    
    3. **Service Bundling**:
        - Bundle internet with tech support at reduced rates
        - Create "Streaming Plus" package with TV/movies
    
    4. **High-Risk Interventions**:
        - Proactive outreach to fiber optic users at 12-month mark
        - Loyalty rewards program at 6/12/18 month milestones
    
    5. **Customer Experience**:
        - Implement predictive churn scoring for targeted retention
        - Create VIP program for customers with >3 services
    """)
    
    st.subheader("Financial Impact Projection")
    st.markdown("""
    | Initiative | Cost | Expected Churn Reduction | Projected Annual Savings |
    |---|---|---|---|
    | Contract incentives | $120K | 15% | $780K |
    | Payment method optimization | $80K | 8% | $416K |
    | Tech support bundles | $200K | 12% | $624K |
    | **Total** | **$400K** | **35%** | **$1.82M** |
    """)