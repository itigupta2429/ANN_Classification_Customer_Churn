import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import io
import base64
import requests

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #e6f3ff;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .high-risk {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .low-risk {
        color: #006400;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .input-section {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">📊 Customer Churn Prediction</h1>', unsafe_allow_html=True)

# Create encoded data for models and preprocessing objects
# In a real deployment, you would load these from external files
# For demo purposes, we'll create simple models and preprocessing

@st.cache_resource
def create_demo_model():
    """Create a simple demo model for prediction"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    
    # Create demo preprocessing objects
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(['Female', 'Male'])
    
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    onehot_encoder.fit(np.array(['France', 'Germany', 'Spain']).reshape(-1, 1))
    
    # Create a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Create demo data to train the model
    np.random.seed(42)
    n_samples = 1000
    
    demo_data = {
        'CreditScore': np.random.randint(350, 850, n_samples),
        'Gender': np.random.choice([0, 1], n_samples),
        'Age': np.random.randint(18, 80, n_samples),
        'Tenure': np.random.randint(0, 10, n_samples),
        'Balance': np.random.uniform(0, 250000, n_samples),
        'NumOfProducts': np.random.randint(1, 4, n_samples),
        'HasCrCard': np.random.choice([0, 1], n_samples),
        'IsActiveMember': np.random.choice([0, 1], n_samples),
        'EstimatedSalary': np.random.uniform(0, 200000, n_samples),
        'Geography_France': np.random.choice([0, 1], n_samples),
        'Geography_Germany': np.random.choice([0, 1], n_samples),
        'Geography_Spain': np.random.choice([0, 1], n_samples),
    }
    
    X_demo = pd.DataFrame(demo_data)
    # Create a synthetic target
    y_demo = (
        (X_demo['Balance'] > 100000) & 
        (X_demo['IsActiveMember'] == 0) & 
        (X_demo['NumOfProducts'] == 1)
    ).astype(int)
    
    # Train the model
    model.fit(X_demo, y_demo)
    
    # Create a scaler
    scaler = StandardScaler()
    scaler.fit(X_demo)
    
    return model, scaler, label_encoder, onehot_encoder

# Load the model and preprocessing objects
try:
    # Try to load from external files if available
    # For Streamlit Cloud, we'll use the demo version
    model, scaler, label_encoder, onehot_encoder = create_demo_model()
    st.sidebar.info("Using demo model for prediction")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Creating demo model for prediction")
    model, scaler, label_encoder, onehot_encoder = create_demo_model()

# Input section
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.header("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    geography = st.selectbox("Geography", onehot_encoder.categories_[0])
    gender = st.selectbox('Gender', label_encoder.classes_)
    age = st.slider("Age", min_value=18, max_value=100, value=40)
    credit_score = st.slider("Credit Score", min_value=350, max_value=850, value=650)

with col2:
    balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=50000.0)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=100000.0)
    tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)

with col3:
    number_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)
    has_credit_card = st.selectbox("Has Credit Card", [0, 1], index=1)
    is_active_member = st.selectbox("Is Active Member", [0, 1], index=1)

st.markdown('</div>', unsafe_allow_html=True)

# Prepare the input data with correct feature order
if st.button("Predict Churn", type="primary"):
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [number_of_products],
        'HasCrCard': [has_credit_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode geography
    geo_encoded = onehot_encoder.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    
    # Ensure all expected columns are present
    expected_columns = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
                        'Geography_France', 'Geography_Germany', 'Geography_Spain']
    
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Reorder columns to match training data
    input_data = input_data[expected_columns]

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction_proba = model.predict_proba(input_data_scaled)[0][1]
    
    # Display the prediction
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.subheader("Prediction Result")
    
    if prediction_proba > 0.5:
        st.markdown(f'<p class="high-risk">Churn Risk: HIGH (Probability: {prediction_proba*100:.2f}%)</p>', 
                   unsafe_allow_html=True)
        st.error("This customer has a high risk of churning. Consider retention strategies.")
    else:
        st.markdown(f'<p class="low-risk">Churn Risk: LOW (Probability: {prediction_proba*100:.2f}%)</p>', 
                   unsafe_allow_html=True)
        st.success("This customer has a low risk of churning.")
    
    # Show probability gauge
    st.progress(float(prediction_proba))
    st.caption(f"Churn Probability: {prediction_proba*100:.2f}%")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Add explanation section
with st.expander("How to interpret the results"):
    st.markdown("""
    - **Churn Probability < 50%**: Low risk of customer churn
    - **Churn Probability >= 50%**: High risk of customer churn
    
    **Factors that influence churn risk**:
    - Higher balance customers are more likely to churn
    - Inactive members have higher churn risk
    - Customers with only one product are more likely to leave
    - Geography can influence churn behavior
    """)

# Footer
st.markdown("---")
st.markdown("### 💡 Tips to Reduce Churn")
st.write("""
- **Proactive Support**: Reach out to customers with high balances
- **Engagement Campaigns**: Target inactive members with special offers
- **Product Recommendations**: Suggest additional products to single-product customers
- **Personalized Offers**: Create tailored offers for at-risk customers
""")
