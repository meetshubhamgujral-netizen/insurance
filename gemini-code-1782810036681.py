import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# --- Page Configuration ---
st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Insurance Charges Prediction App")
st.markdown("""
This app predicts medical insurance charges using a **Linear Regression** model based on personal attributes.
""")

# --- Load Data ---
@st.cache_data
def load_data():
    # Looks for the file in the same directory (ideal for GitHub deployment)
    try:
        df = pd.read_csv("InsuranceLR.csv")
        # Standardize column name just in case of typos
        if 'smocker' in df.columns:
            df.rename(columns={'smocker': 'smoker'}, inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: 'InsuranceLR.csv' not found. Please ensure it is uploaded to your GitHub repository.")
        return None

df = load_data()

if df is not None:
    # Drop index column if it exists to clean up features
    if 'index' in df.columns:
        df = df.drop(columns=['index'])

    # --- Preprocessing & Model Training ---
    # Separate features and target
    X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
    y = df['charges']

    # Convert categorical variables into dummy/indicator variables (One-Hot Encoding)
    # This matches the structure needed for both training and prediction inputs
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_columns = X_encoded.columns.tolist()

    # Split and Train Model
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # --- Sidebar: User Inputs for Prediction ---
    st.sidebar.header("📋 Input Client Information")

    input_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    input_sex = st.sidebar.selectbox("Sex", options=df['sex'].unique())
    input_bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    input_children = st.sidebar.slider("Number of Children", min_value=0, max_value=10, value=0)
    input_smoker = st.sidebar.selectbox("Smoker?", options=df['smoker'].unique())
    input_region = st.sidebar.selectbox("Region", options=df['region'].unique())

    # --- Main Panel UI Layout ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Dataset Overview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("⚙️ Model Performance Metrics")
        st.metric(label="R² Score (Accuracy)", value=f"{r2:.2%}")
        st.metric(label="Mean Absolute Error (MAE)", value=f"${mae:,.2f}")

    with col2:
        st.subheader("🔮 Make a Prediction")
        
        # Construct dataframe matching original training feature layout
        input_data = pd.DataFrame([{
            'age': input_age,
            'sex': input_sex,
            'bmi': input_bmi,
            'children': input_children,
            'smoker': input_smoker,
            'region': input_region
        }])

        # Apply identical dummy encoding alignment
        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        # Generate Prediction
        if st.button("Calculate Predicted Charges", type="primary"):
            prediction = model.predict(input_encoded)[0]
            
            # Ensure model doesn't output negative charges realistically
            if prediction < 0:
                prediction = df['charges'].min()

            st.success(f"### Estimated Insurance Charges: **${prediction:,.2f}**")
            
            # Show a brief breakdown of what was sent to the model
            st.markdown("**Submitted Details:**")
            st.json({
                "Age": input_age, "Sex": input_sex, "BMI": input_bmi, 
                "Children": input_children, "Smoker": input_smoker, "Region": input_region
            })