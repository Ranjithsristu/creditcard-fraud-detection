
from pathlib import Path

import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "creditcard_fraud_model.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = XGBClassifier()
    model.load_model(str(model_path))
    return model


st.title("💳 Credit Card Fraud Detection")
st.write(
    "Upload transaction data to identify potentially fraudulent transactions."
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if data.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()

        # Get the feature names used during model training
        expected_features = model.get_booster().feature_names

        if not expected_features:
            st.error(
                "The model does not contain feature names. "
                "Define the exact training feature columns manually."
            )
            st.stop()

        # Check whether all required columns exist
        missing_features = [
            col for col in expected_features
            if col not in data.columns
        ]

        if missing_features:
            st.error(
                f"Missing required feature columns: {missing_features}"
            )
            st.stop()

        # Select features in the model's expected order.
        # Extra columns such as id, Class, or Time are ignored
        # unless the model was trained using them.
        X = data.loc[:, expected_features].copy()

        # Ensure that model inputs are numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        if X.isna().any().any():
            st.error(
                "Some required features contain missing or "
                "non-numeric values. Please clean the CSV."
            )
            st.stop()

        # Predict fraud probabilities
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results = data.copy()
        results["Fraud Probability"] = probabilities
        results["Prediction"] = [
            "Fraud" if p == 1 else "Legitimate"
            for p in predictions
        ]

        st.subheader("Prediction Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", len(results))
        col2.metric("Potential Fraud", int(predictions.sum()))
        col3.metric(
            "Legitimate",
            int((predictions == 0).sum())
        )

        st.dataframe(results, use_container_width=True)

        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Predictions",
            data=csv_data,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
