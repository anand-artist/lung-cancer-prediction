import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

st.title("🩺 Decision Tree Classifier for Lung Cancer Prediction")

# === Upload training dataset ===
st.header("📂 Upload Training Dataset")
train_file = st.file_uploader("Upload your lung-cancer.csv", type=["csv"])

if train_file is not None:
    data_df = pd.read_csv(train_file)
    st.write("### Training Data Preview:")
    st.dataframe(data_df.head())

    # --- Check for target column ---
    if 'target' not in data_df.columns:
        st.error("❌ Dataset must include a 'target' column.")
    else:
        # Split data
        x = data_df.drop('target', axis=1)
        y = data_df['target']
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=30
        )

        # Train Decision Tree
        DT = DecisionTreeClassifier(criterion='entropy', random_state=42)
        DT.fit(x_train, y_train)

        # Evaluate
        y_pred = DT.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model trained successfully! Accuracy on test split: *{accuracy:.2f}*")

        # === Upload new data for prediction ===
        st.header("🔍 Upload New Data for Prediction")
        test_file = st.file_uploader("Upload test.csv", type=["csv"])

        if test_file is not None:
            test_df = pd.read_csv(test_file)
            st.write("### Test Data Preview:")
            st.dataframe(test_df.head())

            # Drop target column if exists
            if 'target' in test_df.columns:
                X_new = test_df.drop('target', axis=1)
            else:
                X_new = test_df

            # --- Handle feature name mismatches ---
            model_features = list(DT.feature_names_in_)
            new_features = list(X_new.columns)

            missing = [f for f in model_features if f not in new_features]
            extra = [f for f in new_features if f not in model_features]

            if missing or extra:
                st.warning(f"""
                ⚠ Feature name mismatch detected!
                - Missing features (expected by model): {missing}
                - Extra features (not seen during training): {extra}
                """)

                # Common typo fixes (e.g., 1 ↔ l confusion)
                rename_map = {
                    'co1pactness_se': 'compactness_se',
                    'co1pactness_worst': 'compactness_worst',
                    'concavity_1ean': 'concavity_mean',
                    'fractal_di1ension_1ean': 'fractal_dimension_mean',
                    'fractal_di1ension_se': 'fractal_dimension_se',
                    # Add other known typos if needed
                }

                # Reverse map (since model was trained with typos)
                reverse_map = {v: k for k, v in rename_map.items()}
                X_new = X_new.rename(columns=reverse_map)

            # --- Final alignment with model features ---
            try:
                X_new = X_new[DT.feature_names_in_]
                predictions = DT.predict(X_new)
                st.success("✅ Prediction completed successfully!")
                st.dataframe(pd.DataFrame(predictions, columns=['Predicted Target']))
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
