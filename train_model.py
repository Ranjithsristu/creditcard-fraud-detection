
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load the dataset
DATA_PATH = os.path.join("creditcard", "creditcard_2023.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. "
        "Check the CSV file location."
    )

data = pd.read_csv(DATA_PATH)

# 2. Prepare features and target
if "Class" not in data.columns:
    raise ValueError(
        "Target column 'Class' not found. "
        f"Available columns: {data.columns.tolist()}"
    )

X = data.drop(
    columns=["Class", "id", "Time"],
    errors="ignore"
)
y = data["Class"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create and train the model
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

# 5. Evaluate the model
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))

# 6. Save the trained model as JSON
model.save_model("creditcard_fraud_model.json")
print("\nModel saved successfully!")
