from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "train.csv"
MODEL_DIR = BASE_DIR / "model"
CLEAN_DIR = BASE_DIR / "cleaned_data"
MODEL_DIR.mkdir(exist_ok=True)
CLEAN_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Could not find train.csv. Put your loan dataset in the project root folder, "
        "beside app.py and train_model.py."
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

required = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area", "Loan_Status"
]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Your CSV is missing expected columns: {missing}")

df = df[required].copy()
for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]:
    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

df["Dependents"] = df["Dependents"].replace({"3+": "3"})
for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Loan_Status"] = df["Loan_Status"].astype(str).str.strip().str.upper()
df = df[df["Loan_Status"].isin(["Y", "N"])].copy()
df.to_csv(CLEAN_DIR / "loan_data_cleaned.csv", index=False)

X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

numeric_features = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History"
]
categorical_features = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

joblib.dump(model, MODEL_DIR / "loan_model.pkl")

print("\nTraining completed successfully.")
print(f"Rows used: {len(df)}")
print(f"Test accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion matrix (label order N, Y):")
print(confusion_matrix(y_test, predictions, labels=["N", "Y"]))
print(f"\nSaved model: {MODEL_DIR / 'loan_model.pkl'}")
print(f"Saved cleaned dataset: {CLEAN_DIR / 'loan_data_cleaned.csv'}")
print("\nReminder: this is a classroom model, not a real lending decision system.")
