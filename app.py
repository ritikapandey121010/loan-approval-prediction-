from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "loan_model.pkl"

app = Flask(__name__)

def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        if model is None:
            error = "The ML model has not been trained yet. Add train.csv and run: python train_model.py"
        else:
            try:
                dependents = request.form.get("Dependents", "0")
                if dependents == "3+":
                    dependents = "3"

                applicant = {
                    "Gender": request.form["Gender"],
                    "Married": request.form["Married"],
                    "Dependents": dependents,
                    "Education": request.form["Education"],
                    "Self_Employed": request.form["Self_Employed"],
                    "ApplicantIncome": float(request.form["ApplicantIncome"]),
                    "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
                    "LoanAmount": float(request.form["LoanAmount"]),
                    "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
                    "Credit_History": float(request.form["Credit_History"]),
                    "Property_Area": request.form["Property_Area"],
                }
                input_df = pd.DataFrame([applicant])
                prediction = str(model.predict(input_df)[0]).strip().upper()
                result = "Likely to be approved" if prediction in ("Y", "1", "APPROVED") else "Likely not to be approved"
            except (KeyError, ValueError) as exc:
                error = "Please complete every field using valid values. Income and loan fields must be numbers."
            except Exception:
                error = "Something went wrong while making the prediction. Check the model and input fields."

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
