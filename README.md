# LoanLens — Loan Approval Prediction Website

A student machine-learning project using Flask, HTML, CSS, JavaScript, pandas and scikit-learn.

## Important first step: add the dataset

Place your `train.csv` file in this folder, beside `app.py` and `train_model.py`.
The expected target column is `Loan_Status`, with `Y` and `N` labels. The expected input columns are listed in `train_model.py`.

## Run locally on Windows

1. Install Python 3.10 or newer and VS Code.
2. Open this project folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt in VS Code and run:
   `.\.venv\Scripts\activate.bat`

5. Install packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Train the model (creates `model/loan_model.pkl`):

   ```powershell
   python train_model.py
   ```

7. Start the website:

   ```powershell
   python app.py
   ```

8. Open `http://127.0.0.1:5000` in your browser. Keep the terminal open while testing.

## Project layout

```text
loan_approval_website/
├── app.py
├── train_model.py
├── requirements.txt
├── Procfile
├── train.csv                 # add your dataset here; do not upload private data
├── model/
│   └── loan_model.pkl        # generated after training
├── cleaned_data/
│   └── loan_data_cleaned.csv # generated after training
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## Deploy to Render

1. First test the project locally.
2. Create a GitHub repository and upload the project files.
3. Do not commit `.venv`, private information, or confidential applicant data.
4. In Render, create **New → Web Service** and connect your GitHub repository.
5. Set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
6. Deploy and open the public `onrender.com` URL.

### Model file note
This beginner setup expects `model/loan_model.pkl` to be present when the app starts. Because `.pkl` files are ignored by `.gitignore`, either remove the `model/*.pkl` ignore line after creating the model if you intend to commit this educational model, or adapt deployment to train/download the model during build/start. Only commit a model you created and trust. For a class project with a small model, committing the trained model file is the simplest approach; do not commit the dataset if it contains private or restricted data.

## Model and limitations

The script uses a train/test split and a Logistic Regression pipeline with median/mode imputation, scaling and one-hot encoding. It prints test accuracy and a classification report. Results depend on the exact dataset and split. A prediction is not a real bank decision or a reliable assessment of an actual person's creditworthiness; do not collect real sensitive financial information in this student demo.
