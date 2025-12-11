from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

MODEL_PATHS = {
    "logistic": "models/logistic_churn_model.pkl",
    "random_forest": "models/random_forest_churn_model.pkl",
    "gradient_boosting": "models/GB_churn_model.pkl"
}

def load_model(model_key):
    path = MODEL_PATHS.get(model_key)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        selected_model = request.form['model_choice']
        model = load_model(selected_model)

        CreditScore = int(request.form['credit_score'])
        Age = int(request.form['age'])
        Tenure = int(request.form['tenure'])
        Balance = float(request.form['balance'])
        NumOfProducts = int(request.form['numofproducts'])
        HasCrCard = int(request.form['hascrcard'])
        IsActiveMember = int(request.form['isactivemember'])
        EstimatedSalary = float(request.form['estimatedsalary'])
        exited = int(request.form['exited'])

        Geography_France = int(request.form['geography_france'])
        Geography_Germany = int(request.form['geography_germany'])
        Geography_Spain = int(request.form['geography_spain'])
        Gender_Female = int(request.form['gender_female'])

        features = np.array([[CreditScore, Age, Tenure, Balance,
                              NumOfProducts, HasCrCard, IsActiveMember,
                              EstimatedSalary, exited,
                              Geography_France, Geography_Germany,
                              Geography_Spain, Gender_Female]])

        prediction = model.predict(features)[0]

        result = "Customer will churn" if prediction == 1 else "Customer will not churn"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"error: {e}")

if __name__ == "__main__":
    app.run(debug=True)

