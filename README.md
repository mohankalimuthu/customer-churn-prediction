# Customer Churn Prediction
This project predicts whether a customer will churn (leave a service) based on multiple features.
The solution includes:

* A trained Machine Learning model

* A Flask web application

* An HTML interface for user input

* Deployment-ready structure (works on Render)

## Project Overview
Customer churn is one of the biggest challenges in subscription-based businesses.
This project uses a Machine Learning model to analyze customer data and predict if they are likely to churn.
The app provides:

* A user-friendly web interface

* Real-time predictions

* A clean and simple deployment pipeline

## Project Structure
```
project/
│── app.py
│── requirements.txt
│── models/
│     └── churn_model.pkl
│── templates/
│     └── index.html
│── static/
      └── (CSS/JS files)
```

## Technologies Used

### Machine Learning

* Python

* NumPy

* Scikit-Learn

### Web Framework

* Flask (for backend)

* HTML/CSS (for UI)

### Deployment

* Render (cloud hosting)

* Gunicorn (production server)

## How It Works

* User enters input values (Age, Income, etc.) in the HTML form

* Flask receives the data and converts it into model-readable format

* Machine Learning model (.pkl file) predicts churn

* The result is displayed on the webpage

## Model Training (Notebook)

All data preprocessing, model training, and evaluation are done inside the Jupyter Notebook.
The final trained model is saved as:

```
models/churn_model.pkl
```

## Running the Project Locally

1. Install dependencies
  ```
pip install -r requirements.txt
```

2. Run the Flask app
  ```
python app.py
```

3. Open in browser
```
http://127.0.0.1:5000/
```

## Deployment on Render

* Push project to GitHub

* Create a new Web Service on Render

* Set commands:

Build command:
```
pip install -r requirements.txt
```
Start command:
```
gunicorn app:app
```

4. Deploy and get a live URL
Requirements
```
Flask
gunicorn
numpy
scikit-learn
```

## Features

✔️ Simple, responsive UI

✔️ Real-time churn prediction

✔️ Clean code and easy to maintain

✔️ Fully deployable ML system

✔️ Beginner-friendly end-to-end project

## Future Improvements

* Add more input features

* Improve model accuracy

* Add user login system

* Store predictions in database

## Contributing

Contributions are welcome!

If you want to improve the model or add visual dashboards, open a pull request.

## License

This project is open-source under the MIT License.

## Support

If you like this project, please star ⭐ the repository —> **it motivates further development!**
