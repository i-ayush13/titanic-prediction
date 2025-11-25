Titanic Survival Prediction — Logistic Regression

This project builds a binary classification model using the Titanic dataset
to predict whether a passenger survived or not, based on demographic and travel features.

The model is trained using Logistic Regression, evaluated using accuracy score,
and finally deployed through a Flask API with an ngrok tunnel.

📦 Requirements
Python Libraries Used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, request, jsonify
from pyngrok import ngrok

📁 Dataset Used

train.csv from the Titanic Kaggle dataset

Loaded using:

titanic_data = pd.read_csv('/content/train.csv')


It contains:

Column	Description
PassengerId	Unique ID
Survived	Target variable
Pclass	Passenger class
Name	Passenger name
Sex	Gender
Age	Age
SibSp	Siblings/Spouses
Parch	Parents/Children
Ticket	Ticket number
Fare	Ticket fare
Cabin	Cabin number
Embarked	Port of embarkation
🧹 Data Cleaning & Preprocessing
Steps Performed:

✔ checking missing values
✔ dropping irrelevant column Cabin
✔ filling missing Age with mean
✔ filling missing Embarked with mode
✔ encoding categorical columns (Sex, Embarked)
✔ feature selection
✔ splitting into train & test

Encodings applied:

Sex → male=0, female=1  
Embarked → S=0, C=1, Q=2

Features Used For Model:
Pclass, Sex, Age, SibSp, Parch, Fare, Embarked


Target Variable:

Survived

🧠 Model Training

Model Used:

model = LogisticRegression(solver='liblinear', max_iter=1000)


Dataset Split:

train 80%
test  20%


Training accuracy: ~0.80

Testing accuracy: ~0.78

💾 Model Saving
joblib.dump(model, 'logistic_regression_model.pkl')


Resulting file:

logistic_regression_model.pkl

🌐 Deployment (Flask + Ngrok)

REST API endpoints:

GET /

Loads HTML form

POST /predict

Body inputs:

pclass
sex
age
sibsp
parch
fare
embarked


Returns JSON:

{
  "prediction": 0 or 1
}


Ngrok is used to expose the Flask app publicly.

📍 Example Prediction Request
POST /predict


Example:

{
 "pclass": 3,
 "sex": 0,
 "age": 22,
 "sibsp": 1,
 "parch": 0,
 "fare": 7.25,
 "embarked": 0
}


Response:

{"prediction":0}

📌 Future Improvements

hyperparameter tuning

Random Forest & XGBoost comparison

UI enhancement

probability output

model explainability (SHAP)

🏁 Conclusion

This project successfully:

✔ loads + cleans Titanic dataset
✔ trains Logistic Regression model
✔ evaluates accuracy
✔ saves model
✔ deploys API using Flask + ngrok
