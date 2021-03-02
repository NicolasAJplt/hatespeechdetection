#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')



@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form.get("text")
    
    result = "OK" if model.predict([text])[2] else "Hateful"
    proba  = np.max(model.predict_proba([text]))
    
    output=print("It is :", result, "at a proba :", proba)

    return render_template('index.html', prediction_text=output)
 
if __name__ == "__main__":
    from class_def import Preprocess 
    from preprocessing import preprocessing
    app.run(debug=True)