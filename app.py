#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib


def create_app(test_config=None):

    app = Flask(__name__)
    
    @app.route("/")
    def index():
        return render_template("index.html")
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            text = request.form.get("text")
            
            model = joblib.load('model.pkl')
            result = "OK" if model.predict([text])[2] else "Hateful"
            proba  = np.max(model.predict_proba([text]))
            
            output=print("It is :", result, "at a proba :", proba)
        
            return render_template('index.html', prediction_text=output)
     
        except Exception as erro:
            return jsonify(erro=str(erro)), 500
        
    return app