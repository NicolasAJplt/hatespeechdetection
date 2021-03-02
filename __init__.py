#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request


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

            # ler o modelo
            RF_pipeline = joblib.load('model.pkl')
            result = "OK" if RF_pipeline.predict([text])[2] else "Hateful"
            proba  = np.max(RF_pipeline.predict_proba([text]))

            return jsonify(result=result, proba=round(proba*100, 2)), 200
        
        except Exception as erro:
            return jsonify(erro=str(erro)), 500

    return app