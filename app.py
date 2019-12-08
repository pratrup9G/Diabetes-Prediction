# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 12:32:47 2019

@author: BABI
"""
import numpy  as np
import pandas as pa
import matplotlib.pyplot as plt
import seaborn as sn
import pickle 
from flask import Flask,jsonify,request,render_template

app = Flask(__name__)

model = pickle.load(open('xgboost_model','rb'))
scaler = pickle.load(open('scaler','rb'))

@app.route('/')
def home():
    return render_template('diabetes_prediction.html')


@app.route('/predict',methods=["POST"])
def predict():
    age = float(request.form['Age'])
    preg = int(request.form['preg'])
    skin = int(request.form['skin'])
    pressure = int(request.form['pressure'])
    glucose = int(request.form['glucose'])
    BMI = float(request.form['BMI'])
    Diabetes_pedigree = float(request.form['pedigree'])
    
    test_data = pa.DataFrame({'Pregnancies':preg,'Glucose':glucose,'BloodPressure':pressure,'SkinThickness':skin,
                              'BMI':BMI,'DiabetesPedigreeFunction':Diabetes_pedigree,'Age':age},index=[0])
    

    scaled_data= scaler.transform(test_data)

    predict = model.predict(scaled_data)
    
    if (predict == 0):
        return render_template('diabetes_prediction.html',prediction_text='You dont have {}'.format('diabetes'))
    else:
        return render_template('diabetes_prediction.html',prediction_text='You have {}'.format('diabetes'))
            
if __name__ == "__main__":
    app.run(debug=True)        