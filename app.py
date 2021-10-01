# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:01:27 2021

@author: DELL
"""

from flask import Flask, render_template, request
import pickle


# Load the Random Forest CLassifier model

clf = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('tfidf.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Message = request.form['Message']
        data = [Message]
        fact = cv.transform(data).toarray()
        my_prediction = clf.predict(fact)
        
        if(int(my_prediction)==0):
            prediction="WOO! THIS IS A DECEPTIVE REVIEW"
        else:
            prediction="IT IS A NORMAL REVIEW"
        
        return (render_template('index.html', prediction=prediction))


        
if __name__ == '__main__':
    app.run(debug=True)