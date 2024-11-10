import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from metrics import calculate_demographic_parity, calculate_equalized_odds
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_bias', methods=['POST'])
def check_bias():
    # Get user inputs from the form
    y_true = list(map(int, request.form['y_true'].split(',')))
    y_pred = list(map(int, request.form['y_pred'].split(',')))
    protected_attribute = list(map(int, request.form['protected_attribute'].split(',')))
    
    # Calculate bias metrics
    accuracy = accuracy_score(y_true, y_pred)
    demographic_parity = calculate_demographic_parity(y_true, y_pred, protected_attribute)
    equalized_odds = calculate_equalized_odds(y_true, y_pred, protected_attribute)
    
    # Prepare results
    results = {
        "accuracy": accuracy,
        "demographic_parity": demographic_parity,
        "equalized_odds": equalized_odds
    }
    
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
