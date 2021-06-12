import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from predict import predict_out

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    rate = request.form['rate']
    output = predict_out(rate)[0][0]

    output = np.round(output)

    return render_template('index.html', prediction_text='Predicted Sales should be: $ {}K'.format(output))



if __name__ == "__main__":
    app.run(debug=True)