from flask import Flask, render_template, request, json


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
import json
import matplotlib.pyplot as plt

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('upload.html')
    #
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
        return render_template('upload.html', shape=df.shape)
    return "<p>Hello, World!</p>"


if __name__ == '__app__':
    app.run(debug=True)
