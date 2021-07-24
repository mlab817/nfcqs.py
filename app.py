from flask import Flask, render_template, request, json
from models.ltt import run_ols
from models.arima import run_arima


import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import statsmodels.api as sm
import json
import matplotlib.pyplot as plt


port = int(os.environ.get('PORT', 5000))


app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def index():
    response = None
    if request.method == 'GET':
        response = render_template('upload.html')
    #
    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        # validate the uploaded files if it contains 5 columns

        df = pd.read_csv(uploaded_file,
                         thousands=',',
                         header='infer', )

        if len(df.axes[1]) != 5:
            return 'File must have 5 columns: year, production, area harvested, yield, and per capita. Your file has %s' % str(len(df.axes[1]))

        prepared_data = prepare_data(df)

        ols_result = run_ols(data=prepared_data, x_var='yield', y_var='year', log=True)

        if request.is_json:
            response = json.dumps(ols_result, indent=2).replace('NaN', 'null')
        else:
            response = ols_result

        print(run_arima(df, 'yield'))

    return response


# TODO: Make this prepare data dynamic
def prepare_data(data=pd.DataFrame()):
    """
    Prepare uploaded data for processing by defining the correct data types

    :param data:
    :return: prepared data
    """
    data.columns = ["year", "production", "area_harvested", "yield", "per_capita"]

    # convert data to float as sometimes the data is not of correct type (e.g. string instead of float)
    data["production"] = pd.to_numeric(data["production"])
    data["area_harvested"] = pd.to_numeric(data["area_harvested"])
    data["yield"] = pd.to_numeric(data["yield"])
    data["per_capita"] = pd.to_numeric(data["per_capita"])

    # verify data types
    data.describe()

    return data


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
