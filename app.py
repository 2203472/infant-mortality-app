import math

import pickle

import flask
import pandas as pd
from scipy.stats import stats
from sklearn import preprocessing
import seaborn as sns

from flask import request, url_for, redirect, render_template
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Use pickle to load in the pre-trained model.
with open(f'model/povery-prediction-lightgbm.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__,
                  static_folder='static',
                  template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():

    with open(f'datasets/raw-features.pkl', 'rb') as f:
        df = pickle.load(f)

    if flask.request.method == 'GET':
        return flask.render_template('main.html')

    if flask.request.method == 'POST':
        if request.form['btn'] == 'predict':

            return render_template('main.html')

        if request.form['btn'] == 'Plot':
            region = request.form["region"]
            return redirect(url_for("plot", rgn=region))


@app.route('/<rgn>')
def plot(rgn):

    return render_template('plot.html')


if __name__ == '__main__':
    app.run()
