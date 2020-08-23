from flask import Flask, render_template, url_for, flash, redirect, request
from rf_project import app
from rf_project.forms import predict_UTS_, predict_YS_
import numpy as np
from rf_project import model_uts, model_ys


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')


@app.route("/predict_uts", methods=['GET', 'POST'])
def predict_uts():
    form = predict_UTS_()
    out = 0.0
    if request.method == 'POST':

        a = request.form.get('f_PcntCa')
        b = request.form.get('f_PcntC')
        c = request.form.get('f_PcntNb')
        d = request.form.get('f_PcntMn')
        e = request.form.get('f_PcntN')
        features = [a, b, c, d, e]
        final = [np.array(features)]
        prediction = model_uts.predict(final)
        out = round(prediction[0], 2)

    return render_template('predict_uts.html', title='predict_uts', form=form, prediction_text='The predicted Ultimate Tensile Strength is {}'.format(out))


@app.route("/predict_ys", methods=['GET', 'POST'])
def predict_ys():
    form = predict_YS_()
    out = 0.0
    if request.method == 'POST':

        m = request.form.get('f_PcntCa')
        n = request.form.get('f_PcntTi')
        o = request.form.get('f_PcntNb')
        p = request.form.get('f_PcntMn')
        q = request.form.get('slab_width')
        features_ys = [m, n, o, p, q]
        final_ys = [np.array(features_ys)]
        prediction_ys = model_ys.predict(final_ys)
        out = round(prediction_ys[0], 2)

    return render_template('predict_ys.html', title='predict_ys', form=form, prediction_text='The predicted Yield Strength is {}'.format(out))
