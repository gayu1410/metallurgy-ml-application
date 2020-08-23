from flask import Flask
# import sklearn
import pickle

app = Flask(__name__)

model_uts = pickle.load(open('model_rfUTS.pkl', 'rb'))
model_ys = pickle.load(open('model_rfYS.pkl', 'rb'))

app.config['SECRET_KEY'] = 'UatJ0YQtSJ-3LobiUHJKgSmRxlg-Awr7d4Qk5t9Vy5s'

from rf_project import routes
