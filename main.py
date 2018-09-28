import logging
import random
import pickle
import os

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_validate


from flask import Flask
from flask import jsonify


MODEL_NAME = 'iris_LR'
TRAIN_PCT = 0.7

app = Flask(__name__)

@app.route('/')
def home():
    """Welcome Screen"""
    return 'Welcome'


@app.route('/ok')
def ok():
    """Health Check"""
    return 'ok'


@app.route('/train_model')
def train_model():
    """Train simple model, save as pickle file."""

    # perform train/test split
    X_train, X_test, y_train, y_test = import_tts()

    # fit the model to the dataset
    logging.info('[main.train_model] fitting model for %s rows', len(X_train))
    model = LR()
    model.fit(X_train, y_train)

    # cross validate
    cv = cross_val(model, X_train, y_train)

    # generate confusion matrix from predictions
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    model_outputs = {
        'algorithm': 'Logistic Regression',
        'raw_accuracy': model.score(X_test, y_test),
        'cross_val_train_acc': np.mean(cv['train_score']),
        'cross_val_test_acc': np.mean(cv['test_score']),
        'coeffs': model.coef_[0].__str__(),
        'confusion_matrix': cm.__str__()
    }

    logging.info('[main.train_model] model_outputs - %s', model_outputs)

    # save model
    filename = '{model_name}.pkl'.format(model_name='iris_LR')
    pickle.dump(model, open(filename, 'wb'))

    return jsonify(model_outputs)

@app.route('/train_RFC')
def train_RFC():

    # perform train/test split
    X_train, X_test, y_train, y_test = import_tts()

    # fit the model to the dataset
    logging.info('[main.train_RFC] fitting RFC for %s rows', len(X_train))
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    # cross validate
    cv = cross_val(rfc, X_train, y_train)

    # generate confusion matrix from predictions
    y_pred = rfc.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    rfc_outputs = {
        'algorithm': 'Random Forest',
        'raw_accuracy': rfc.score(X_test, y_test),
        'cross_val_train_acc': np.mean(cv['train_score']),
        'cross_val_test_acc': np.mean(cv['test_score']),
        'confusion_matrix': cm.__str__()
    }

    logging.info('[main.train_RFC] rfc_outputs - %s', rfc_outputs)

    # save model
    filename = '{model_name}.pkl'.format(model_name='iris_RFC')
    pickle.dump(rfc, open(filename, 'wb'))

    return jsonify(rfc_outputs)

def import_tts(df=None, train_size=TRAIN_PCT):
    """
    Import iris data, train/test split using TRAIN_PCT
    """
    if df is None:
        logging.info('[main.import_tts] getting iris dataframe')
        df = get_iris()

    # define feature set
    X = df.drop('target', 1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    return (X_train, X_test, y_train, y_test)

def cross_val(model, X, y, k=3):
    cv = cross_validate(model, X, y, cv=k, return_train_score=True)
    return cv


@app.route('/predict_random', methods=['GET'])
def predict_random():
    """
    Generate a random series for each of our features and pass that to the
    predict function,  This is used for load testing, to mock out what a live
    API would generate and send to this service
    """

    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    data = [[get_rand_float() for x in range(len(features))]]

    logging.info('[predict_random] %s ', data)

    try:
        prediction = predict(data, features)
    except Exception as err:
        logging.error('[predict_random] err - %s ', err)
        return jsonify({'err': 'err'})

    return jsonify({'predicted_class': prediction[0]})


@app.route('/predict_RFC', methods=['GET'])
def predict_RFC():

    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    data = [[get_rand_float() for x in range(len(features))]]

    logging.info('[predict_RFC] %s ', data)

    try:
        prediction = predict(data, features, 'iris_RFC')
    except Exception as err:
        logging.error('[predict_RFC] err - %s ', err)
        return jsonify({'err': 'err'})

    return jsonify({'predicted_class': prediction[0]})


def predict(data, features, model=MODEL_NAME):
    """load the model from disk, make the prediction"""
    filename = '{model_name}.pkl'.format(model_name=model)
    loaded_model = pickle.load(open(filename, 'rb'))

    test_df = pd.DataFrame(data, columns=features)

    return loaded_model.predict(test_df)


def get_iris():
    """Bring the iris data from the scikit learn package into a dataframe"""

    iris = load_iris()
    rows = np.c_[iris['data'], iris['target']]
    columns = iris['feature_names'] + ['target']
    df = pd.DataFrame(data=rows, columns=columns).dropna()

    df[['target']] = df[['target']].astype(str)

    # map the enum to the species class
    lookup = {'0.0': 'Setosa', '1.0': 'Versicolour', '2.0': 'Virginica'}
    df['target'] = df['target'].map(lambda k: lookup.get(k, ''))

    return df


def get_rand_float():

    return random.uniform(1.0, 4.0)


if __name__ == '__main__':
    # for local development
    app.run(host='127.0.0.1', port=8080, debug=True)

    # # with docker
    # app.run(host='0.0.0.0', port=8080, debug=True)


# https://cloud.google.com/appengine/docs/standard/python3/quickstart
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# https://cloud.google.com/appengine/docs/standard/python3/runtime
