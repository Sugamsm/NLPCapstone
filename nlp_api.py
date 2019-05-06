import flask
from flask import request, jsonify
import json

from keras.models import load_model
import get_novel_prediction

app = flask.Flask(__name__)
app.config["DEBUG"] = True

novel_model = load_model('./novel/novel_trained_model.model')
novel_data_train_max_length_sequence = 75


@app.route('/', methods=['GET'])
def first():
    return '<h1>HOME</h1>'


@app.route('/novel', methods=['GET', 'POST'])
def getData():
    text = request.args['data']
    prediction = get_novel_prediction.getPrediction(novel_model, text)
    d = {'data': text + prediction}
    return json.dumps(s)


app.run()
