import flask
from flask import request, jsonify
import json

from keras.models import load_model
import helper_functions

app = flask.Flask(__name__)
app.config["DEBUG"] = True

novel_model = load_model('S:/SUNY/Semester 4/Project/Code/novel/novel_trained_model.model')
novel_data_train_max_length_sequence = 75


print('Reached here')
def getPoemModel():
    print('Called model')
    poem_model = load_model('S:/SUNY/Semester 4/Project/Code/poem/poems_model_local_new.h5py')
    print(poem_model.input_shape)
    return poem_model
poem_model = getPoemModel()
novel_model = None
articles_model = None


@app.route('/', methods=['GET'])
def first():
    return '<h1>HOME</h1>'


@app.route('/novel', methods=['GET', 'POST'])
def getNovel():
    text = request.args['data']
    prediction = getPoemPrediction(novel_model, text)
    d = {'data': text + prediction}
    return json.dumps(s)

@app.route('/poem', methods=['GET', 'POST'])
def getPoems():
    text = request.args['data']
    prediction = helper_functions.getPoemPrediction(text, poem_model)
    d = {'data': text + " " + prediction}
    return json.dumps(d)

app.run()
