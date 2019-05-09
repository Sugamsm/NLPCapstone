import flask
from flask import request, jsonify
import json
import urllib.parse

from keras.models import load_model
import helper_functions

app = flask.Flask(__name__)
app.config["DEBUG"] = True

print('Reached here')
def getPoemModel():
    print('Called model')
    poem_model = load_model('poem/poems_model_local_new.h5py')
    print(poem_model.input_shape)
    return poem_model

def getArticlesModel():
    print('Called model')
    articles_model = load_model('articles/article_headlines.h5py')
    print(articles_model.input_shape)
    return articles_model

def getNovelModel():
    print('Called model')
    novel_model = load_model('novel/novel_trained_model_old.model')
    print(novel_model.input_shape)
    return novel_model
poem_model = getPoemModel()
novel_model = None
article_model = getArticlesModel()


@app.route('/', methods=['GET'])
def first():
    return '<h1>HOME</h1>'


@app.route('/novel', methods=['GET', 'POST'])
def getNovel():
    text = request.args['data']
    text = urllib.parse.unquote(text)
    text = urllib.parse.unquote_plus(text)
    prediction = helper_functions.getNovelPrediction(text, novel_model)
    d = {'data': text + prediction}
    return json.dumps(s)

@app.route('/poem', methods=['GET', 'POST'])
def getPoems():
    text = request.args['data']
    text = urllib.parse.unquote(text)
    text = urllib.parse.unquote_plus(text)
    prediction = helper_functions.getPoemPrediction(text, poem_model)
    d = {'data': text + " " + prediction}
    return json.dumps(d)

@app.route('/articles', methods=['GET', 'POST'])
def getArticles():
    text = request.args['data']
    text = urllib.parse.unquote(text)
    text = urllib.parse.unquote_plus(text)
    prediction = helper_functions.getArticlePrediction(text, article_model)
    d = {'data': text + " " + prediction}
    return json.dumps(d)

app.run()
