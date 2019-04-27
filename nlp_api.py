import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def first():
    return '<h1>YOLO</h1>'


@app.route('/some', methods=['POST'])
def getData():
    print("hao")


app.run()
