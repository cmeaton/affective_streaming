from flask import Flask, Response, request

from streaming import process_frame

app = Flask(__name__)


@app.route('/process')
def process():
    if request.method == 'OPTIONS':
        resp = Response("Hello world")
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    # process_frame()
    return 'Hello, Connor!'
