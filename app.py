import base64

from flask import Flask, make_response, request

from streaming import (
    convert_img_bytes_to_numpy_array,
    convert_numpy_array_to_img,
    process_frame,
)

app = Flask(__name__)


@app.route('/process', methods=['POST', 'OPTIONS'])
def process():
    if request.method == 'OPTIONS':
        resp = make_response()
    else:
        frame = convert_img_bytes_to_numpy_array(request.files['webcam'].read())
        byte_data = convert_numpy_array_to_img(process_frame(frame))
        decoded_data = base64.b64encode(byte_data).decode('utf-8')
        resp = make_response(decoded_data)
        resp.headers['Content-Type'] = 'text/plain'

    resp.headers['Access-Control-Allow-Origin'] = "*"
    resp.headers['Access-Control-Allow-Headers'] = "*"
    resp.headers['Access-Control-Allow-Methods'] = "*"

    return resp
