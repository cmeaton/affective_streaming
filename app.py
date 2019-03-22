from flask import Flask, Response, request

from streaming import convert_img_bytes_to_numpy_array, convert_numpy_array_to_img, process_frame

app = Flask(__name__)


@app.route('/process', methods=['POST', 'OPTIONS'])
def process():
    if request.method == 'OPTIONS':
        resp = Response()
    else:
        frame = convert_img_bytes_to_numpy_array(request.files['webcam'].read())
        resp_data = convert_numpy_array_to_img(process_frame(frame))
        resp = Response(resp_data)
        resp.headers['Content-Type'] = 'text/plain'
        # resp.headers['Content-Disposition'] = 'attachment'

    resp.headers['Access-Control-Allow-Origin'] = "*"
    resp.headers['Access-Control-Allow-Headers'] = "*"
    resp.headers['Access-Control-Allow-Methods'] = "*"

    return resp
