# Owner: Joseph Wong
# Last updated: 3/21/20
import cv2
import sys
import time
import threading

from flask import Flask, render_template, Response
from flask_basicauth import BasicAuth

global live_frames

live_frames = []

# App credentials, update username and password
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'username'
app.config['BASIC_AUTH_PASSWORD'] = 'password'
app.config['BASIC_AUTH_FORCE'] = True

basic_auth = BasicAuth(app)

@app.route('/')
@basic_auth.required
def index():
    return render_template('index.html')

def gen():
    global live_frames

    while True:
        frame = live_frames[-1]
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_live_stream():
    """
    Description: Uploads stream to web app using html/index.html
    Parameters: None
    Return: None
    """

    app.run(host='0.0.0.0', debug=False)
