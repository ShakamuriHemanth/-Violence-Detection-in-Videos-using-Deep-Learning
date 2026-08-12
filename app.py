import os
import MySQLdb
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash,Response
 
import threading
from werkzeug.utils import secure_filename
import numpy as np
import joblib
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from database import *
import cv2
from pathlib import Path
from sendmail import sendmail
import re 
from geopy.geocoders import Nominatim

app = Flask(__name__)
app.secret_key='detection'
 
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
 


selected_features = ['Human using weapon','non violence','people fighting','theft']
 


@app.route("/")
def home():
    return render_template("index.html")

def get_location_name(latitude, longitude):
    geolocator = Nominatim(user_agent="location_lookup")
    location = geolocator.reverse((latitude, longitude), language='en')
    return location.address


def extract_street_name(location_name):
    # Use regular expression to find the street name pattern
    match = re.search(r'\b(\d+-\d+,?\s)?([\w\s]+),', location_name)
    if match:
        return match.group(2)
    else:
        return None

def get_location_names():
    import json
    import json
    from urllib.request import urlopen
    url='http://ipinfo.io/json'
    response=urlopen(url)
    location=json.load(response)
    print(location)
    latitude, longitude = map(float, location['loc'].split(','))

    print("Latitude:", latitude)
    print("Longitude:", longitude)                              
                             
    # Display
    print(location)
                   
    location_name = get_location_name(latitude, longitude)
    street_name = extract_street_name(location_name)
    if street_name:
        print("Street Name:", street_name)
    else:
        print("Street Name not found.")      
        print("Location Name:", location_name)
        result = json.dumps(location)                                   
    return location_name

@app.route("/registera")
def registera():
    return render_template("register.html")

@app.route("/logina")
def logina():
    return render_template("login.html")
@app.route("/menua")
def menua():
    return render_template("menu.html")

@app.route("/register",methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username=request.form['username']
        email=request.form['email']
        password=request.form['password']
        status = user_reg(username,email,password)
        if status == 1:
            return render_template("/login.html")
        else:
            return render_template("/register.html",m1="failed")        
    

@app.route("/login",methods=['POST','GET'])
def login():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        status = user_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1:                                      
            return render_template("/menu.html", m1="sucess")
        else:
            return render_template("/login.html", m1="Login Failed")
             
# @app.route("/")
# def home():
#     return render_template("index.html")

@app.route('/logouta')
def logout():
    # Clear the session data
    session.clear()
    return redirect(url_for('logina'))






model = load_model('violence.h5')

# Define the activities (classes)
activities = ['Human using weapon','non violence','people fighting','theft'] # Replace with your actual activity labels

# Global variables for video streaming
video_frame = None
video_stream = cv2.VideoCapture()

process_thread = None  # Global variable for the process thread
stop_processing = False  # Flag variable to indicate when to stop processing




def process_video():
    global video_frame, video_stream, stop_processing
    x=0
    y=0
    z=0
    while not stop_processing:
        ret, frame = video_stream.read()
        if not ret:
            break

        # Preprocess the frame (resize and normalize)
        resized_frame = cv2.resize(frame, (64, 64))
        normalized_frame = resized_frame / 255.0

        # Add the batch dimension
        input_frame = np.expand_dims(normalized_frame, axis=0)

        # Make prediction on the input frame
        pred = model.predict(input_frame)
        pred_label = np.argmax(pred)
        activity = activities[pred_label]
        print     
        if activity=="Human using weapon" and x==0:
           loc=get_location_names()
           sendmail("shakamurihemanth17@gmail.com","Human using weapon Detected",loc)
           x=1
        if activity=='people fighting' and y==0:
            loc=get_location_names()
            sendmail("shakamurihemanth17@gmail.com","people fighting Detected",loc)            
            y=1
        if activity=='theft' and z==0:
            loc=get_location_names()
            sendmail("shakamurihemanth17@gmail.com","theft Detected",loc)            
            z=1

        # Draw the predicted activity label on the frame
        cv2.putText(frame, activity, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Update the global video frame for streaming
        video_frame = frame.copy()

@app.route('/predictpage')
def predictpage():
    # Stop the process thread if it's running
    global stop_processing, process_thread

    if process_thread and process_thread.is_alive():
        stop_processing = True
        process_thread.join()
        stop_processing = False

    return render_template('predictpage.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global video_frame

    while True:
        if video_frame is not None:
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', video_frame)
            frame = buffer.tobytes()

            # Yield the frame in the byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/predict', methods=['POST','GET'])
def predict():
    global video_stream, process_thread, stop_processing, video_frame

    # Get the uploaded video file
    video_file = request.files['video']

    # Save the uploaded video file
    video_path = 'static/uploads/' + video_file.filename
    video_file.save(video_path)

    # Release the previous video stream if any
    video_stream.release()

    # Reset the video_frame to None
    video_frame = None

    # Load the video
    video_stream = cv2.VideoCapture(video_path)

    # Stop the process thread if it's running
    if process_thread and process_thread.is_alive():
        stop_processing = True
        process_thread.join()
        stop_processing = False

    # Start processing the video in a separate thread
    process_thread = threading.Thread(target=process_video)
    process_thread.start()

    return render_template('result.html', video_path='/video_feed')


if __name__ == "__main__":
    app.run(debug=True)
