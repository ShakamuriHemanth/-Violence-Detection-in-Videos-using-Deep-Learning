import os
import re
import threading
from urllib.request import urlopen

import cv2
import numpy as np

from flask import (
    Flask,
    Response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from geopy.geocoders import Nominatim
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

from sendmail import sendmail


# ---------------------------------------------------------
# Flask configuration
# ---------------------------------------------------------

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "violence-detection-demo-secret"
)

app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["OUTPUT_FOLDER"] = "static/outputs"

# Ensure folders exist on Render
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)


# ---------------------------------------------------------
# Demo login configuration
# ---------------------------------------------------------

DEMO_USERNAME = os.environ.get("DEMO_USERNAME", "demo")
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "demo123")


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------

MODEL_PATH = "violence.h5"

activities = [
    "Human using weapon",
    "non violence",
    "people fighting",
    "theft",
]

print("Loading violence detection model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# ---------------------------------------------------------
# Global video processing variables
# ---------------------------------------------------------

video_frame = None
video_stream = cv2.VideoCapture()

process_thread = None
stop_processing = False


# ---------------------------------------------------------
# Home
# ---------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------------
# Login / Register
# ---------------------------------------------------------

@app.route("/registera")
def registera():
    return render_template("register.html")


@app.route("/logina")
def logina():
    return render_template("login.html")


@app.route("/menua")
def menua():
    return render_template("menu.html")


@app.route("/register", methods=["POST", "GET"])
def signup():

    # Portfolio/demo deployment does not use a database.
    # Redirect visitors to the demo login.

    return render_template(
        "login.html",
        m1="Demo version: use username demo and password demo123"
    )


@app.route("/login", methods=["POST", "GET"])
def login():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if username == DEMO_USERNAME and password == DEMO_PASSWORD:

            session["user"] = username

            return render_template(
                "menu.html",
                m1="Login successful"
            )

        return render_template(
            "login.html",
            m1="Invalid login. Use demo / demo123"
        )

    return render_template("login.html")


@app.route("/logouta")
def logout():

    session.clear()

    return redirect(url_for("logina"))


# ---------------------------------------------------------
# Location helper
# ---------------------------------------------------------

def get_location_name(latitude, longitude):

    try:

        geolocator = Nominatim(
            user_agent="violence_detection_location_lookup"
        )

        location = geolocator.reverse(
            (latitude, longitude),
            language="en"
        )

        if location:
            return location.address

    except Exception as error:
        print("Location lookup error:", error)

    return "Location unavailable"


def extract_street_name(location_name):

    if not location_name:
        return None

    match = re.search(
        r"\b(\d+-\d+,?\s)?([\w\s]+),",
        location_name
    )

    if match:
        return match.group(2).strip()

    return None


def get_location_names():

    try:

        # This detects the server location on Render,
        # not necessarily the browser user's location.
        response = urlopen(
            "https://ipinfo.io/json",
            timeout=5
        )

        import json

        location = json.load(response)

        coordinates = location.get("loc")

        if not coordinates:
            return "Location unavailable"

        latitude, longitude = map(
            float,
            coordinates.split(",")
        )

        location_name = get_location_name(
            latitude,
            longitude
        )

        return location_name

    except Exception as error:

        print("Unable to determine location:", error)

        return "Location unavailable"


# ---------------------------------------------------------
# Optional alert helper
# ---------------------------------------------------------

def send_detection_alert(subject):

    try:

        location = get_location_names()

        sendmail(
            "shakamurihemanth17@gmail.com",
            subject,
            location
        )

    except Exception as error:

        # Email failure should NOT stop video detection.
        print("Email notification failed:", error)


# ---------------------------------------------------------
# Video processing
# ---------------------------------------------------------

def process_video():

    global video_frame
    global video_stream
    global stop_processing

    weapon_alert_sent = False
    fighting_alert_sent = False
    theft_alert_sent = False

    while not stop_processing:

        ret, frame = video_stream.read()

        if not ret:
            break

        # Resize frame to match model input
        resized_frame = cv2.resize(
            frame,
            (64, 64)
        )

        # Normalize
        normalized_frame = resized_frame.astype(
            "float32"
        ) / 255.0

        # Add batch dimension
        input_frame = np.expand_dims(
            normalized_frame,
            axis=0
        )

        try:

            prediction = model.predict(
                input_frame,
                verbose=0
            )

            prediction_index = int(
                np.argmax(prediction)
            )

            activity = activities[
                prediction_index
            ]

        except Exception as error:

            print("Prediction error:", error)

            activity = "Prediction error"

        print("Detected activity:", activity)

        # -------------------------------------------------
        # Optional email alerts
        # -------------------------------------------------

        if (
            activity == "Human using weapon"
            and not weapon_alert_sent
        ):

            threading.Thread(
                target=send_detection_alert,
                args=("Human using weapon detected",),
                daemon=True,
            ).start()

            weapon_alert_sent = True

        elif (
            activity == "people fighting"
            and not fighting_alert_sent
        ):

            threading.Thread(
                target=send_detection_alert,
                args=("People fighting detected",),
                daemon=True,
            ).start()

            fighting_alert_sent = True

        elif (
            activity == "theft"
            and not theft_alert_sent
        ):

            threading.Thread(
                target=send_detection_alert,
                args=("Theft detected",),
                daemon=True,
            ).start()

            theft_alert_sent = True

        # -------------------------------------------------
        # Draw result on video
        # -------------------------------------------------

        cv2.putText(
            frame,
            activity,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        video_frame = frame.copy()

    video_stream.release()


# ---------------------------------------------------------
# Prediction page
# ---------------------------------------------------------

@app.route("/predictpage")
def predictpage():

    global stop_processing
    global process_thread

    if (
        process_thread
        and process_thread.is_alive()
    ):

        stop_processing = True

        process_thread.join(
            timeout=3
        )

        stop_processing = False

    return render_template(
        "predictpage.html"
    )


# ---------------------------------------------------------
# MJPEG video stream
# ---------------------------------------------------------

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


def generate_frames():

    global video_frame

    while True:

        if video_frame is None:
            continue

        success, buffer = cv2.imencode(
            ".jpg",
            video_frame
        )

        if not success:
            continue

        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )


# ---------------------------------------------------------
# Upload and prediction
# ---------------------------------------------------------

@app.route(
    "/predict",
    methods=["POST", "GET"]
)
def predict():

    global video_stream
    global process_thread
    global stop_processing
    global video_frame

    if request.method == "GET":

        return redirect(
            url_for("predictpage")
        )

    if "video" not in request.files:

        return render_template(
            "predictpage.html",
            error="Please select a video file."
        )

    video_file = request.files["video"]

    if not video_file.filename:

        return render_template(
            "predictpage.html",
            error="Please select a video file."
        )

    filename = secure_filename(
        video_file.filename
    )

    if not filename:

        return render_template(
            "predictpage.html",
            error="Invalid filename."
        )

    video_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    video_file.save(video_path)

    # Stop previous processing thread
    stop_processing = True

    if (
        process_thread
        and process_thread.is_alive()
    ):

        process_thread.join(
            timeout=3
        )

    stop_processing = False

    # Release previous video
    if video_stream.isOpened():
        video_stream.release()

    video_frame = None

    # Load uploaded video
    video_stream = cv2.VideoCapture(
        video_path
    )

    if not video_stream.isOpened():

        return render_template(
            "predictpage.html",
            error="Unable to open the uploaded video."
        )

    # Start processing
    process_thread = threading.Thread(
        target=process_video,
        daemon=True
    )

    process_thread.start()

    return render_template(
        "result.html",
        video_path="/video_feed"
    )



# ---------------------------------------------------------
# Built-in sample video prediction
# ---------------------------------------------------------

@app.route("/predict_sample/<video_name>")
def predict_sample(video_name):

    global video_stream
    global process_thread
    global stop_processing
    global video_frame

    allowed_videos = {
        "p1.mp4",
        "p2.mp4",
        "p3.mp4",
        "p4.mp4",
        "p5.mp4",
        "p6.mp4",
    }

    if video_name not in allowed_videos:
        return "Invalid sample video", 404

    video_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        video_name,
    )

    if not os.path.exists(video_path):
        return "Sample video not found", 404

    # Stop any currently running video process
    stop_processing = True

    if process_thread and process_thread.is_alive():
        process_thread.join(timeout=3)

    stop_processing = False

    # Release previous video stream
    if video_stream.isOpened():
        video_stream.release()

    video_frame = None

    # Open the selected built-in sample video
    video_stream = cv2.VideoCapture(video_path)

    if not video_stream.isOpened():
        return "Unable to open sample video", 500

    # Start processing the sample video
    process_thread = threading.Thread(
        target=process_video,
        daemon=True,
    )

    process_thread.start()

    return render_template(
        "result.html",
        video_path="/video_feed",
    )

# ---------------------------------------------------------
# Health check for Render
# ---------------------------------------------------------

@app.route("/health")
def health():

    return {
        "status": "ok",
        "service": "violence-detection-ai"
    }, 200


# ---------------------------------------------------------
# Local development
# ---------------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
