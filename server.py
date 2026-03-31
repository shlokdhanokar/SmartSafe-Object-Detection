"""
    Backend engine for the smart safety detection platform.
    Leverages a YOLO-based deep learning model to perform
    real-time inference on webcam frames streamed via WebSocket,
    annotating detected objects with bounding boxes before
    returning the processed feed to the browser client.

    Author  : Shlok Dhanokar
    Project : SmartSafe — Helmet & Package Detection System
"""

import os
import sys
import base64

import cv2
import numpy as np
from flask import Flask, send_file
from flask_socketio import SocketIO
from ultralytics import YOLO


# ───────────────── configuration block ─────────────────

CUSTOM_WEIGHTS   = "best.pt"
FALLBACK_WEIGHTS = "yolov8n.pt"
CONFIDENCE_FLOOR = 0.50          # ignore predictions weaker than this
TARGET_CLASSES   = {"helmet", "BOX"}
JPEG_QUALITY     = 85
SERVER_PORT      = 5000

# ───────────────── model initialisation ────────────────

def pick_model_weights():
    """Choose custom weights when available; otherwise fall back
    to the generic pre-trained checkpoint."""
    if os.path.isfile(CUSTOM_WEIGHTS):
        return CUSTOM_WEIGHTS
    print(f"[INFO] '{CUSTOM_WEIGHTS}' not found — "
          f"falling back to '{FALLBACK_WEIGHTS}'")
    return FALLBACK_WEIGHTS


weights_file   = pick_model_weights()
detector       = YOLO(weights_file)
custom_mode_on = (weights_file == CUSTOM_WEIGHTS)

print(f"[READY] Model loaded from '{weights_file}'  "
      f"(custom={custom_mode_on})")

# ───────────────── flask + socketio setup ──────────────

webapp = Flask(__name__)
ws     = SocketIO(webapp, cors_allowed_origins="*")


@webapp.route("/")
def serve_homepage():
    """Return the front-end page that captures the webcam feed."""
    return send_file("index.html")


# ───────────────── helper utilities ────────────────────

def decode_frame_from_base64(raw_payload):
    """Convert a base64-encoded data-URL string into an OpenCV
    BGR image array."""
    header_removed = raw_payload.split(",", 1)[1]
    binary_blob    = base64.b64decode(header_removed)
    pixel_buffer   = np.frombuffer(binary_blob, dtype=np.uint8)
    bgr_image      = cv2.imdecode(pixel_buffer, cv2.IMREAD_COLOR)
    return bgr_image


def encode_frame_to_base64(cv_image):
    """Compress an OpenCV image to JPEG and wrap it as a base64
    data-URL ready for the browser."""
    encode_params  = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    success, buf   = cv2.imencode(".jpg", cv_image, encode_params)
    if not success:
        return None
    b64_string = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64_string}"


def should_keep_detection(class_name, score):
    """Decide whether a single detection passes our filters.
    In custom-model mode only specific classes matter; with
    the generic model we accept anything above the threshold."""
    if score < CONFIDENCE_FLOOR:
        return False
    if custom_mode_on:
        return class_name in TARGET_CLASSES
    return True


def draw_annotation(canvas, top_left, bottom_right, text):
    """Overlay a rectangle and a label on the given image."""
    green = (0, 255, 0)
    cv2.rectangle(canvas, top_left, bottom_right, green, 2)
    label_pos = (top_left[0], top_left[1] - 8)
    cv2.putText(canvas, text, label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, green, 2)


# ───────────────── websocket event handler ─────────────

@ws.on("video_frame")
def on_incoming_frame(payload):
    """Receive a single webcam frame, run detection, and push
    the annotated result back if anything was found."""
    try:
        image = decode_frame_from_base64(payload)
        if image is None:
            return

        predictions    = detector(image)
        anything_found = False

        for pred in predictions:
            for det in pred.boxes:
                coords     = det.xyxy[0]
                x_min      = int(coords[0])
                y_min      = int(coords[1])
                x_max      = int(coords[2])
                y_max      = int(coords[3])
                cls_index  = int(det.cls[0])
                score      = float(det.conf[0])
                class_name = detector.names[cls_index]

                if not should_keep_detection(class_name, score):
                    continue

                anything_found = True
                tag = f"{class_name} {score:.0%}"
                draw_annotation(image,
                                (x_min, y_min),
                                (x_max, y_max),
                                tag)

        if anything_found:
            out_data = encode_frame_to_base64(image)
            if out_data:
                ws.emit("processed_frame", out_data)

    except Exception as err:
        print(f"[ERROR] Frame processing failed — {err}",
              file=sys.stderr)


# ───────────────── entry point ─────────────────────────

if __name__ == "__main__":
    print(f"[START] Launching server on port {SERVER_PORT} …")
    ws.run(webapp,
           host="0.0.0.0",
           port=SERVER_PORT,
           debug=True,
           allow_unsafe_werkzeug=True)
