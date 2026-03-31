# SmartSafe-Object-Detection

**Author:** Shlok Dhanokar

A browser-based safety monitoring tool that performs **real-time object detection** through a webcam feed. The backend runs a YOLOv8 deep learning model and streams annotated frames back to the frontend over WebSocket.

## Overview

| Component | Technology |
|-----------|-----------|
| Detection model | YOLOv8 (Ultralytics) |
| Backend | Flask + Flask-SocketIO |
| Frontend | Vanilla HTML / JS with Socket.IO client |
| Image processing | OpenCV + NumPy |

The system was developed to detect **helmets** and **packages/boxes** in live video — applicable for safety compliance checks at construction sites, warehouses, and logistics hubs.

## How It Works

1. The browser captures frames from the user's webcam using the MediaDevices API
2. Each frame is base64-encoded and transmitted to the backend via WebSocket
3. The server decodes the image, runs YOLOv8 inference, and overlays bounding boxes on detected objects
4. Annotated frames are pushed back to the browser and rendered beside the raw feed

## Setup

```bash
pip install flask flask-socketio opencv-python ultralytics numpy
```

## Running

```bash
python server.py
```

Then open **http://localhost:5000** in a browser and click *Start Camera*.

## Model Weights

Place your trained `best.pt` weights file in the project root. If the file is absent, the server automatically uses the pre-trained `yolov8n.pt` checkpoint (80-class COCO detector) as a fallback.

### Training your own model

```python
from ultralytics import YOLO

net = YOLO("yolov8n.pt")
net.train(data="path/to/data.yaml", epochs=50, imgsz=640, batch=16)

import shutil
shutil.copy("runs/detect/train/weights/best.pt", "./best.pt")
```

## Project Structure

```
├── server.py        # Flask-SocketIO backend with YOLO inference
├── index.html       # Frontend webcam capture & display
├── best.pt          # Custom trained weights (user-provided)
└── README.md
```

## License

For educational purposes only.
