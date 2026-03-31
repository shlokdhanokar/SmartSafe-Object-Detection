# SmartSafe-Object-Detection

**Author:** Shlok Dhanokar

A robust, real-time safety monitoring tool that leverages deep learning to perform **live object detection** through a webcam feed. The backend is powered by a YOLOv8 neural network, streaming annotated frames back to an interactive frontend via WebSocket Protocol.

This project was developed from scratch to detect **helmets** and **packages/boxes** in live video — a practical application for safety compliance checks at construction sites, warehouses, and logistics hubs.

---

## 🚀 Execution & Setup Guide

The project is designed to be fully executable strictly through the command line. Follow these step-by-step instructions to set up the environment and run the application. Assume Windows PowerShell or standard Unix terminals (Bash/Zsh).

### 1. Prerequisites
Ensure you have the following installed on your system:
- **Python 3.8+** (Verify with `python --version`)
- **Git** (Verify with `git --version`)
- A working webcam connected to your machine.

### 2. Clone the Repository
Open your terminal and clone the project to your local machine:
```bash
git clone https://github.com/shlokdhanokar/SmartSafe-Object-Detection.git
cd SmartSafe-Object-Detection
```

### 3. Environment Setup (Recommended)
It is highly recommended to isolate dependencies using a virtual environment.
```bash
# Create the virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

### 4. Dependency Installation
Once inside the project root (and virtual environment), install all required Python libraries.
```bash
pip install flask flask-socketio opencv-python ultralytics numpy
```

### 5. Configuration (Model Weights)
The application relies on YOLOv8 weights to perform inference. 
- The required file `yolov8n.pt` will automatically download on the first run if not present.
- If you have custom trained weights (e.g., `best.pt`), place the file in the root directory. The application is configured to automatically detect and prioritize custom weights over the generic fallback model.

### 6. Execution
Start the backend WebSocket server directly from your terminal:
```bash
python server.py
```
*You should see console output confirming the model is loaded and the Flask server is running on `http://0.0.0.0:5000`.*

### 7. Accessing the Application
1. Open your web browser (Chrome/Edge/Firefox).
2. Navigate to: **http://localhost:5000**
3. Click the **"Start Camera"** button on the interface. Grant camera permissions if prompted.
4. The raw feed will appear, followed instantly by the processed feed showing bounding boxes and confidence scores around detected objects.

---

## 🏗️ Project Architecture & Technologies

| Layer | Component/Technology | Purpose |
|-------|----------------------|---------|
| **AI Model** | YOLOv8 (Ultralytics) | Single-shot deep learning object detection for high-speed inference. |
| **Backend** | Flask & Flask-SocketIO | Serves the UI and maintains a persistent bi-directional WebSocket connection. |
| **Frontend** | Vanilla HTML5, CSS3, JS | Handles MediaDevices API for webcam capture and paints incoming base64 frames. |
| **Processing** | OpenCV & NumPy | Handles base64 string decoding, matrix transformations, and visual annotations. |

## 🧠 Custom Training Details (Optional)
If you wish to train the model yourself on a custom dataset, use the following skeleton script:
```python
from ultralytics import YOLO

# Initialize the architecture
net = YOLO("yolov8n.pt")

# Train on custom YAML data
net.train(data="path/to/data.yaml", epochs=50, imgsz=640, batch=16)
```

## 📜 License
For educational purposes.
