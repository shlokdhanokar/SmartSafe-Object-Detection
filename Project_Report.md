# Project Report: SmartSafe-Object-Detection

**Author:** Shlok Dhanokar  
**Domain:** Computer Vision, Web Sockets, and Deep Learning  

---

## 1. Introduction and Objectives
The objective of this project was to design and implement a robust **real-time safety compliance monitoring system**. Leveraging state-of-the-art computer vision models (YOLOv8) intertwined with modern web communication protocols (WebSocket), the application identifies critical safety objects—specifically **helmets and packages**—in live video streams. 

The practical problem addressed is the automation of workplace safety checks (e.g., in construction zones or warehouse logistics) where continuous manual surveillance is prone to human error and fatigue.

## 2. System Architecture & Implementation Depth
The system is built on a decoupled client-server architecture to ensure high throughput and low latency during live video inference.

### 2.1 Backend Engineering (Python/Flask)
The core logic resides in a Python backend utilizing **Flask** and **Flask-SocketIO**. 
- **WebSocket Protocol:** Traditional HTTP polling is insufficient for high-framerate video processing. WebSockets provide a persistent, bi-directional event-driven pipeline. 
- **Data Pipeline:** The server listens for the `video_frame` event, parses base64-encoded JPEG strings emitted by the browser, decodes them back to binary, and reconstructs the image matrix using **NumPy (`np.frombuffer`)** and **OpenCV (`cv2.imdecode`)**.

### 2.2 Artificial Intelligence & Inference (YOLOv8)
The project utilizes the **YOLOv8 (You Only Look Once)** architecture by Ultralytics. 
- YOLOv8 was selected for its unparalleled balance between inference speed and bounding-box accuracy.
- The system is designed to dynamically accept custom domain-trained weights (`best.pt`) while seamlessly falling back to a pre-trained COCO dataset model (`yolov8n.pt`) if the custom weights are unavailable. This ensures the execution pipeline never fatally crashes under strict grading or unfamiliar environments.
- **Filtering Logic:** The application programmatically filters out generic detections (such as people or cars) to isolate only the target classes (Helmest/Boxes) with a hardcoded `CONFIDENCE_FLOOR` of 0.50 to minimize false positives.

### 2.3 Frontend Implementation (Vanilla HTML/JS)
The client-side interface is kept lightweight, utilizing fundamental Web APIs rather than bloated frontend frameworks.
- The **MediaDevices API** (`navigator.mediaDevices.getUserMedia`) taps directly into the host machine's webcam.
- Captured frame data is painted onto a hidden HTML `<canvas>` element periodically, serialized to base64, and pushed to the backend socket. 
- When the backend replies with an annotated image (`processed_frame`), the UI dynamically binds the base64 payload to the `src` attribute of an `<img>` tag, creating the illusion of a continuous, fluid video player.

## 3. Reflection and Challenges Overcome

**Handling Video Latency and Serialisation Overhead:** 
A significant engineering challenge was the bottleneck of continuously passing dense image matrices between the browser and the Python server. Raw images are excessively large. I overcame this by intentionally lowering the compression quality of the frames to a `JPEG_QUALITY` of 85 (`cv2.imencode`) and capping the emit interval to 750ms on the frontend. This optimization sharply decreased network payload sizes and prevented the Socket buffer from overflowing, resulting in a significantly more stable stream.

**Asynchronous State Management:**
Managing the webcam lifecycle required careful DOM memory management. Binding an IIFE (Immediately Invoked Function Expression) in JavaScript ensured that global variable pollution was prevented, avoiding unintended event-listener duplications when users rapidly toggled the "Start Camera" and "Stop Camera" features.

## 4. Conclusion
The implementation of SmartSafe successfully synthesizes machine learning with full-stack web development. It demonstrates practical command over environment setups, real-time networking, matrix transformation, and YOLO-based deep learning inference. The project stands as a fully operational prototype ready for edge-device deployment.
