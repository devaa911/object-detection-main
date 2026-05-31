# 👁️ Real-Time Audio-Visual Object Analytics Engine

An advanced, edge-optimized **Computer Vision Pipeline** that integrates object detection with multi-threaded audio synthesis. The framework handles real-time video streams through a **Flask Web Gateway**, passes image grids into an optimized **YOLO architecture**, and provides immediate audio feedback using a decoupled, non-blocking **Text-to-Speech (TTS)** engine thread.

---

## 🌌 System Architecture Workflow

```text
 Webcam Stream ──► OpenCV Frame Capture ──► YOLO Detection Core ──► Vector Coordinate Mapping
                                                                     │
 ┌───────────────────────────────────────────────────────────────────┘
 │
 ├──► Render Interface ──► JPEG Buffer Conversion ──► Flask Multi-part Web Stream UI
 └──► Async Audio Engine ──► Deduplication Check ──► Thread Spawning ──► pyttsx3 Female TTS

realtime-vision-portal/
│
├── app.py                         # Your main Flask & Vision application code
├── README.md                      # Repository configuration documentation
│
└── templates/
    └── index.html                 # Main interface template containing the image feed

# Core Vision & deep learning dependencies
pip install opencv-python torch torchvision torchaudio ultralytics flask

# Cross-platform Text-To-Speech library
pip install pyttsx3


sudo apt-get update
sudo apt-get install espeak festival xsel libasound2-dev


python app.py


🚀 Initializing YOLO Model weights...
Downloading [https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5su.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5su.pt) to 'yolov5su.pt'...
 * Running on [http://127.0.0.1:5000](http://127.0.0.1:5000)
 * Debug mode: on
