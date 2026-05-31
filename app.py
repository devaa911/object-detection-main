import cv2
import torch
import pyttsx3
import threading
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# Initialize YOLO model
model = YOLO("yolov5s.pt")  # Replace with your trained model if needed

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Set voice to female
voices = engine.getProperty("voices")
for voice in voices:
    if "female" in voice.name.lower():
        engine.setProperty("voice", voice.id)
        break

# Function to perform text-to-speech in a separate thread
def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Perform object detection
        results = model(frame)
        detected_objects = set()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{model.names[cls]} {conf:.2f}"

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Speak detected object once per frame
                object_name = model.names[cls]
                if object_name not in detected_objects:
                    detected_objects.add(object_name)
                    speak(object_name)

        # Encode the frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
