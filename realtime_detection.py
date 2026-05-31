import torch
import cv2
import pyttsx3
from ultralytics import YOLO

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Set voice to female
voices = engine.getProperty("voices")
for voice in voices:
    if "female" in voice.name.lower():  # Look for a female voice
        engine.setProperty("voice", voice.id)
        break

# Load YOLOv5 model
model = YOLO("yolov5s.pt")  # Use your trained model if needed

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)  # Detect objects

    detected_objects = set()  # Avoid repeating the same object in one frame

    # Process results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class index
            label = f"{model.names[cls]} {conf:.2f}"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Speak detected object once per frame
            object_name = model.names[cls]
            if object_name not in detected_objects:
                detected_objects.add(object_name)
                engine.say(object_name)
                engine.runAndWait()

    # Show frame
    cv2.imshow("YOLOv5 Real-Time Detection with Voice", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
