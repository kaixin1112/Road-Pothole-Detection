import cv2
import degirum as dg
import degirum_tools
from pprint import pprint
import time

# Configuration
model_name = "yolov8n"
display_name = "AI Dashcam"
inference_host_address = "@local"
zoo_url = "zoo_url"
token = ''  # Leave empty for local inference

# Load the model
model = dg.load_model(
    model_name=model_name,
    inference_host_address=inference_host_address,
    zoo_url=zoo_url,
    token=token
)

# Define color map (BGR format for OpenCV)
label_colors = {
    "Pothole": (0, 255, 255),  # Yellow
    "Manhole": (255, 0, 0),   # Blue
    "unknown": (255, 255, 255) # White (default)
}

recording = False
video_writer = None
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Open webcam and set lower resolution
cap = cv2.VideoCapture('/dev/video0')
if not cap.isOpened():
    raise RuntimeError("Unable to access the webcam.")

with degirum_tools.Display(display_name) as output_display:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (720, 480))
        frame = cv2.flip(frame, -1)

        # Run inference
        inference_result = model(frame)

        for det in inference_result.results:
            label = det.get("label", "unknown")
            score = det.get("confidence", det.get("score", 0.0))
            bbox = det.get("bbox", [0, 0, 0, 0])
            
            if label == "Pothole" and score >= 0.75:
                x1, y1, x2, y2 = map(int, bbox)
                color = label_colors.get(label, label_colors["unknown"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label}: {score:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

        # Show output
        output_display.show(frame)

        # Record if enabled
        if recording and video_writer:
            video_writer.write(frame)

        # Keyboard listener
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recording = not recording
            if recording:
                print("Recording started...")
                if video_writer is None:
                    height, width = frame.shape[:2]
                    video_writer = cv2.VideoWriter("output4.mp4", fourcc, 30.0, (width, height))
            else:
                print("Recording stopped.")

cap.release()
if video_writer:
    video_writer.release()
print(f"Video stream '{display_name}' has been processed.")
