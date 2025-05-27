import cv2
import degirum as dg
import degirum_tools
from pprint import pprint

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
    "Pothole": (0, 0, 255),   # Red
    "Manhole": (255, 0, 0),   # Blue
    "unknown": (255, 255, 255) # White (default)
}

# Open webcam and preprocess frames
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to access the webcam.")

with degirum_tools.Display(display_name) as output_display:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # === Flip the webcam feed ===
        frame = cv2.flip(frame, -1)  # Flip both horizontally and vertically

        # Run inference
        inference_result = model(frame)

        for det in inference_result.results:
            label = det.get("label", "unknown")
            score = det.get("confidence", det.get("score", 0.0))
            bbox = det.get("bbox", [0, 0, 0, 0])
            
            if score >= 0.7:
                pprint(det)
                
                # Convert to integers
                x1, y1, x2, y2 = map(int, bbox)

                # Get color for label, default to white
                color = label_colors.get(label, label_colors["unknown"])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Display label and score
                text = f"{label}: {score:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

        # Show output
        output_display.show(frame)

cap.release()
print(f"Video stream '{display_name}' has been processed.")