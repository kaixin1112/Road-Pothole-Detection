import cv2
import degirum as dg
import degirum_tools
from gps3 import gps3
import time
import os
import math
import threading

# ========== Shared Variables ==========
recording = False
video_writer = None
video_counter = 1

# ========== Save Video Filename Function ==========

def get_next_video_filename():
    i = 1
    while os.path.exists(f"outputA{i}.mp4"):
        i += 1
    return f"outputA{i}.mp4", i

# ========== Load Model ==========

model = dg.load_model(
    model_name="yolov8n",
    inference_host_address="@local",
    zoo_url="zoo_url",
    token=""
)

label_colors = {
    "Pothole": (0, 255, 255),
    "unknown": (255, 255, 255)
}

# ========== Camera Setup ==========

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("Unable to access the webcam.")

# ========== FPS Setup ==========

prev_time = time.time()
fps = 0

# ========== Main Loop ==========

with degirum_tools.Display("AI Dashcam") as output_display:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, -1)
        
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        inference_result = model(frame)

            
        # Show FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show Time
        timestamp = time.strftime("%Y-%m-%d | %H:%M:%S")
        cv2.putText(frame, f"Time: {timestamp}", (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        for det in inference_result.results:
            label = det.get("label", "unknown")
            score = det.get("confidence", det.get("score", 0.0))
            bbox = det.get("bbox", [0, 0, 0, 0])

            if score >= 0.7 and label.lower() == "pothole":
                x1, y1, x2, y2 = map(int, bbox)
                color = label_colors.get(label, label_colors["unknown"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write frame if recording
        if recording and video_writer is not None:
            video_writer.write(frame)

        cv2.imshow("AI Dashcam", frame)

        # Check for 'r' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if not recording:
                filename, video_counter = get_next_video_filename()
                print(f"📹 Start recording: {filename}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(filename, fourcc, 30, (1280, 720))
                recording = True
            else:
                print("🛑 Stop recording.")
                recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
        elif key == ord('q'):
            break

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
print("📹 Stream ended.")
