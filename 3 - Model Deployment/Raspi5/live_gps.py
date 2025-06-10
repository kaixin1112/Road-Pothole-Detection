import cv2
import degirum as dg
import degirum_tools
import time
import os
import math
import threading
from location import GPSLocator


global last_saved_lat, last_saved_lon, image_counter, last_saved_detections_count

# ========== Global Variables ==========
current_lat = None
current_lon = None
last_saved_lat = None
last_saved_lon = None
image_counter = 1
last_saved_detections_count = 0
lock = threading.Lock()
recording = False
video_writer = None
video_counter = 1
video_size = (1280, 720)

# Initialize GPS
gps = GPSLocator()


def get_next_video_filename():
    i = 1
    while os.path.exists(f"outputA{i}.mp4"):
        i += 1
    return f"outputA{i}.mp4", i

# ========== Utility ==========
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

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

        frame = cv2.resize(frame, video_size)
        frame = cv2.flip(frame, -1)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        inference_result = model(frame)

        timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")

        # Show FPS and Time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {timestamp}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show current GPS coordinates
        location = gps.get_location()
        if location:
            lat = location["latitude"]
            lon = location["longitude"]
        else:
            lat = None
            lon = None
        
        if lat and lon:
            cv2.putText(frame, f"Lat: {lat:.5f}, Lon: {lon:.5f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        detections_above_08 = []
        frame_width, frame_height = frame.shape[1], frame.shape[0]

        for det in inference_result.results:
            label = det.get("label", "unknown")
            score = det.get("confidence", det.get("score", 0.0))
            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = map(int, bbox)
            
            if score >= 0.7 and label.lower() == "pothole":
                color = label_colors.get(label, label_colors["unknown"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if score >= 0.8:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    in_lower_middle = (
                        frame_width * 1/3 < center_x < frame_width * 2/3 and
                        center_y > frame_height / 2
                    )
                    if in_lower_middle:
                        detections_above_08.append((label, score))


        if detections_above_08 and lat and lon:
            with lock:
                save = False
                if last_saved_lat is None or last_saved_lon is None:
                    save = True
                else:
                    distance = haversine(lat, lon, last_saved_lat, last_saved_lon)
                    if distance >= 0.001:  # Considered different location (≈1.11m)
                        save = True
                    elif len(detections_above_08) > last_saved_detections_count:
                        save = True

                if save:
                    os.makedirs("detections", exist_ok=True)
                    while os.path.exists(f"detections/image{image_counter}.png") or os.path.exists(f"detections/image{image_counter}.txt"):
                        image_counter += 1

                    img_name = f"detections/image{image_counter}.png"
                    txt_name = f"detections/image{image_counter}.txt"
                    cv2.imwrite(img_name, frame)

                    with open(txt_name, "w") as f:
                        for label, score in detections_above_08:
                            f.write(f"{label} {round(score, 2)} {lat:.5f} {lon:.5f} {timestamp}\n")

                    print(f"✅ Detection saved: {img_name}")
                    last_saved_lat = lat
                    last_saved_lon = lon
                    last_saved_detections_count = len(detections_above_08)
                    image_counter += 1


        cv2.imshow("AI Dashcam", frame)
        
        if recording and video_writer:
            video_writer.write(frame)
        
        # Check for 'r' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if not recording:
                filename, video_counter = get_next_video_filename()
                print(f"📹 Start recording: {filename}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(filename, fourcc, 30, video_size)
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
cv2.destroyAllWindows()
gps.close()
print("📹 Stream ended.")
