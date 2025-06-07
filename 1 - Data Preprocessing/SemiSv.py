import cv2
import os
from ultralytics import YOLO

# === CONFIGURATION ===
video_path = os.path.abspath('D:/Ching/Documents/Road-Pothole-Detection/video/DJI_20250531143156_0004_D.MP4')
output_dir = os.path.abspath('D:/Ching/Documents/Road-Pothole-Detection/data/images/detect')
model_path = os.path.abspath('best.pt')  # Your .pt file
confidence_threshold = 0.65

# === LOAD MODEL ===
model = YOLO(model_path)
model.fuse()

# === VIDEO SETUP ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

fps_video = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(round(fps_video / 30))  # 10 FPS target
frame_count = 0
saved_count = 0

# === CROPPING PARAMETERS ===
crop_w, crop_h = 1080, 720
center_x = 1920 // 2
bottom_y = 1080
x1 = center_x - crop_w // 2
y1 = bottom_y - crop_h
x2 = x1 + crop_w
y2 = y1 + crop_h

os.makedirs(output_dir, exist_ok=True)

# === PROCESS VIDEO ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        results = model(frame)
        boxes = results[0].boxes

        if boxes is not None:
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                if cls == 0 and conf >= confidence_threshold:
                    cropped = frame[y1:y2, x1:x2]
                    video_basename = os.path.splitext(os.path.basename(video_path))[0]
                    filename = os.path.join(output_dir, f'{video_basename}_A_{saved_count:04d}.png')
                    print(f"Saving detection with conf {conf:.2f} at frame {frame_count} to {filename}")
                    cv2.imwrite(filename, cropped)
                    saved_count += 1
                    break  # Only one save per frame

    frame_count += 1

cap.release()
