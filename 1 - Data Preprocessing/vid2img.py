import cv2
import os

video_path = os.path.abspath('D:/Ching/Documents/Road-Pothole-Detection/video/DJI_20250602095322_0007_D.MP4')
output_dir = os.path.abspath('D:/Ching/Documents/Road-Pothole-Detection/data/images')
fps_target = 10

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

fps_video = cap.get(cv2.CAP_PROP_FPS)
print("Video FPS:", fps_video)

if fps_video == 0:
    print("Error: Video FPS is 0.")
    exit()

frame_interval = int(round(fps_video / fps_target))
print("Frame Interval:", frame_interval)

frame_count = 0
saved_count = 0

crop_w, crop_h = 1080, 720
center_x = 1920 // 2
bottom_y = 1080

x1 = center_x - crop_w // 2
y1 = bottom_y - crop_h
x2 = x1 + crop_w
y2 = y1 + crop_h

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        cropped = frame[y1:y2, x1:x2]  # Crop from lower middle
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        filename = os.path.join(output_dir, f'{video_basename}_test_{saved_count:04d}.png')
        print(f"Saving frame {frame_count} to {filename}")
        cv2.imwrite(filename, cropped)
        saved_count += 1

    frame_count += 1

cap.release()
