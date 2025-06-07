import cv2
import os
from pathlib import Path

def letterbox(img, new_size=(1024, 1024), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    scale = min(new_size[0] / shape[0], new_size[1] / shape[1])
    resized = cv2.resize(img, (int(shape[1]*scale), int(shape[0]*scale)))
    result = cv2.copyMakeBorder(
        resized,
        top=(new_size[0] - resized.shape[0]) // 2,
        bottom=(new_size[0] - resized.shape[0] + 1) // 2,
        left=(new_size[1] - resized.shape[1]) // 2,
        right=(new_size[1] - resized.shape[1] + 1) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )
    return result

src_dir = Path("data/images/train")
dst_dir = Path("data/images/calib1024")
dst_dir.mkdir(exist_ok=True)

for img_path in src_dir.glob("*.png"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    padded = letterbox(img)
    cv2.imwrite(str(dst_dir / img_path.name), padded)
