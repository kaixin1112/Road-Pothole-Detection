import os

def delete_images_without_labels(labels_dir, images_dir):
    label_basenames = set(os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith('.txt'))
    for image_file in os.listdir(images_dir):
        if image_file.endswith('.png'):
            image_basename = os.path.splitext(image_file)[0]
            if image_basename not in label_basenames:
                image_path = os.path.join(images_dir, image_file)
                os.remove(image_path)
                print(f"Deleted image: {image_path}")

def delete_labels_without_images(labels_dir, images_dir):
    image_basenames = set(os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith('.png'))
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            label_basename = os.path.splitext(label_file)[0]
            if label_basename not in image_basenames:
                label_path = os.path.join(labels_dir, label_file)
                os.remove(label_path)
                print(f"Deleted label: {label_path}")

# Example usage:
labels_dir = r'D:/Ching/Downloads/job_2531124_annotations_2025_06_03_12_39_08_yolo 1.1/obj_train_data'
images_dir = r'D:/Ching/Documents/Road-Pothole-Detection/data/images/detect'

# delete_images_without_labels(labels_dir, images_dir)
delete_images_without_labels(labels_dir, images_dir)