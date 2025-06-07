import os

# Set your paths here
txt_folder = r"D:/Ching/Documents/Road-Pothole-Detection/data/labels/train"
png_folder = r"D:/Ching/Documents/Road-Pothole-Detection/data/images/calib1024"

# Loop through all .txt files
for filename in os.listdir(txt_folder):
    if filename.endswith(".txt"):
        txt_path = os.path.join(txt_folder, filename)

        # Check if the file is empty
        if os.path.getsize(txt_path) == 0:
            # Build corresponding .png path
            base_name = os.path.splitext(filename)[0]
            png_path = os.path.join(png_folder, base_name + ".png")

            # Delete the .png file if it exists
            if os.path.exists(png_path):
                os.remove(png_path)
                print(f"Deleted: {png_path}")
            else:
                print(f"PNG not found: {png_path}")
