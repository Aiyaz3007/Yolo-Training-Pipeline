import os
import random
import yaml
import cv2
import shutil
import glob

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Paths from config
raw_data_dir = config["paths"]["raw_data_dir"]
sanity_check_dir = os.path.join(raw_data_dir, "sanity")

image_dir = os.path.join(raw_data_dir, "images")
label_dir = os.path.join(raw_data_dir, "labels")

classes = config["classes"]["names"]
os.makedirs(sanity_check_dir, exist_ok=True)

def parse_yolo_label(label_path):
    """Read YOLO format label file and return bounding boxes"""
    bboxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            data = line.strip().split()
            class_id = int(data[0])
            x, y, w, h = map(float, data[1:])
            bboxes.append((class_id, x, y, w, h))
    return bboxes

def draw_bboxes(image_path, label_path, save_path):
    """Draw bounding boxes on the image"""
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    bboxes = parse_yolo_label(label_path)

    for class_id, x, y, bw, bh in bboxes:
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        color = (0, 255, 0)  # Green
        label = classes[class_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(save_path, image)

def perform_sanity_check(num_samples=5):
    """Select random images from raw data, draw bounding boxes, and save"""
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    
    if not image_paths:
        print("No images found in:", image_dir)
        return

    random.shuffle(image_paths)
    selected_images = image_paths[:num_samples]

    # Clear old sanity check images
    if os.path.exists(sanity_check_dir):
        shutil.rmtree(sanity_check_dir)
    os.makedirs(sanity_check_dir, exist_ok=True)

    for image_path in selected_images:
        filename = os.path.basename(image_path)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

        if not os.path.exists(label_path):
            continue  # Skip if label is missing

        save_path = os.path.join(sanity_check_dir, filename)
        draw_bboxes(image_path, label_path, save_path)

    print(f"✅ Sanity check completed! {len(selected_images)} images saved in {sanity_check_dir}")

if __name__ == "__main__":
    perform_sanity_check(num_samples=5)
