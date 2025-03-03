import os
import shutil
import glob
import yaml
import random
import json
import matplotlib.pyplot as plt
from collections import Counter

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Paths from config
raw_data_dir = config["paths"]["raw_data_dir"]
processed_data_dir = config["paths"]["processed_data_dir"]

dataset_paths = config["paths"]["dataset_paths"]
sanity_check_dir = config["paths"]["sanity_check_dir"]

train_ratio = config["split"]["train_ratio"]
val_ratio = config["split"]["val_ratio"]
test_ratio = config["split"]["test_ratio"]
random_seed = config["split"]["seed"]

# Ensure consistent splits
random.seed(random_seed)

# Create directories for train, val, test splits
for split in ["train", "val", "test"]:
    os.makedirs(dataset_paths["images"].format(split=split), exist_ok=True)
    os.makedirs(dataset_paths["labels"].format(split=split), exist_ok=True)

def split_data():
    """Splits images and labels into train, validation, and test sets based on config ratios."""
    image_paths = glob.glob(os.path.join(raw_data_dir, "images", "*.jpg"))
    
    if not image_paths:
        print("⚠️ No images found in:", raw_data_dir)
        return

    random.shuffle(image_paths)
    total_images = len(image_paths)
    
    train_split = int(total_images * train_ratio)
    val_split = train_split + int(total_images * val_ratio)

    train_images = image_paths[:train_split]
    val_images = image_paths[train_split:val_split]
    test_images = image_paths[val_split:]

    def move_files(images, split_name):
        dest_image_dir = dataset_paths["images"].format(split=split_name)
        dest_label_dir = dataset_paths["labels"].format(split=split_name)
        class_counts = Counter()

        for img_path in images:
            filename = os.path.basename(img_path)
            label_path = os.path.join(raw_data_dir, "labels", filename.replace(".jpg", ".txt"))

            if not os.path.exists(label_path):
                print(f"⚠️ Label missing for {filename}, skipping...")
                continue

            shutil.copy(img_path, os.path.join(dest_image_dir, filename))
            shutil.copy(label_path, os.path.join(dest_label_dir, filename.replace(".jpg", ".txt")))

            # Count occurrences of each class
            with open(label_path, "r") as f:
                for line in f:
                    class_id = line.strip().split()[0]
                    class_counts[class_id] += 1
        return class_counts

    # Move train, val, and test images
    train_counts = move_files(train_images, "train")
    val_counts = move_files(val_images, "val")
    test_counts = move_files(test_images, "test")

    # Save split information
    split_info = {
        "train": len(train_images),
        "val": len(val_images),
        "test": len(test_images),
        "train_class_distribution": dict(train_counts),
        "val_class_distribution": dict(val_counts),
        "test_class_distribution": dict(test_counts),
    }
    with open(os.path.join(processed_data_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=4)

    # Plot class distribution
    plt.figure(figsize=(8, 5))
    for split_name, counts in zip(["Train", "Validation", "Test"], [train_counts, val_counts, test_counts]):
        plt.bar(counts.keys(), counts.values(), alpha=0.5, label=split_name)
    plt.xlabel("Class ID")
    plt.ylabel("Count")
    plt.title("Class Distribution in Train, Val, Test Splits")
    plt.legend()
    plt.savefig(os.path.join(processed_data_dir, "class_distribution.png"))
    plt.close()

    print(f"✅ Train-Val-Test Split Completed!")
    print(f"   🟢 Train: {len(train_images)} images")
    print(f"   🟡 Validation: {len(val_images)} images")
    print(f"   🔵 Test: {len(test_images)} images")
    print(f"   📂 Processed data saved in {processed_data_dir}")

if __name__ == "__main__":
    split_data()
