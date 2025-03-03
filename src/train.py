import os
import yaml
import mlflow
import urllib.request
from ultralytics import YOLO
from dvclive import Live
from ultralytics import settings
import torch

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

with open("configs/default.yaml", "r") as f:
    ARGS = yaml.safe_load(f)

# Extract configurations
yolo_version = config["training"]["yolo_version"]
epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
img_size = config["training"]["img_size"]
learning_rate = config["training"]["learning_rate"]
weight_decay = config["training"]["weight_decay"]
optimizer = config["training"]["optimizer"]
model_weights_path = config["training"]["pretrained_model_save_path"].format(yolo_version=yolo_version)
training_config_path = config["training"]["training_config_path"]
device = config["training"].get("device", "cpu")  # Default to "cuda" if not specified

use_dvc_live = config["logging"]["use_dvc_live"]
mlflow_tracking_uri = config["logging"]["mlflow_tracking_uri"]
experiment_name = config["logging"]["experiment_name"]

# Paths
model_url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{yolo_version}.pt"

project_name = config["project"]["name"]
version_to_add = "v{count}"

checkpoint_dir = config["training"]["checkpoint_dir"].format(project_name=project_name)
os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure checkpoint directory exists

# Versioning
version_counts = len([x for x in os.listdir(checkpoint_dir) if x.startswith("v")])+1
model_save_path = config["training"]["model_save_path"].format(
    project_name=project_name, version=version_to_add.format(count=version_counts)
)
sub_checkpoint_dir = os.path.join(checkpoint_dir, version_to_add.format(count=version_counts))
os.makedirs(sub_checkpoint_dir, exist_ok=True)  # Ensure the sub-version directory exists

# Download weights if not available
if not os.path.exists(model_weights_path):
    print(f"🔽 Downloading {yolo_version} weights...")
    urllib.request.urlretrieve(model_url, model_weights_path)
    print(f"✅ Downloaded: {model_weights_path}")

# Load YOLO model
model = YOLO(model_weights_path)
# Run MODE mode using the custom arguments ARGS (guess TASK)
model.named_parameters

# Training parameters
train_config = {
    "imgsz": img_size,
    "epochs": epochs,
    "batch": batch_size,
    "lr0": learning_rate,
    "weight_decay": weight_decay,
    "optimizer": optimizer.lower(),
    "project": checkpoint_dir,
    # "name": version_to_add.format(count=version_counts),
    # "name": "v1",
    "cache": True,
    "device": device,
    "cfg": "configs/default.yaml"
}

# Disable default MLflow logging by YOLO
settings.update({"mlflow": False})

# Initialize DVC Live if enabled
if use_dvc_live:
    live = Live(config["logging"]["dvc_live_dir"], dvcyaml=False)

# Initialize MLflow
mlflow.set_tracking_uri(mlflow_tracking_uri)
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    mlflow.set_experiment(mlflow.create_experiment(experiment_name))
else:
    mlflow.set_experiment(experiment_name)

# Start MLflow run with a custom name
with mlflow.start_run(run_name=sub_checkpoint_dir):
    mlflow.log_params(train_config)

    # Train model with epoch-based logging
    results = model.train(data=training_config_path, **train_config)

    # Save model weights (not full model instance)
    torch.save(model.state_dict(), model_save_path)

    # Log model to MLflow
    mlflow.log_artifact(model_save_path)

    with open("log.txt", "w") as f:
        f.write(str(results))

    # Log best results with DVC Live
    if use_dvc_live:
        live.log_metric("best_mAP", results.best_map)
        live.log_metric("best_loss", results.best_loss)
        live.next_step()

        # Extract relevant metrics
        mlflow.log_metric("mAP50", results.box.map50)  
        mlflow.log_metric("mAP50-95", results.box.map)  
        mlflow.log_metric("fitness", results.box.fitness)  
        mlflow.log_metric("precision", results.box.precision)  
        mlflow.log_metric("recall", results.box.recall)  

    print(f"✅ Training completed! Model saved at {model_save_path}")
