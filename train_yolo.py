from ultralytics import YOLO
import torch

# Load pre-trained YOLOv8 nano model (transfer learning as in paper)
model = YOLO('yolov8n.pt')  # Or 'yolov8s.pt' for small variant

# Train on your dataset
results = model.train(
    data='path/to/data.yaml',  # Update to your YAML path
    epochs=100,                # As per paper (50-100 for convergence)
    imgsz=256,                 # Matches SDNET resolution
    batch=16,                  # Adjust based on GPU (paper uses 32 on A100)
    device=0 if torch.cuda.is_available() else 'cpu',  # GPU if available
    workers=8,                 # DataLoader threads
    augment=True,              # Enable augmentations (flip, rotate, as in paper)
    lr0=0.01,                  # Initial learning rate (tuned in paper)
    optimizer='AdamW',         # As recommended
    patience=20,               # Early stopping
    save=True,                 # Save checkpoints
    project='runs/detect',     # Save dir
    name='crack_train'         # Run name
)

# Validate on test set
metrics = model.val(data='path/to/data.yaml', split='test')

# Export model (optional, for ONNX/TensorRT)
model.export(format='onnx')  # For faster inference
print("Training complete! Best model: runs/detect/crack_train/weights/best.pt")
print(f"mAP@0.5: {metrics.box.map50:.3f}")  # Print as in paper