import os
import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Initialize YOLOv8 model from scratch (no pretrained weights)
model = YOLO('yolov8m.pt')  # Use 'yolov8n.yaml' for a small model. You can also try 'yolov8s.yaml' for a slightly larger model.

# Train the model from scratch
results = model.train(data = 'config.yaml', epochs = 50)
