from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='/mnt/d/Science/computer_vision/image_classification_YOLOV8/waether_dataset', epochs=20, imgsz=64)