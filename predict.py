from ultralytics import YOLO
import numpy as np

model = YOLO('./runs/classify/train2/weights/last.pt')
results = model('./weather_dataset/val/sunrise/sunrise310.jpg')
print(type(results))
print(results)
names_dict = results[0].names
probs = results[0].probs.data.tolist()
print(names_dict)
print(probs)
print(names_dict[np.argmax(probs)])
