import os
from cv2 import cv2
import torch

model = torch.hub.load("ultralytics/yolov5", "yolov5s")

cap = cv2.VideoCapture("traffic.mp4")

while True:
    _, frame = cap.read()
    
    # Inference
    results = model(frame)
    labels = results.xyxyn[0][:, -1].numpy()
    cord_thresholds = results.xyxyn[0][:, :-1].numpy()
    print(labels)
    print(cord_thresholds)
    results.show()
    break

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key != -1: 
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)