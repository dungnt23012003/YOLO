from ultralytics import settings
from ultralytics import YOLO
import torch
import faulthandler


if __name__ == "__main__":
    faulthandler.enable()  # start @ the beginning
    model = YOLO("/runs/detect/train_10_epochs_wider_face\\weights\\last.pt")
    results = model.train(data="wider.yaml", epochs=10, batch=3, imgsz=640, device=[0], amp=False, resume=True)