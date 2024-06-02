from ultralytics import settings
from ultralytics import YOLO
import torch
import faulthandler

# settings.update({"runs_dir": "D:\\PycharmProjects\\YOLO\\runs", "weights_dir": "D:\\PycharmProjects\\YOLO\\weights", "datasets_dir": "D:\\PycharmProjects\\YOLO\\datasets"})
# if torch.cuda.is_available():
#     device = "0"
# else:
#     device = "cpu"
#
# if device == "0":
#     torch.cuda.set_device(0)
# print(device)


if __name__ == "__main__":
    faulthandler.enable()  # start @ the beginning
    # model = YOLO("C:\\Users\\Tuand\\PycharmProjects\\YOLO\\runs\\detect\\train8\\weights\\last.pt")
    # results = model.train(data="wider.yaml", epochs=5, batch=3, imgsz=640, save_period=5, device=[0], amp=False, resume=True)



