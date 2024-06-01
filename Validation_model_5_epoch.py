from ultralytics import YOLO
import faulthandler


if __name__ == "__main__":
    faulthandler.enable()  # start @ the beginning
    model = YOLO("C:\\Users\\Tuand\\PycharmProjects\\YOLO\\runs\\detect\\train8\\weights\\last.pt")

    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps