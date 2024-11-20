from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())

    model = YOLO("yolo11n.pt")
    results = model.train(data="./rock-paper-scissors.v14i.yolov11/data.yaml", batch=3, epochs=20, imgsz=640, device=[0])

    validate = model("./test_img.jpg")
    print(validate)
