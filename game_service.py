import random
from ultralytics import YOLO

model = YOLO("./trained_model.pt")


def classify_image():
    result = model.predict(["./player_sign.jpg"])
    box = result[0].boxes
    result[0].save(filename="./player_sign_labeled.jpg")

    if box:
        label = box.cls
        confidence = box.conf
        print(f"Label: {model.names[int(label)]}, Confidence: {confidence}")
        return model.names[int(label)]
    else:
        print("No detections.")


def random_choice():
    return random.choice(["Scissors", "Stone", "Paper"])


player_sign = classify_image()
computer_sign = random_choice()

print(player_sign)
print(computer_sign)

