import random
from ultralytics import YOLO

model = YOLO("./trained_model.pt")


def game():
    player_sign = classify_image()
    computer_sign = random_choice()
    return player_sign, computer_sign, determine_winner(player_sign, computer_sign)


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
    return random.choice(["Scissors", "Rock", "Paper"])


def determine_winner(player, computer):
    rules = {
        "Rock": "Paper",
        "Paper": "Scissors",
        "Scissors": "Rock"
    }
    if player == computer:
        return "Draw"

    if rules[player] == computer:
        return f"computer wins!"
    else:
        return f"player wins!"










