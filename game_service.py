import random
from ultralytics import YOLO
import numpy as np

model = YOLO("./trained_model.pt")

ACTIONS = ["Scissors", "Rock", "Paper"]
ACTION_TO_INDEX = {action: i for i, action in enumerate(ACTIONS)}
Q_TABLE = np.zeros((len(ACTIONS), len(ACTIONS)))
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.5  # Exploration rat

REWARDS = {
    ("Rock", "Scissors"): -5,  # Computer loses
    ("Paper", "Rock"): -5,
    ("Scissors", "Paper"): -5,
    ("Rock", "Rock"): -3,       # Draw
    ("Paper", "Paper"): -3,
    ("Scissors", "Scissors"): -3,
    ("Scissors", "Rock"): 3,   # Computer wins
    ("Rock", "Paper"): 3,
    ("Paper", "Scissors"): 3,
}


def game():
    player_sign = classify_image()

    if player_sign not in ACTIONS:
        player_sign = random.choice(ACTIONS)
    state_index = ACTION_TO_INDEX[player_sign]
    computer_sign = choose_action(state_index)

    result_list = list()
    result_list.append(f"Player has {player_sign}")
    result_list.append(f"Computer has {computer_sign}")
    result_list.append(determine_winner(player_sign, computer_sign))
    return result_list


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


def choose_action(state_index):
    """Choose an action using epsilon-greedy policy."""
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS)  # Explore
    else:
        return ACTIONS[np.argmax(Q_TABLE[state_index])]


def update_q_table(state_index, action_index, reward, next_state_index):
    """Update the Q-value for a given state-action pair."""
    best_next_action = np.max(Q_TABLE[next_state_index])  # Best future action
    Q_TABLE[state_index, action_index] = (
        Q_TABLE[state_index, action_index]
        + ALPHA * (reward + GAMMA * best_next_action - Q_TABLE[state_index, action_index])
    )


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


class GameCounter:
    def __init__(self):
        self.player = 0
        self.computer = 0

    def count(self, result):
        print(result)
        if "player wins!" in result:
            self.player += 1
        if "computer wins!" in result:
            self.computer += 1








