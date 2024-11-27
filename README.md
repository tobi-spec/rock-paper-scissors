# Rock Paper Sissors

The project implements the game of rock, scissors, paper against the computer and was intended to try out 
various machine learning algorithms. 
For this purpose, an object recognition was developed to make the 3 characters detectable by the computer via the camera. 
The object recognition is based on the Yolov11 model which was retrained with a rock-paper-scissors computer vision dataset 
from Roboflow. 
To make the game more interesting, the computer uses a Q-learning algorithm to make the best choice for the next round. 
The UI was created with the Python library NiceGUI. 

The program can be started by running game_main.py

The dataset for training and the results from train_model.py are excluded from git (see gitignore)
The dataset is from roboflow: https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw
