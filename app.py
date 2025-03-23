import tkinter as tk
import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import time
import random

class HomeScreen:
    def __init__(self, root):
        self.root = root
        self.initialize()

    
    def initialize(self):
        self.root.title("Sign Bridge")
        self.root.geometry("800x600")

        title_label = tk.Label(self.root, text="Welcome to Sign Bridge!", font=("Arial", 18), bd=0)
        title_label.pack(pady=100)

        # learn = tk.Button(self.root, text="Learn", font=("Arial", 14), bd=0, width=20,
        #                         command=self.learn)

        # multiple_choice = tk.Button(self.root, text="Multiple Choice", font=("Arial", 14), bd=0, width=20,
        #                        command=self.multiple_choice)
        
        # webcam_quiz = tk.Button(self.root, text="Webcam Quiz", font=("Arial", 14), bd=0, width=20, command=self.webcam_quiz)
    
        # sign_to_text = tk.Button(self.root, text="Sign to Text", font=("Arial", 14), bd=0, width=20,
        #                            command=self.sign_to_text)
        
        # learn.pack()
        # multiple_choice.pack()
        # webcam_quiz.pack()
        # sign_to_text.pack()


root = tk.Tk()
root.geometry("800x600")
home = HomeScreen(root)

root.mainloop()