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
from PIL import Image, ImageTk

class HomeScreen:
    def __init__(self, root):
        self.root = root
        self.initialize()

    
    def initialize(self):
        self.root.title("Sign Bridge")
        self.root.geometry("800x600")

        title_label = tk.Label(self.root, text="Welcome to Sign Bridge!", font=("Arial", 18), bd=0)
        title_label.pack(pady=100)

        learn = tk.Button(self.root, text="Learn", font=("Arial", 14), bd=0, width=20,
                                command=self.learn)

        # multiple_choice = tk.Button(self.root, text="Multiple Choice", font=("Arial", 14), bd=0, width=20,
        #                        command=self.multiple_choice)
        
        # webcam_quiz = tk.Button(self.root, text="Webcam Quiz", font=("Arial", 14), bd=0, width=20, command=self.webcam_quiz)
    
        # sign_to_text = tk.Button(self.root, text="Sign to Text", font=("Arial", 14), bd=0, width=20,
        #                            command=self.sign_to_text)
        
        learn.pack()
        # multiple_choice.pack()
        # webcam_quiz.pack()
        # sign_to_text.pack()
    
    def learn(self):
        self.root.withdraw()
        learn_window = tk.Toplevel(self.root)
        learn_window.title("Learn Alphabets")
        learn_window.geometry("800x600")

        Learning(learn_window, self.back_to_home)
    
    
    def back_to_home(self, learn_window):
        learn_window.destroy()
        self.root.deiconify()


class Learning:
    def __init__(self, root, back_func):
        self.root = root
        self.back_func = back_func
        self.initialize()

    def initialize(self):
        self.image_folder = "images/alphabetsLearning"
        self.image_files = [f for f in os.listdir(self.image_folder) 
                            if os.path.isfile(os.path.join(self.image_folder, f))
                             and not f.startswith('.') ]

        self.learn()

        back_button = tk.Button(self.root, text="Back", font=("Arial", 12), bd=0, width=10,
                                command=lambda: self.back_func(self.root))
        back_button.place(x=10, y=10)

        next_button = tk.Button(self.root, text="Next", font=("Arial", 12), bd=0, width=10, command=self.next_image)
        next_button.place(x=400, y=10)

    def learn(self):
        self.clear_widgets()

        random_image_file = random.choice(self.image_files)
        image_path = os.path.join(self.image_folder, random_image_file)

        alphabet = random_image_file.split(".")[0].upper() 
        alphabet_label = tk.Label(self.root, text="This sign indicates the alphabet " + alphabet + " in ASL ", font=("Arial", 14), bd=0)
        alphabet_label.pack(pady=50)

        try:
            self.image = Image.open(image_path)
            self.image.thumbnail((800, 800))  # Adjust the maximum thumbnail size
            self.photo = ImageTk.PhotoImage(self.image)

            image_label = tk.Label(self.root, image=self.photo, bd=0, highlightthickness=0)
            image_label.pack(pady=50)  

        except Exception as ex:
            print(f"Error loading image: {ex}")
        
            
    def next_image(self):
        self.learn()

    def clear_widgets(self):
        for widget in self.root.winfo_children():
            widget.pack_forget()
    
    


root = tk.Tk()
root.geometry("800x600")
home = HomeScreen(root)

root.mainloop()