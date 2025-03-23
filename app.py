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
        learn.pack()

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
        
        # Get sorted list of image files (ensuring alphabetical order)
        self.image_files = sorted(
            [f for f in os.listdir(self.image_folder)
             if os.path.isfile(os.path.join(self.image_folder, f)) and not f.startswith('.')]
        )

        self.curr_index = 0  # Start from first letter

        # Label to display alphabet
        self.alphabet_label = tk.Label(self.root, font=("Arial", 14), bd=0)
        self.alphabet_label.pack(pady=50)

        # Image Label
        self.image_label = tk.Label(self.root, bd=0, highlightthickness=0)
        self.image_label.pack(pady=50)

        # Back button
        self.back_button = tk.Button(self.root, text="Back", font=("Arial", 12), bd=0, width=10,
                                     command=lambda: self.back_func(self.root))
        self.back_button.place(x=10, y=10)

        # Previous button
        self.prev_button = tk.Button(self.root, text="Previous", font=("Arial", 12), bd=0, width=10,
                                     command=self.previous_image)
        self.prev_button.place(x=300, y=10)

        # Next button
        self.next_button = tk.Button(self.root, text="Next", font=("Arial", 12), bd=0, width=10,
                                     command=self.next_image)
        self.next_button.place(x=450, y=10)

        self.learn()

    def learn(self):
        """Display the current alphabet image"""
        # Get current image
        image_file = self.image_files[self.curr_index]
        image_path = os.path.join(self.image_folder, image_file)

        # Extract alphabet from filename
        alphabet = image_file.split(".")[0].upper()
        self.alphabet_label.config(text=f"This sign indicates the letter {alphabet} in ASL")

        try:
            self.image = Image.open(image_path)
            self.image.thumbnail((400, 400))  # Maintain aspect ratio
            self.photo = ImageTk.PhotoImage(self.image)

            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo  # Keep reference

        except Exception as ex:
            print(f"Error loading image: {ex}")

        self.update_buttons()  # Ensure buttons are updated

    def next_image(self):
        """Move to next letter if possible"""
        if self.curr_index < len(self.image_files) - 1:
            self.curr_index += 1
            self.learn()

    def previous_image(self):
        """Move to previous letter if possible"""
        if self.curr_index > 0:
            self.curr_index -= 1
            self.learn()

    def update_buttons(self):
        """Enable/Disable navigation buttons based on position"""
        self.prev_button.config(state=tk.NORMAL if self.curr_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.curr_index < len(self.image_files) - 1 else tk.DISABLED)


root = tk.Tk()
root.geometry("800x600")
home = HomeScreen(root)

root.mainloop()