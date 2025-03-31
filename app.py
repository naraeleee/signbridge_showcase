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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Load the model
data_dict = pickle.load(open('./data/data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)


f = open('./data/model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
import pickle


model_dict = pickle.load(open('./data/model.p', 'rb'))
model = model_dict['model']



class HomeScreen:
    def __init__(self, root):
        self.root = root
        self.initialize()

    def initialize(self):
        self.root.title("Welcome to Sign Bridge")
        self.root.attributes('-fullscreen', True)
        self.root.bind("<Escape>", self.exit_fullscreen)

        self.root.update_idletasks()

        # Background image
        try:
            self.background_image = Image.open("background.jpeg")
            self.background_photo = ImageTk.PhotoImage(self.background_image)
        except FileNotFoundError:
            print("Error: background.jpeg not found.")
            return  

        self.canvas = tk.Canvas(self.root, width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.canvas.create_image(0, 0, image=self.background_photo, anchor="nw")
        self.canvas.image = self.background_photo
        self.canvas.pack(fill="both", expand=True)

        # Title Label
        title_label = tk.Label(self.root, text="Welcome to Sign Bridge!", font=("Arial", 20), bd=0, bg="white", fg="black", highlightthickness=0)
        title_label.place(relx=0.5, rely=0.35, anchor="center")

        # Logo Image
        try:
            self.logo_image = Image.open("images/logo.png")
            self.logo_image = self.logo_image.resize((200, 200), Image.LANCZOS) 
            self.logo_photo = ImageTk.PhotoImage(self.logo_image)

            logo_label = tk.Label(self.root, image=self.logo_photo, bg="#60A4AC") 
            logo_label.place(relx=0.5, rely=0.2, anchor="center")

        except FileNotFoundError:
            print("Error: logo.png not found.")

        # Home Menu
        learn = tk.Button(self.root, text="Learn Alphabets", font=("Arial", 25), width=20, fg="black", bg="white", highlightthickness=0, highlightbackground="#ffffff", command=self.learn)
        multiple_choice = tk.Button(self.root, text="Multiple Choice", font=("Arial", 25), width=20, fg="black", bg="white", highlightthickness=0, highlightbackground="#ffffff", command=self.multiple_choice)
        webcam_quiz = tk.Button(self.root, text="Webcam Quiz", font=("Arial", 25), width=20, fg="black", bg="white", highlightthickness=0, highlightbackground="#ffffff", command=self.webcam_quiz)

        learn.place(relx=0.5, rely=0.45, anchor="center")
        multiple_choice.place(relx=0.5, rely=0.55, anchor="center")
        webcam_quiz.place(relx=0.5, rely=0.65, anchor="center")


    def learn(self):
        self.root.withdraw()
        learn_window = tk.Toplevel(self.root)
        learn_window.title("Learn Alphabets")
        learn_window.attributes('-fullscreen', True) 
        learn_window.bind("<Escape>", lambda event: learn_window.attributes('-fullscreen', False))

        Learning(learn_window, self.back_to_home)
    
    def multiple_choice(self):
        self.root.withdraw()  
        quiz_window = tk.Toplevel(self.root)
        quiz_window.title("Alphabet Quiz")
        quiz_window.attributes('-fullscreen', True)
        quiz_window.bind("<Escape>", lambda event: quiz_window.attributes('-fullscreen', False))

        MultipleChoice(quiz_window, self.back_to_home)

    def webcam_quiz(self):
        self.root.withdraw()
        quiz_window = tk.Toplevel(self.root)
        quiz_window.attributes('-fullscreen', True)
        self.root.configure(bg="white")
        quiz_window.bind("<Escape>", lambda event: quiz_window.attributes('-fullscreen', False))
        WebcamQuiz(quiz_window, self.back_to_home)



    def back_to_home(self, learn_window):
        learn_window.destroy()
        self.root.deiconify()

    
    def exit_fullscreen(self, event=None):
        """Exit full-screen mode when Escape is pressed"""
        self.root.attributes('-fullscreen', False)


class Learning:
    def __init__(self, root, back_func):
        self.root = root
        self.back_func = back_func
        self.initialize()

    def initialize(self):
        self.root.title("Learning - Sign Bridge")
        self.root.attributes('-fullscreen', True)
        self.root.bind("<Escape>", self.exit_fullscreen)

        try:
            self.background_image = Image.open("background.jpeg")  
            self.background_photo = ImageTk.PhotoImage(self.background_image)
        except FileNotFoundError:
            print("Error: background.jpeg not found.")
            return  

        self.canvas = tk.Canvas(self.root, width=self.background_photo.width(), height=self.background_photo.height())
        self.canvas.create_image(0, 0, image=self.background_photo, anchor="nw")
        self.canvas.image = self.background_photo 
        self.canvas.pack(fill="both", expand=True)

        self.image_folder = "images/alphabetsLearning"
        
        self.image_files = sorted(
            [f for f in os.listdir(self.image_folder)
             if os.path.isfile(os.path.join(self.image_folder, f)) and not f.startswith('.')]
        )

        self.curr_index = 0 

        self.alphabet_label = tk.Label(self.root, font=("Arial", 14), bd=0, bg="white", fg="black", highlightthickness=0)
        self.alphabet_label.place(relx=0.5, rely=0.1, anchor="center")

        # Image Label
        self.image_label = tk.Label(self.root, bd=0, highlightthickness=0, bg="white")
        self.image_label.place(relx=0.5, rely=0.4, anchor="center")

        # Back button
        self.back_button = tk.Button(self.root, text="Back", font=("Arial", 20), bd=0, width=10, fg="black", bg="white", highlightthickness=0, highlightbackground="#ffffff",
                                     command=lambda: self.back_func(self.root))
        self.back_button.place(x=10, y=10)

        # Previous button
        self.prev_button = tk.Button(self.root, text="Previous", font=("Arial", 20), bd=0, width=10, fg="black", bg="white", highlightthickness=0, highlightbackground="#ffffff",
                                     command=self.previous_image)
        self.prev_button.place(x=540, y=10)

        # Next button
        self.next_button = tk.Button(self.root, text="Next", font=("Arial", 20), bd=0, width=10, fg="black", bg="white", highlightthickness=0, highlightbackground="#ffffff",
                                     command=self.next_image)
        self.next_button.place(x=700, y=10)

        # Start the learning process
        self.learn()

    def learn(self):
        # Get current image
        image_file = self.image_files[self.curr_index]
        image_path = os.path.join(self.image_folder, image_file)

        try:
            self.image = Image.open(image_path)
            self.image.thumbnail((400, 400)) 
            self.photo = ImageTk.PhotoImage(self.image)

            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo

            # Extract alphabet from filename
            alphabet = image_file.split(".")[0].upper()
            self.alphabet_label.config(text=f"This sign indicates the letter {alphabet} in ASL", bg="white", fg="black", highlightthickness=0, font=("Arial", 25))


        except Exception as ex:
            print(f"Error loading image: {ex}")

        self.update_buttons()


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

    def exit_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", False)


class MultipleChoice:
    def __init__(self, root, back_func):
        self.root = root
        self.back_func = back_func
        self.initialize()

    def initialize(self):
        self.root.title("Multiple Choice - Sign Bridge")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="#A2C5CC")
        self.root.bind("<Escape>", self.exit_fullscreen)


        self.image_folder = "images/alphabetsQuiz"
        self.image_files = [f for f in os.listdir(self.image_folder) if os.path.isfile(os.path.join(self.image_folder, f))]

        self.ask_question()

        back_button = tk.Button(self.root, text="Back", font=("Arial", 12), bd=0, width=10, bg="white", fg="black", highlightthickness=0, highlightbackground="white",
                                command=lambda: self.back_func(self.root))
        back_button.place(x=10, y=10)


    def ask_question(self):
        self.clear_widgets()

        filtered_images = [img for img in self.image_files if img.split(".")[0].upper() != "M"]

        random_image_file = random.choice(filtered_images)
        image_path = os.path.join(self.image_folder, random_image_file)

        self.image = Image.open(image_path)
        self.image.thumbnail((300, 300)) 
        self.photo = ImageTk.PhotoImage(self.image)

        image_label = tk.Label(self.root, image=self.photo, bd=0, highlightthickness=0, bg="white")
        image_label.pack(pady=50) 

        correct_answer = random_image_file.split(".")[0].upper()  

        question_text = "Which alphabet does this hand sign indicate?"
        question_label = tk.Label(self.root, text=question_text, font=("Arial", 20), bd=0, bg="white", fg="black")
        question_label.pack(pady=10)

        answers = [correct_answer]
        incorrect_options = self.get_incorrect_options(correct_answer)

        answers.extend(incorrect_options)
        random.shuffle(answers)  

        self.answer_buttons = []
        for answer in answers:
            button = tk.Button(self.root, text=answer, font=("Arial", 25), width=20, bd=0, highlightthickness=0, bg="#5BE1C7",
                               highlightbackground="white")
            button.pack(pady=10)  
            button.config(command=lambda ans=answer: self.check_answer(ans, correct_answer))
            self.answer_buttons.append(button)

    def get_incorrect_options(self, correct_answer):
        incorrect_options = []
        all_image_files = [f for f in self.image_files if f != correct_answer.lower() + ".png"]  # Exclude the correct answer from the list

        while len(incorrect_options) < 3:
            random_image_file = random.choice(all_image_files)
            incorrect_option = random_image_file.split(".")[0].upper()
            if incorrect_option != correct_answer and incorrect_option not in incorrect_options:
                incorrect_options.append(incorrect_option)

        return incorrect_options

    def check_answer(self, selected_answer, correct_answer):
        if selected_answer == correct_answer:
            result_text = "Correct!"
        else:
            result_text = "Incorrect! The correct answer was " + correct_answer + "."

        result_label = tk.Label(self.root, text=result_text, font=("Arial", 20), bd=0, bg="white", fg="black", highlightthickness=0)
        result_label.pack(pady=10)

        self.root.after(2000, self.display_next_question_button)

    def display_next_question_button(self):
        next_button = tk.Button(self.root, text="Next Question", font=("Arial", 20), bd=0, width=20, bg="white", fg="black", highlightbackground="white",
                                command=self.ask_question, highlightthickness=0)
        next_button.pack(pady=10)

        # Disable answer buttons until the next question
        for button in self.answer_buttons:
            button.config(state=tk.DISABLED)

    def clear_widgets(self):
        for widget in self.root.winfo_children():
            widget.pack_forget()

    def exit_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", False)



class WebcamQuiz:
    def __init__(self, root, back_func):
        self.root = root
        self.back_func = back_func
        self.initialize()

    def initialize(self):
        self.root.configure(bg="white")
        self.correct_answers = 0
        self.total_answers = 0
        self.webcam_quiz()

        # Back button
        back_button = tk.Button(self.root, text="Back", font=("Arial", 12), bd=0, width=10,
                                command=lambda: self.back_func(self.root))
        back_button.place(x=10, y=10)

        

    def clear_widgets(self):
        for widget in self.root.winfo_children():
            widget.pack_forget()

    def get_random_alphabet(self):
        choices = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        return random.choice(choices)
        
    def webcam_quiz(self):
        self.clear_widgets()
        self.root.withdraw()
        self.root.configure(bg="white")

        cap = cv2.VideoCapture(0)

        show_start_message = True
        show_start_message_time = 0

        current_alphabet = self.get_random_alphabet()

        correct_answer_displayed = False
        correct_answer_time = 0
        next_question_time = 0

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 
                    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 
                    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 
                    24: 'Y', 25: 'Z'}

        skipped = []

        while True:
            data_aux_left = []  # left hand
            data_aux_right = []  # right hand
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = cap.read()

            if show_start_message:
                text = "Show your hand to start the quiz"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_color = (255, 255, 51)
                thickness = 2

                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = int((frame.shape[1] - text_size[0]) / 2)
                text_y = int((frame.shape[0] + text_size[1]) / 2)

                cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness)

            H, W, _ = frame.shape

            instructions = "Press 's' to skip the current question and 'q' to end this quiz"
            cv2.putText(frame, instructions, (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 51), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            predicted_character = ""

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # Determine if it's the left or right hand
                    if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:
                        data_aux = data_aux_left
                    else:
                        data_aux = data_aux_right

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

            if x_ and y_:
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

            if data_aux:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

            text = "Sign " + current_alphabet + "."
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (50, 50)
            font_scale = 1
            font_color = (255, 255, 51)
            thickness = 2
            cv2.putText(frame, text, position, font, font_scale, font_color, thickness)

            if predicted_character == current_alphabet and not correct_answer_displayed:
                    show_start_message = False
                    correct_answer_displayed = True
                    correct_answer_time = time.time()

            if correct_answer_displayed and time.time() - correct_answer_time >= 3:
                    correct_answer_displayed = False
                    next_question_time = time.time()
                    current_alphabet = self.get_random_alphabet()
                    self.correct_answers += 1
                    self.total_answers += 1

            if correct_answer_displayed:
                text = "Correct!"
                font = cv2.FONT_HERSHEY_SIMPLEX
                position = (100, 100)
                font_scale = 1
                font_color = (255, 255, 51)
                thickness = 2
                cv2.putText(frame, text, position, font, font_scale, font_color, thickness)

            if time.time() - next_question_time >= 3:
                text = "Sign " + current_alphabet + "."

            skip_button_pressed = cv2.waitKey(1) & 0xFF == ord('s')

            if skip_button_pressed:
                skipped.append(current_alphabet)
                current_alphabet = self.get_random_alphabet()
                self.total_answers += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                self.back_func(self.root) 
                return

            cv2.imshow('frame', frame)
            cv2.waitKey(1)



root = tk.Tk()
root.geometry("800x600")
home = HomeScreen(root)


root.mainloop()