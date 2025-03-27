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


# Model 
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
        self.root.configure(bg="white")
        self.root.bind("<Escape>", self.exit_fullscreen)

        title_label = tk.Label(self.root, text="Welcome to Sign Bridge!", font=("Arial", 18), bd=0)
        title_label.pack(pady=100)

        learn = tk.Button(self.root, text="Learn", font=("Arial", 14), bd=0, width=20, relief="flat", highlightthickness=0, highlightbackground="#ffffff", highlightcolor="#ffffff", command=self.learn)
        multiple_choice = tk.Button(self.root, text="Multiple Choice", font=("Arial", 14), bd=0, width=20, relief="flat", highlightthickness=0, highlightbackground="#ffffff", highlightcolor="#ffffff", command=self.multiple_choice)
        webcam_quiz = tk.Button(self.root, text="Webcam Quiz", font=("Arial", 14), bd=0, width=20, relief="flat", highlightthickness=0, highlightbackground="#ffffff", highlightcolor="#ffffff", command=self.webcam_quiz)
        
        # sign_to_text = tk.Button(self.root, text="Sign to Text", font=("Arial", 14), bd=0, width=20,
        #                            command=self.sign_to_text)


        learn.pack()
        multiple_choice.pack()
        webcam_quiz.pack()
        # sign_to_text.pack()

    def learn(self):
        self.root.withdraw()
        learn_window = tk.Toplevel(self.root)
        learn_window.title("Learn Alphabets")
        learn_window.attributes('-fullscreen', True) 
        self.root.configure(bg="white")
        learn_window.bind("<Escape>", lambda event: learn_window.attributes('-fullscreen', False))

        Learning(learn_window, self.back_to_home)
    
    def multiple_choice(self):
        self.root.withdraw()  
        quiz_window = tk.Toplevel(self.root)
        quiz_window.title("Alphabet Quiz")
        quiz_window.attributes('-fullscreen', True)
        self.root.configure(bg="white")
        quiz_window.bind("<Escape>", lambda event: quiz_window.attributes('-fullscreen', False))

        MultipleChoice(quiz_window, self.back_to_home)

    def webcam_quiz(self):
        self.root.withdraw()
        quiz_window = tk.Toplevel(self.root)
        quiz_window.attributes('-fullscreen', True)
        self.root.configure(bg="white")
        quiz_window.bind("<Escape>", lambda event: quiz_window.attributes('-fullscreen', False))
        WebcamQuiz(quiz_window, self.back_to_home)

    
    # def sign_to_text(self):
    #     self.root.withdraw()
    #     stt_window = tk.Toplevel(self.root)
    #     stt_window.title("Translate Sign Language into Text")
    #     stt_window.geometry("500x800")

    #     SignToText(stt_window, self.back_to_home)

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
        self.root.configure(bg="white")
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


class MultipleChoice:
    def __init__(self, root, back_func):
        self.root = root
        self.back_func = back_func
        self.initialize()

    def initialize(self):
        self.image_folder = "images/alphabetsQuiz"
        self.root.configure(bg="white")
        self.image_files = [f for f in os.listdir(self.image_folder) if os.path.isfile(os.path.join(self.image_folder, f))]

        self.ask_question()

        back_button = tk.Button(self.root, text="Back", font=("Arial", 12), bd=0, width=10,
                                command=lambda: self.back_func(self.root))
        back_button.place(x=10, y=10)

    def ask_question(self):
        self.clear_widgets()

        random_image_file = random.choice(self.image_files)
        image_path = os.path.join(self.image_folder, random_image_file)

        self.image = Image.open(image_path)
        self.image.thumbnail((300, 300))  # Adjust the maximum thumbnail size as desired
        self.photo = ImageTk.PhotoImage(self.image)

        image_label = tk.Label(self.root, image=self.photo, bd=0, highlightthickness=0)
        image_label.pack(pady=50) 

        correct_answer = random_image_file.split(".")[0].upper()  

        question_text = "Which alphabet does this hand sign indicate?"
        question_label = tk.Label(self.root, text=question_text, font=("Arial", 14), bd=0)
        question_label.pack(pady=10)

        answers = [correct_answer]
        incorrect_options = self.get_incorrect_options(correct_answer)

        answers.extend(incorrect_options)
        random.shuffle(answers)  

        self.answer_buttons = []
        for answer in answers:
            button = tk.Button(self.root, text=answer, font=("Arial", 12), width=20, bd=0, highlightthickness=0,
                               highlightbackground="#B8E2F2")
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

        result_label = tk.Label(self.root, text=result_text, font=("Arial", 14), bd=0)
        result_label.pack(pady=10)

        self.root.after(2000, self.display_next_question_button)

    def display_next_question_button(self):
        next_button = tk.Button(self.root, text="Next Question", font=("Arial", 12), bd=0, width=20,
                                command=self.ask_question, highlightthickness=0)
        next_button.pack(pady=10)

        # Disable answer buttons until the next question
        for button in self.answer_buttons:
            button.config(state=tk.DISABLED)

    def clear_widgets(self):
        for widget in self.root.winfo_children():
            widget.pack_forget()



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
            # Initialize data_aux here
            data_aux_left = []  # for the left hand
            data_aux_right = []  # for the right hand
            data_aux = []  # Initialize the main variable to store hand landmarks data
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
            predicted_character = ""  # Initialize the predicted_character variable here

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

            # Make sure data_aux is always initialized before prediction
            if data_aux:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]  # Assign prediction to predicted_character

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

            # Check only if predicted_character is assigned
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

        
    

# class SignToText:
#     def __init__(self, root, back_func):
#         self.root = root
#         self.back_func = back_func
#         self.initialize()
    
#     def initialize(self):
#         self.translate()
#         back_button = tk.Button(self.root, text="Back", font=("Arial", 12), bd=0, width=10,
#                                 command=lambda: self.back_func(self.root))
#         back_button.place(x=10, y=10)
    
#     def translate(self):
#         self.clear_widgets()
#         start_time = time.time()
#         result = ""
#         data_dict = pickle.load(open('./data/data.pickle', 'rb'))
#         data = np.asarray(data_dict['data'])
#         labels = np.asarray(data_dict['labels'])
#         x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
#         model = RandomForestClassifier()
#         model.fit(x_train, y_train)
#         y_predict = model.predict(x_test)
#         model_dict = pickle.load(open('./data/model.p', 'rb'))
#         model = model_dict['model']

#         # Change to 0, 1, or 2 if error occurs
#         cap = cv2.VideoCapture(0)


#         mp_hands = mp.solutions.hands
#         mp_drawing = mp.solutions.drawing_utils
#         mp_drawing_styles = mp.solutions.drawing_styles

#         hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

#         labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 
#                     8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
#         while True:

#             elapsed_time = time.time() - start_time

#             data_aux = []
#             x_ = []
#             y_ = []

#             ret, frame = cap.read()

#             H, W, _ = frame.shape

#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             results = hands.process(frame_rgb)
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp_drawing.draw_landmarks(
#                         frame,  # image to draw
#                         hand_landmarks,  # model output
#                         mp_hands.HAND_CONNECTIONS,  # hand connections
#                         mp_drawing_styles.get_default_hand_landmarks_style(),
#                         mp_drawing_styles.get_default_hand_connections_style())

#                 for hand_landmarks in results.multi_hand_landmarks:
#                     for i in range(len(hand_landmarks.landmark)):
#                         x = hand_landmarks.landmark[i].x
#                         y = hand_landmarks.landmark[i].y

#                         x_.append(x)
#                         y_.append(y)

#                     for i in range(len(hand_landmarks.landmark)):
#                         x = hand_landmarks.landmark[i].x
#                         y = hand_landmarks.landmark[i].y
#                         data_aux.append(x - min(x_))
#                         data_aux.append(y - min(y_))

#                 x1 = int(min(x_) * W) - 10
#                 y1 = int(min(y_) * H) - 10

#                 x2 = int(max(x_) * W) - 10
#                 y2 = int(max(y_) * H) - 10

#                 prediction = model.predict([np.asarray(data_aux)])

#                 predicted_character = labels_dict[int(prediction[0])]


#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#                 cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#                             cv2.LINE_AA)
            
                
#                 if elapsed_time >= 2:
#                     result += predicted_character
#                     start_time = time.time()

#                 elif elapsed_time < 2:
#                     result += " "
#                     start_time = time.time()
                
#             display_frame = frame.copy()
                
#             # prints the result of accumulated letters, but adjust the location
#             cv2.putText(display_frame, result, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (51, 255, 255), 3,
#                     cv2.LINE_AA)
            
            
#             cv2.imshow('frame', display_frame)
#             cv2.waitKey(1)

#             # clear the data for the next iteration
#             data_aux.clear()
#             x_.clear()
#             y_.clear()
        
#             cap.release()
#             cv2.destroyAllWindows()

#     def perform_translation(self):
#         # Get the text from the entry field
#         text = self.gesture_entry.get()

#         # Implement translation logic here

#         # Display the translated text
#         # translation_label = tk.Label(self.root, text=translated_text, font=("Arial", 12), bd=0)
#         # translation_label.pack(pady=10)

#     def clear_widgets(self):
#         for widget in self.root.winfo_children():
#             widget.pack_forget()

    
    

root = tk.Tk()
root.geometry("800x600")
root.configure(bg="white")
home = HomeScreen(root)

root.mainloop()