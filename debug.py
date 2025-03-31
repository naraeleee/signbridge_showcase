class Learning:
    def __init__(self, root, back_func):
        self.root = root
        self.back_func = back_func
        self.initialize()

    def initialize(self):
        self.root.title("Learning - Sign Bridge")
        self.root.attributes('-fullscreen', True)
        self.root.bind("<Escape>", self.exit_fullscreen)

        # Load background image
        try:
            self.background_image = Image.open("background.jpeg")  # Replace with your image file
            self.background_photo = ImageTk.PhotoImage(self.background_image)
        except FileNotFoundError:
            print("Error: background.jpeg not found.")
            return  # Exit if image not found

        # Create canvas for background image
        self.canvas = tk.Canvas(self.root, width=self.background_photo.width(), height=self.background_photo.height())
        self.canvas.create_image(0, 0, image=self.background_photo, anchor="nw")
        self.canvas.image = self.background_photo  # Keep a reference
        self.canvas.pack(fill="both", expand=True)

        # Initialize the learning content (alphabet images)
        self.image_folder = "images/alphabetsLearning"
        
        # Get sorted list of image files (ensuring alphabetical order)
        self.image_files = sorted(
            [f for f in os.listdir(self.image_folder)
             if os.path.isfile(os.path.join(self.image_folder, f)) and not f.startswith('.')]
        )

        self.curr_index = 0  # Start from first letter

        # Label to display alphabet
        self.alphabet_label = tk.Label(self.root, font=("Arial", 14), bd=0, bg="white", fg="black", highlightthickness=0)
        self.alphabet_label.place(relx=0.5, rely=0.1, anchor="center")

        # Image Label
        self.image_label = tk.Label(self.root, bd=0, highlightthickness=0, bg="white")
        self.image_label.place(relx=0.5, rely=0.4, anchor="center")

        # Back button
        self.back_button = tk.Button(self.root, text="Back", font=("Arial", 12), bd=0, width=10, fg="black", bg="white", highlightthickness=0, highlightbackground="#ffffff",
                                     command=lambda: self.back_func(self.root))
        self.back_button.place(x=10, y=10)

        # Previous button
        self.prev_button = tk.Button(self.root, text="Previous", font=("Arial", 12), bd=0, width=10, fg="black", bg="white", highlightthickness=0, highlightbackground="#ffffff",
                                     command=self.previous_image)
        self.prev_button.place(x=300, y=10)

        # Next button
        self.next_button = tk.Button(self.root, text="Next", font=("Arial", 12), bd=0, width=10, fg="black", bg="white", highlightthickness=0, highlightbackground="#ffffff",
                                     command=self.next_image)
        self.next_button.place(x=450, y=10)

        # Start the learning process
        self.learn()

    def learn(self):
        # Get current image
        image_file = self.image_files[self.curr_index]
        image_path = os.path.join(self.image_folder, image_file)

        # Extract alphabet from filename
        alphabet = image_file.split(".")[0].upper()
        self.alphabet_label.config(text=f"This sign indicates the letter {alphabet} in ASL", bg="white", fg="black", highlightthickness=0)

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

    def exit_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", False)