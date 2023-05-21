# app2.py

import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from face_recognition import init_model, init_detector, preprocess_and_predict

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Attendance Register")
        self.geometry("640x480")
        
        # Center the application window
        window_width, window_height = 640, 480
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x_coordinate = int((screen_width / 2) - (window_width / 2))
        y_coordinate = int((screen_height / 2) - (window_height / 2))
        self.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

        self.model = init_model("face_recognition_model.h5")
        self.detector = init_detector()

        self.start_button = ttk.Button(self, text="Register your attendance", command=self.start_webcam)
        self.start_button.pack(pady=20)
        self.message_label = ttk.Label(self, text="", font=("Helvetica", 16))
        self.message_label.pack(pady=20)
        self.canvas = tk.Canvas(self, width=640, height=360)
        self.canvas.pack()

    def start_webcam(self):
        cap = cv2.VideoCapture(0)

        def update_frame():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                image_tk = ImageTk.PhotoImage(image)
                self.canvas.create_image(0, 0, anchor=tk.CENTER, image=image_tk)
                self.canvas.image = image_tk
                faces = self.detector.detect_faces(frame)

                for face in faces:
                    x, y, width, height = face['box']
                    face_crop = frame[y:y + height, x:x + width]
                    name = preprocess_and_predict(face_crop, self.model)

                    if name == "Unknown":
                        self.message_label.configure(text=f"Hi, would you like to register for classes?")

                    else:
                        # Register the person's attendance and display the message
                        self.register_attendance(name)
                    break

                if not faces:
                    self.after(10, update_frame)

        update_frame()

    def register_attendance(self, name):
        self.message_label.configure(text=f"{name}, your attendance has been registered.")

if __name__ == "__main__":
    app = App()
    app.mainloop()

