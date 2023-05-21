import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# Load saved model
model = load_model("face_recognition_model.h5")

detector = MTCNN()

class_labels = {
    0: 'A',
    1: 'K',
    2: 'N',
    3: 'R',
    4: 'Robert'
}

img_height, img_width = 224, 224

def preprocess_and_predict(face):
    face_resized = cv2.resize(face, (img_height, img_width))
    face_normalized = face_resized.astype("float32") / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=0)
    probabilities = model.predict(face_expanded)
    predicted_class = np.argmax(probabilities, axis=1)

    if np.max(probabilities) > 0.8: # treshold value
        return class_labels[predicted_class[0]]
    else:
        return None

# Initialise the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect faces
    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, width, height = face['box']
        face_crop = frame[y:y+height, x:x+width]
        
        # Predict the person's name
        name = preprocess_and_predict(face_crop)

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
        if name is not None:
            cv2.putText(frame, f"Welcome {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Hi, would you like to register for classes?", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
