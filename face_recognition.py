import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

def init_model(model_path):
    return load_model(model_path)

def init_detector():
    return MTCNN()

class_labels = {
        0: 'A',
        1: 'K',
        2: 'N',
        3: 'R',
        4: 'Robert'
        }

img_height, img_width = 224, 224

def preprocess_and_predict(face, model):
    face_resized = cv2.resize(face, (img_height, img_width))
    face_normalized = face_resized.astype("float32") / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=0)
    probabilities = model.predict(face_expanded)
    # predicted_class = np.argmax(probabilities, axis=1)
    max_prob_index = np.argmax(probabilities, axis=1)
    max_prob_value = np.max(probabilities, axis=1)

    # If confidence is below the treshold, return "Unknown"
    if max_prob_value < 0.9:
        return "Unknown"

    return class_labels[max_prob_index[0]]
