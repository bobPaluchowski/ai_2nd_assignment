from mtcnn import MTCNN
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# New directory structure
base_dir = 'ai'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'validation')

# Define image size and number of classes
img_height, img_width = 224, 224
num_classes = 5

# Data generator with data augmentation for the training set
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_datagen = ImageDataGenerator(rescale=1./255)

# Data generator for the testing and validation sets
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training, testing, and validation sets from respective directories
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=32,
                                                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(img_height, img_width),
                                                  batch_size=32,
                                                  class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(img_height, img_width),
                                                        batch_size=32,
                                                        class_mode='categorical')

# Load pre-trained base model without the top classification layer
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                               include_top=False,
                                               weights='imagenet')

# Freeze the base model layers
base_model.trainable = False

# Create the classifier model
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)

# Compile and train the model
model.compile(optimizer=optimizers.legacy.Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 20

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=epochs)

# Evaluate the model's performance on the testing set
loss, accuracy = model.evaluate(test_generator)
print(f"Test loss: {loss}, Test accuracy: {accuracy}")

# Save the model for future use
model.save("face_recognition_model.h5")

# Load the saved model
model = load_model("face_recognition_model.h5")


