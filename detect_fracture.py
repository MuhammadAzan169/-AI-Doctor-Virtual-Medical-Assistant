import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import os

model_elbow_frac = tf.keras.models.load_model("fracture_models/ResNet50_Elbow_frac_best.h5")
model_hand_frac = tf.keras.models.load_model("fracture_models/ResNet50_Hand_frac_best.h5")
model_shoulder_frac = tf.keras.models.load_model("fracture_models/ResNet50_Shoulder_frac_best.h5")
model_parts = tf.keras.models.load_model("fracture_models/ResNet50_BodyParts.h5")

categories_parts = ["elbow", "wrist", "shoulder"]
categories_fracture = ["fractured", "normal"]

def predict_fracture(image_path):
    size = 224
    temp_img = image.load_img(image_path, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    part_idx = np.argmax(model_parts.predict(images), axis=1).item()
    body_part = categories_parts[part_idx]

    if body_part == "wrist":
        part_model = model_hand_frac
        model_name = "Hand"
    elif body_part == "elbow":
        part_model = model_elbow_frac
        model_name = "Elbow"
    elif body_part == "shoulder":
        part_model = model_shoulder_frac
        model_name = "Shoulder"

    frac_idx = np.argmax(part_model.predict(images), axis=1).item()
    fracture_status = categories_fracture[frac_idx]

    if fracture_status == "fractured":
        return f"Fracture detected on {model_name}"
    else:
        return f"No fracture detected on {model_name}"