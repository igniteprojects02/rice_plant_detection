import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


model = tf.keras.models.load_model("rice_disease_model.h5")

class_labels = np.load("class_labels.npy", allow_pickle=True).item()

def predict_disease(image_path, confidence_threshold=99):
    """
    Predicts the disease in the plant image. If the confidence is below the threshold, 
    it reports no disease detected.

    :param image_path: Path to the plant image file.
    :param confidence_threshold: Minimum confidence (%) to report a disease.
    """
    
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

   
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions) * 100

    # Get the label or report no disease detected
    if confidence >= confidence_threshold:
        predicted_label = class_labels[predicted_class[0]]
        print(f"Predicted Disease: {predicted_label}")
        print(f"Confidence: {confidence:.2f}%")
    else:
        print("No disease detected with sufficient confidence.")
        print(f"Highest Confidence: {confidence:.2f}%")

# Test with an image
image_path = "paddy1.jpg"  # Replace with the path to your test image
predict_disease(image_path)