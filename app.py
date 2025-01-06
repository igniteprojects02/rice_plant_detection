from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model and class labels
model = tf.keras.models.load_model("rice_disease_model.h5")
class_labels = np.load("class_labels.npy", allow_pickle=True).item()

UPLOAD_FOLDER = 'stastic/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dictionary for medicines based on the predicted disease
medicines = {
    'Bacterialblight': [
        "Copper oxychloride (fungicide)",
        "Streptomycin (antibiotic)",
        "Plant growth regulators (PGRs)"
    ],
    'Brownspot': [
        "Carbendazim (fungicide)",
        "Tricyclazole (fungicide)",
        "Tebuconazole (fungicide)"
    ],
    'Leafsmut': [
        "Benomyl (fungicide)",
        "Carbendazim (fungicide)",
        "Mancozeb (fungicide)"
    ]
}

def predict_disease(image_path, confidence_threshold=90):
    """
    Predicts the disease in the plant image. If the confidence is below the threshold,
    it reports no disease detected.
    """
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions) * 100

    if confidence >= confidence_threshold:
        predicted_label = class_labels[predicted_class[0]]
        # Get medicine recommendations based on the predicted disease
        recommended_medicines = medicines.get(predicted_label, [])
        return f"Predicted Disease: {predicted_label}", f"Confidence: {confidence:.2f}%", recommended_medicines
    else:
        return "No disease detected with sufficient confidence.", f"Highest Confidence: {confidence:.2f}%", []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' not in request.files:
            return render_template('index.html', error="No file uploaded.")
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="No file selected.")

        if file:
            # Save the uploaded image
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # Perform prediction
            result, confidence, recommended_medicines = predict_disease(image_path)

            # Render result with medicines
            return render_template('index.html', result=result, confidence=confidence, recommended_medicines=recommended_medicines, image_path=image_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
