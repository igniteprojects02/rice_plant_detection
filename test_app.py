# pip install -r requirements.txt


from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Initialize TensorFlow model and class labels
model = tf.keras.models.load_model("rice_disease_model.h5")
class_labels = np.load("class_labels.npy", allow_pickle=True).item()

UPLOAD_FOLDER = 'static/upload'
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

def predict_disease(image_path, confidence_threshold=70):
    """
    Predicts the disease in the plant image. If the confidence is below the threshold,
    it reports the leaf as healthy.
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
        return predicted_label, confidence, recommended_medicines
    else:
        return "Healthy Leaf", confidence, []

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
            # Save the uploaded image to the local filesystem
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Get the selected weather condition
            weather = request.form.get('weather', 'None')

            # Perform prediction
            predicted_label, confidence, recommended_medicines = predict_disease(image_path)

            # Diseases related to the selected weather condition
            weather_diseases = {
                'Sunny': ['Brownspot', 'Leafsmut'],
                'Rainy': ['Bacterialblight', 'Leafsmut'],
                'Cloudy': ['Bacterialblight'],
                'Windy': ['Brownspot']
            }.get(weather, [])

            # Render result with medicines and weather diseases
            return render_template(
                'index.html',
                result=predicted_label,
                confidence=f"Confidence: {confidence:.2f}%",
                recommended_medicines=recommended_medicines,
                image_path=image_path,
                weather=weather,
                weather_diseases=weather_diseases
            )

    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # 204 No Content


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
