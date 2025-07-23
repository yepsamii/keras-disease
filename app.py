from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

# Load the Keras model
model = load_model('model.keras')

# Define class names for skin disease classification
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Define full names corresponding to the class names
full_names = [
    'Actinic keratosis',
    'Basal cell carcinoma',
    'Benign keratosis-like lesions',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic nevi',
    'Vascular lesions'
]

def prepare_image(file):
    # Read the file as bytes and decode to an image in BGR format
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Resize to 224x224 pixels
    image = cv2.resize(image, (224, 224))
    # Normalize pixel values to [0, 1]
    image = image / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Process the image
            processed_image = prepare_image(file)
            # Make prediction
            prediction = model.predict(processed_image)
            # Get the predicted class index
            predicted_class_idx = np.argmax(prediction[0])
            # Get the full name of the predicted class
            predicted_full_name = full_names[predicted_class_idx]
            # Get the confidence score
            confidence = float(prediction[0][predicted_class_idx])
            # Render the template with prediction results
            return render_template('index.html', predicted_full_name=predicted_full_name, confidence=confidence)
    # Render the template without prediction for GET requests
    return render_template('index.html', predicted_full_name=None, confidence=None)

if __name__ == '__main__':
    app.run(debug=True)