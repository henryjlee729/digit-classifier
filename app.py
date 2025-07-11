import base64
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the trained Keras model (.keras file)
model = tf.keras.models.load_model('mnist_digit_classifier.keras')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data['image']

    # Remove the base64 prefix
    img_str = img_data.split(",")[1]

    # Decode base64 to bytes
    img_bytes = base64.b64decode(img_str)

    # Open the image and preprocess
    img = Image.open(BytesIO(img_bytes)).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28

    img_array = np.array(img).astype(np.float32)
    img_array = 255 - img_array  # Optional: invert if necessary (white background, black digits)
    img_array /= 255.0  # Normalize to [0,1]

    # Reshape for Keras: (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=(0, -1))

    # Predict using the Keras model
    predictions = model.predict(img_array)
    predicted_label = int(np.argmax(predictions, axis=1)[0])

    return jsonify({'prediction': predicted_label})

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
