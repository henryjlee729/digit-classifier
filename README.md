# MNIST Digit Classifier

This is a simple web-based digit recognition app using a trained **Keras model** and **Flask** for the backend. Users can draw a digit (0–9) in the browser, and the model will predict the number in real time.  This is an introductory project for me to explore neural networks and deep learning for a common problem.

Model taken from [here](https://github.com/ZahraMohit/mnist-digit-classification).

## Features

- Draw digits on a web canvas
- Classify handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset
- Real-time predictions using a Flask API
- Lightweight and easy to set up

## Technologies Used

- Python
- TensorFlow / Keras
- Flask
- HTML + CSS + JavaScript (frontend)
- Pillow (PIL) for image processing
- NumPy

## Installation

### 1. Clone the Repository
```
git clone https://github.com/your-username/mnist-digit-classifier.git
cd mnist-digit-classifier
```

### 2. Install Required Libraries

Open your terminal or command prompt and run:
```
pip install flask tensorflow pillow numpy
```

### 3. Click run on app.py