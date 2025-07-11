# Required libraries
import base64
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flask import Flask, request, jsonify, render_template

# Network architecture
class Network(nn.Module): # Creates a custom neural network
    def __init__(self): # Constructor
        super(Network, self).__init__() # Sets up the constructor
        self.hidden_0 = nn.Linear(784, 128) # First hidden layer of 784 pixels into 128
        self.hidden_1 = nn.Linear(128, 64) # Second hidden layer of 128 to 64
        self.output = nn.Linear(64, 10) # Output layer of 64 to 10 possible outputs (0 to 9)
        self.softmax = nn.LogSoftmax(dim=1) # Applies logsoftmax over the 10 outputs for classification
        self.activation = nn.ReLU() # Applies ReLU activation function to add non-linearity
        self.dropout = nn.Dropout(0.25) # Randomly drops 25% of the neurons during training to help prevent overfitting

    def forward(self, x): # Defines how the data flows in the neural network
        x = x.view(x.shape[0], -1) # Flattens 28 pixels by 28 pixel images into one 784 
        x = self.hidden_0(x) # Applies the first linear transformation
        x = self.activation(x) # Applies ReLU
        x = self.dropout(x) # Applies dropout
        x = self.hidden_1(x) # Applies the second linear transformation
        x = self.activation(x) # Applies ReLU
        x = self.dropout(x) # Applies dropout
        x = self.output(x) # Applies the third linear transformation
        x = self.softmax(x) # Does logsoftmax
        
        return x # Returns the loglikelihoods of 0-9 (Ex: [0.0, 0.1, 0.0, 0.0, 0.0, 0.7, 0.1, 0.1, 0.0, 0.0])

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = Network()
model.load_state_dict(torch.load('MNIST_model.pth', map_location=torch.device('cpu')))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data['image']

    # Remove the prefix 'data:image/png;base64,'
    img_str = img_data.split(",")[1]

    # Decode base64 string to bytes
    img_bytes = base64.b64decode(img_str)

    # Open image with PIL
    img = Image.open(BytesIO(img_bytes)).convert('L')  # convert to grayscale

    img = img.resize((28, 28))

    # Convert image to numpy array and normalize
    img_array = np.array(img).astype(np.float32)
    img_array = 255 - img_array  # invert colors (white background to black)
    img_array /= 255.0

    # Reshape to match model input (batch_size, 1, 28, 28)
    tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.exp(output)
        prediction = torch.argmax(probs, dim=1).item()

    return jsonify({'prediction': prediction})

# Displays index.html
@app.route("/")
def home():
    return render_template("index.html") 

# Runs the app with debug information
if __name__ == "__main__":
    app.run(debug=True)
