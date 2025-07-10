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
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.hidden_0 = nn.Linear(784, 128)
        self.hidden_1 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
        self.softmax = nn.LogSoftmax(dim=1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.hidden_0(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.hidden_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

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

if __name__ == "__main__":
    app.run(debug=True)
