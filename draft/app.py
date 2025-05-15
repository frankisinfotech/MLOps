import numpy as np
import pickle
from flask import Flask, request, jsonify

# Initialize the Flask application
app = Flask(__name__)

# Define the home route to provide a welcome message
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Rental Price Prediction API!"

# Define the prediction route to handle POST requests
@app.route('/predict', methods=['POST'])
def predict():
    # Path to the pre-trained model file
    model_path = 'model/model.pkl'
    
    # Load the pre-trained model using pickle
    model = pickle.load(open(model_path, 'rb'))

    # Parse the input JSON data from the request
    user_input = request.json

    # Extract the required features (rooms and area) from the input
    # Default to 0 if the values are not provided
    rooms = int(user_input.get('rooms', 0))
    area = int(user_input.get('area', 0))

    # Preprocess the input data into the required format for the model
    user_input_preprocessed = np.array([[rooms, area]])

    # Use the model to make a prediction
    prediction = model.predict(user_input_preprocessed)
    
    # Prepare the output in a JSON-friendly format
    output = {"Rental Prediction using Built Model V5": float(prediction[0])}

    # Return the prediction as a response
    return output

# Run the Flask application
if __name__ == '__main__':
    # Host the app on all available IPs, port 5000, with debug mode enabled
    app.run(host='0.0.0.0', port=5000, debug=True)
