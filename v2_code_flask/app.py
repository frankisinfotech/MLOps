import numpy as np
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Predict using the model
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Rental Price Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    model_path = 'model/rental_prediction_model.pkl'
    model = pickle.load(open(model_path, 'rb'))

    # Pass the input data to the model
    user_input = request.json

    # Rooms and area are the features used for prediction
    # Ensure the input is in the correct format
    rooms = int(user_input.get('rooms',0))
    area = int(user_input.get('area',0))

    user_input_preprocessed = np.array([[rooms, area]])

    # Make a prediction
    prediction = model.predict(user_input_preprocessed)
    output = {"Rental Prediction using Built Model V3": float(prediction[0])}

    return output

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
