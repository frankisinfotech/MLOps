import numpy as np
import pickle
import json

with open('model/model.pkl', 'rb') as file:
    model = pickle.load(file)

predictions = model.predict(np.array([[1,500]]))
print("Prediced Rental Price: ", predictions[0])
