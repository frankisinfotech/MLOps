import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

rentalDF = pd.read_csv('data/rental_1000.csv')

# Features - X [ rooms && area ]
# Label    - y [ price]
X = rentalDF[['rooms','area']].values
y = rentalDF['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

lr = LinearRegression()
model = lr.fit(X_train,y_train)

# Save the model using Pickling (serialization)
with open('model/model.pkl', 'wb') as file:
    pickle.dump(model, file)
