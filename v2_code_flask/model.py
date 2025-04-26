# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import os

# Check if the required directories exist, if not create them
if not os.path.exists('model'):
    os.makedirs('model')

# Load the dataset
data_path = 'data/rental_1000.csv'

# Read the dataset
df = pd.read_csv(data_path)

# Feature engineering: Select Features (X) and Label (y)
X = df[['rooms', 'area']].values
y = df['price'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Linear Regression
lr = LinearRegression()
lr_model = lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)
lr_rmse = np.sqrt(np.mean((lr.predict(X_test) - y_test) ** 2))
print(f"Linear Regression - R^2 Score: {lr_score:.4f}, RMSE: {lr_rmse:.4f}")

# Save the best model (Linear Regression in this case) to a file
model = lr_model
model_path = 'model/rental_prediction_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved successfully at {model_path}!")
