# Use the official Python 3 base image
FROM python:3

# Set the working directory inside the container
WORKDIR /mlapp

# Copy the requirements file into the container
COPY requirements.txt ./

# Install the Python dependencies specified in the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pre-trained model file into the container
COPY model/model.pkl ./model/model.pkl

# Copy the application script into the container
COPY app.py ./app.py

# Expose port 5000 for the application
EXPOSE 5000

# Specify the command to run the application
CMD [ "python", "./app.py" ]
