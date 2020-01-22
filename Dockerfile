# Use an official Python runtime as a parent image
FROM python:3.7-slim-stretch

# Set the working directory to /dockerTutorial
WORKDIR /dockerTutorial

# Copy the current directory contents into the container at /dockerTutorial
COPY . /dockerTutorial

#Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt

# Define environment variable
ENV COUNT 0

# Run test.py when the container launches
CMD ["python", "train_and_classify_clinton_tweets.py"]
