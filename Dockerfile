# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies for OpenCV and any other libraries needed for your app
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8501 (Flask or Gunicorn server will run here)
EXPOSE 8501

# Define environment variable to prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Command to run your Flask app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8501", "app:app"]
