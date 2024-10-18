# Use a more specific base image
FROM python:3.9.16-slim-buster

# Setting environment variable to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system dependencies with retry logic
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Hugging Face Transformers and Flask for the API
RUN pip install --no-cache-dir transformers[torch] flask

# Copy the inference script to the container
COPY inference.py /app/inference.py

# Set the working directory
WORKDIR /app

# Expose the port for the Flask API
EXPOSE 5000

# Command to run the Flask API
CMD ["python", "inference.py"]