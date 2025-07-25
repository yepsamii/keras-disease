# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirement.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirement.txt

# Copy project files
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Set environment variable for Flask
ENV FLASK_APP=app.py

# Run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"] 