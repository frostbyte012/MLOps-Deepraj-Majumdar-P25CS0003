# Use a lightweight python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the scripts and data
COPY translate.py .
COPY evaluate.py .
# Note: Ensure input.txt and reference.txt are in the same folder when building
COPY input.txt .
COPY reference.txt .

# Run translation by default
CMD ["python", "translate.py"]