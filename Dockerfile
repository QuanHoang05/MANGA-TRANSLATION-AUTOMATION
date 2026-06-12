FROM python:3.10-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create fonts directory and download a Vietnamese-supported font (Nunito Bold)
RUN mkdir -p /workspace/fonts && \
    wget -O /workspace/fonts/Nunito-Bold.ttf "https://github.com/google/fonts/raw/main/ofl/nunito/Nunito-Bold.ttf"

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Run the app
CMD ["python", "run.py"]
