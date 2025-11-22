# 1. Base Image (Lightweight Python)
FROM python:3.9-slim

# 2. Set Working Directory inside container
WORKDIR /app

# 3. Install System Dependencies (Required for OpenCV & Video)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy Requirements and Install Python Libs
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the project files
COPY . .

# 6. Expose the Port Streamlit uses
EXPOSE 8501

# 7. Command to run the app
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0"]