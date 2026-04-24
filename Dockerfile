FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

WORKDIR /workspace/yeppoh

# System deps for Genesis rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default: run training
CMD ["python", "scripts/train.py"]
