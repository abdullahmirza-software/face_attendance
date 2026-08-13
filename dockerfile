FROM python:3.10-slim

# Install system dependencies explicitly mapped for OpenCV / UniFace on Debian
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxrender1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy and install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files and ensure correct permissions
COPY --chown=user . .
USER user

# Bake the face-detection/recognition ONNX weights into the image so
# startup never depends on GitHub being reachable (it was intermittently
# closing connections mid-download, crashing boot). Runs as `user` so the
# cache lands in /home/user/.uniface/models, the same $HOME the app uses
# at runtime.
RUN python prefetch_models.py

# Hugging Face configuration
ARG HOST
ARG PORT
ENV HOST=${HOST:-0.0.0.0}
ENV PORT=${PORT:-7860}
EXPOSE ${PORT:-7860}

CMD uvicorn app:app --host $HOST --port $PORT
