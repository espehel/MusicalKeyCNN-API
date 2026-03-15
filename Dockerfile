FROM python:3.11-slim

# System deps for audio processing (librosa/torchaudio need libsndfile + ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU — pinned to match local version
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch==2.10.0 --index-url ${TORCH_INDEX}

COPY . .

EXPOSE 8000

ENV MODEL_PATH=checkpoints/keynet.pt
ENV DEVICE=cpu

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
