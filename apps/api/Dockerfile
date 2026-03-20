FROM python:3.11-slim

# System deps for audio processing (librosa/essentia need libsndfile + ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU first (pinned, separate index)
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch==2.10.0 --index-url ${TORCH_INDEX}

# Install local packages in dependency order
COPY packages/key-prediction packages/key-prediction
COPY packages/bpm-analysis packages/bpm-analysis
RUN pip install --no-cache-dir packages/key-prediction packages/bpm-analysis

# Install the API app
COPY apps/api apps/api
RUN pip install --no-cache-dir apps/api

# Copy model checkpoint
COPY packages/key-prediction/checkpoints checkpoints

EXPOSE 8000

ENV MODEL_PATH=checkpoints/keynet.pt
ENV DEVICE=cpu

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
