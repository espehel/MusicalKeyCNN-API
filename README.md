# MusicalKeyCNN

Monorepo for audio analysis tools built around a CNN-based musical key classifier.

---

## Structure

```
packages/
  key-prediction/   CNN model for musical key detection (Camelot Wheel output)
  bpm-analysis/     BPM detection using Essentia
apps/
  api/              FastAPI service exposing key and BPM endpoints
```

---

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then sync the full workspace:

```sh
uv sync
```

This installs all packages and their dependencies into a shared virtual environment.

PyTorch is not auto-selected for CPU/CUDA — install the right build first:

```sh
# CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA — see https://pytorch.org/get-started/locally/
pip install torch
```

---

## Running the API

Use `uv run` to run commands inside the workspace environment without manually activating a venv:

```sh
uv run uvicorn api.main:app --reload
```

Or activate the venv once and use commands directly:

```sh
source .venv/bin/activate
uvicorn api.main:app --reload
```

Set environment variables to configure the model path and device:

```sh
MODEL_PATH=packages/key-prediction/checkpoints/keynet.pt DEVICE=cpu uv run uvicorn api.main:app --reload
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status, device, model path |
| POST | `/predict` | Predict musical key from an audio file |
| POST | `/analyze/bpm` | Detect BPM from an audio file |

Supported audio formats: `.mp3`, `.wav`, `.flac`, `.ogg`

**Example:**

```sh
curl -X POST http://localhost:8000/predict -F "file=@track.mp3"
# {"file":"track.mp3","camelot":"8A","key":"A minor","id":7}

curl -X POST http://localhost:8000/analyze/bpm -F "file=@track.mp3"
# {"file":"track.mp3","bpm":128.0,"confidence":0.9532,"beat_count":312,"beats":[...]}
```

---

## Docker

Build and run the API from the repo root:

```sh
docker build -f apps/api/Dockerfile -t musical-key-api .
docker run -p 8000:8000 musical-key-api
```

---

## Packages

- [`packages/key-prediction`](packages/key-prediction/README.md) — model architecture, training, evaluation, and prediction utilities
- `packages/bpm-analysis` — Essentia-based BPM analysis
