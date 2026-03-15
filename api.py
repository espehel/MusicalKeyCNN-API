import os
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from eval import load_model
from predict_keys import SUPPORTED_EXTENSIONS, camelot_output, preprocess_mp3
MODEL_PATH = Path(os.getenv("MODEL_PATH", "checkpoints/keynet.pt"))
DEVICE = torch.device(os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

app = FastAPI(title="MusicalKeyCNN", description="Predict the musical key of an audio file.")

model = None


@app.on_event("startup")
def load():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model checkpoint not found at {MODEL_PATH}")
    model = load_model(str(MODEL_PATH), DEVICE)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model": str(MODEL_PATH)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Use: {SUPPORTED_EXTENSIONS}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        spec = preprocess_mp3(tmp_path).to(DEVICE)
        with torch.no_grad():
            outputs = model(spec.unsqueeze(0))
            pred = int(torch.argmax(outputs, dim=1).cpu().item())

        camelot_str, key_text = camelot_output(pred)
        return JSONResponse({"file": file.filename, "camelot": camelot_str, "key": key_text, "id": pred})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)
