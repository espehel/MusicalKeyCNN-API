# key-prediction

CNN-based musical key classification using the architecture from Korzeniowski & Widmer (2018) [\[1\]](#literature). Provides a pre-trained model and utilities for preprocessing, training, evaluation, and prediction using the [Camelot Wheel](https://mixedinkey.com/camelot-wheel/) convention.

---

## Table of Contents

- [Installation](#installation)
- [Key Prediction](#key-prediction)
- [Dataset Preparation](#dataset-preparation)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Literature](#literature)

---

## Installation

```sh
pip install -e .
```

PyTorch is listed as a dependency but you may want to install a specific build (CPU/CUDA) first:

```sh
# CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA — see https://pytorch.org/get-started/locally/ for the right command
pip install torch
```

---

## Key Prediction

Predict the musical key of a single file or a folder of audio files using the pre-trained model:

```sh
python -c "
from key_prediction import load_model, preprocess_mp3, camelot_output
import torch

device = torch.device('cpu')
model = load_model('checkpoints/keynet.pt', device)
spec = preprocess_mp3('your_song.mp3').to(device)
with torch.no_grad():
    pred = int(torch.argmax(model(spec.unsqueeze(0)), dim=1))
camelot_str, key_text = camelot_output(pred)
print(camelot_str, key_text)
"
```

**Output fields:**

| Field | Example | Description |
|-------|---------|-------------|
| Camelot | `8A` | Position on the Camelot Wheel |
| Key | `A minor` | Musical key name |
| ID | `7` | Class index (0–23) |

---

## Dataset Preparation

Training and evaluation require the GiantSteps datasets:

- [GiantSteps MTG Key Dataset](https://github.com/GiantSteps/giantsteps-mtg-key-dataset) — training
- [GiantSteps Key Dataset](https://github.com/GiantSteps/giantsteps-key-dataset) — evaluation

Place or symlink them under a `Dataset/` folder (relative to where you run the scripts):

```
Dataset/
    giantsteps-key-dataset/
    giantsteps-mtg-key-dataset/
```

---

## Preprocessing

Generate CQT spectrograms for all tracks and pitch-shifted variants before training:

```sh
python preprocess_data.py
```

Output `.pkl` files are written to `Dataset/mtg-preprocessed-audio/` and `Dataset/giantsteps-preprocessed-audio/`.

---

## Training

```sh
python train.py
```

The best checkpoint is saved to `checkpoints/keynet.pt`. Edit `train.py` to adjust hyperparameters (batch size, learning rate, etc.).

---

## Evaluation

Compute MIREX scores on the GiantSteps evaluation set:

```sh
python -c "
from pathlib import Path
from torch.utils.data import DataLoader
from key_prediction import load_model, evaluate_mirex, print_mirex_report
from key_prediction.dataset import KeyDataset
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = KeyDataset('Dataset/giantsteps-key-dataset', 'Dataset/giantsteps-preprocessed-audio', chunk_samples=float('inf'), pitch_range=(0,0))
loader = DataLoader(dataset, batch_size=1, shuffle=False)
model = load_model('checkpoints/keynet.pt', device)
scores = evaluate_mirex(model, loader, device)
print_mirex_report(scores)
"
```

Results on GiantSteps:

| Method | Weighted | Correct | Fifth | Relative | Parallel | Other |
|--------|----------|---------|-------|----------|----------|-------|
| `keynet.pt` | 73.51 | 66.72 | 8.11 | 6.79 | 3.48 | 14.90 |
| Mixed In Key 10.3 | 75.70 | 69.37 | 8.11 | 5.13 | 3.64 | 13.74 |
| RekordBox 7.12 | 65.53 | 56.79 | 11.92 | 5.96 | 4.97 | 20.36 |

Scoring follows the [MIREX key evaluation](https://www.music-ir.org/mirex/wiki/2025:Audio_Key_Detection) standard: 1.0 × correct + 0.5 × fifth + 0.3 × relative + 0.2 × parallel.

---

## Literature

- \[1\] F. Korzeniowski and G. Widmer. "Genre-Agnostic Key Classification With Convolutional Neural Networks". *ISMIR* (2018) — [arXiv](https://arxiv.org/abs/1808.05340)
- \[2\] F. Korzeniowski and G. Widmer. "End-to-End Musical Key Estimation Using a Convolutional Neural Network". *EUSIPCO* (2017) — [arXiv](https://arxiv.org/abs/1706.02921)
