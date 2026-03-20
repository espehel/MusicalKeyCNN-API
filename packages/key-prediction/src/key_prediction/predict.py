from pathlib import Path
import torch
import librosa
import numpy as np

from .dataset import CAMELOT_MAPPING

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


def preprocess_mp3(audio_path, sample_rate=44100, n_bins=105, hop_length=8820):
    """
    Loads an audio file, converts to mono, resamples, and extracts a log-magnitude CQT spectrogram.
    Then slices result as in MTG preprocessed dataset (removes last frequency bin and converts to torch tensor).

    Args:
        audio_path (Path): Path to audio file.
        sample_rate (int): Target sampling rate for audio.
        n_bins (int): Number of CQT bins.
        hop_length (int): Hop length for CQT.

    Returns:
        torch.Tensor: Shape (1, freq_bins, time_frames), ready for model input.
    """
    waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    cqt = librosa.cqt(waveform, sr=sample_rate, hop_length=hop_length, n_bins=n_bins, bins_per_octave=24, fmin=65)
    spec = np.abs(cqt)
    spec = np.log1p(spec)

    # Remove last frequency bin
    chunk = spec[:, 0:-2]
    spec_tensor = torch.tensor(chunk, dtype=torch.float32)
    if spec_tensor.ndim == 2:
        spec_tensor = spec_tensor.unsqueeze(0)  # Shape: (1, freq, time)
    return spec_tensor


def camelot_output(pred_camelot):
    """
    Formats the Camelot prediction:
    - Indexing as in DJ software: ID (1-12) + Mode (A=minor, B=major)
    - minor: 1-12A, major: 1-12B

    Args:
        pred_camelot (int): 0-23, neural network output

    Returns:
        (str, str): camelot_str (e.g. "6A"), key_text (from CAMELOT_MAPPING, possibly two synonyms)
    """
    idx = (pred_camelot % 12) + 1        # 1-based index for wheel
    mode = "A" if pred_camelot < 12 else "B"
    camelot_str = f"{idx}{mode}"

    # fetch key string(s) for this camelot index
    names = [k for k, v in CAMELOT_MAPPING.items() if v == pred_camelot]
    if names:
        key_text = "/".join(sorted(set(names)))
    else:
        key_text = "Unknown"
    return camelot_str, key_text


def get_audio_files(path):
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{path.suffix}'. Supported: {SUPPORTED_EXTENSIONS}")
        return [path]
    elif path.is_dir():
        files = [f for f in path.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not files:
            raise ValueError(f"No audio files found in {path}")
        return files
    else:
        raise FileNotFoundError(f"{path} is not a valid file or folder.")
