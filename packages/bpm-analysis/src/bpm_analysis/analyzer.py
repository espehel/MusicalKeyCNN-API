from pathlib import Path
import numpy as np
import essentia.standard as es


def analyze_bpm(audio_path) -> dict:
    """
    Analyzes the BPM of an audio file using Essentia's RhythmExtractor2013.

    Uses the 'multifeature' method which combines multiple beat tracking algorithms
    for robust tempo estimation across genres.

    Args:
        audio_path (str or Path): Path to audio file (.mp3, .wav, .flac, .ogg).

    Returns:
        dict with keys:
            bpm (float): Estimated tempo in beats per minute.
            confidence (float): Detection confidence score.
            beat_count (int): Number of detected beats.
            beats (list[float]): Beat positions in seconds.
    """
    loader = es.MonoLoader(filename=str(audio_path), sampleRate=44100)
    audio = loader()

    extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, ticks, confidence, _, tick_intervals = extractor(audio)

    return {
        "bpm": round(float(bpm), 2),
        "confidence": round(float(confidence), 4),
        "beat_count": len(ticks),
        "beats": [round(float(t), 4) for t in ticks],
    }
