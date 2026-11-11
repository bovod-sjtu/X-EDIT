"""Auditable channel preparation matching AuK's native mono conditioning."""

from pathlib import Path
import hashlib

import numpy as np
import soundfile as sf


def prepare_source(source: Path, directory: Path):
    samples, rate = sf.read(source, dtype="float64", always_2d=True)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("AuK input must contain finite, nonempty audio")
    channels = samples.shape[1]
    prepared = source
    if channels > 1:
        mono = samples.mean(axis=1)
        if np.any(samples) and not np.any(mono):
            raise ValueError("AuK channel mean cancels the signal; prepare a suitable source explicitly")
        prepared = directory / "mono_source.wav"
        sf.write(prepared, mono, rate, subtype="DOUBLE")
    return prepared, {
        "policy": "identity" if channels == 1 else "native_channel_mean",
        "source_channels": channels,
        "model_input_channels": 1,
        "sample_rate": rate,
        "frames": len(samples),
        "source_modified": False,
        "spatial_image_preserved": channels == 1,
        "prepared_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
    }
