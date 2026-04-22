"""
wildnode_ai/audio_detection/preprocess.py
=========================================
PURPOSE: Convert raw audio (.wav) files into Mel Spectrogram images (2D arrays)
         that can be fed into a CNN model for classification.

HOW IT WORKS:
  1. Reads a .wav audio file using Librosa
  2. Computes a Mel Spectrogram (a visual representation of sound frequency over time)
  3. Saves the result as a .npy (NumPy) array for training

BEGINNER NOTE:
  Think of a Mel Spectrogram as a 'fingerprint' of the sound.
  Each animal (elephant, tiger) has a unique fingerprint pattern.
  Our CNN learns to recognize these patterns.
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# ─── Configuration ───────────────────────────────────────────────────────────
SAMPLE_RATE    = 22050   # Standard audio sample rate (22,050 samples/second)
DURATION       = 5       # Clip duration in seconds (we use 5-second chunks)
N_MELS         = 128     # Number of Mel frequency bands (height of spectrogram)
HOP_LENGTH     = 512     # Samples between frames (controls time resolution)
N_FFT          = 2048    # FFT window size (controls frequency resolution)

# Exported shape constant (imported by train.py and model.py)
INPUT_SHAPE_CONST = (128, 216, 1)
# ─────────────────────────────────────────────────────────────────────────────


def load_audio(file_path: str, duration: int = DURATION) -> np.ndarray:
    """
    Load an audio file and return a fixed-length waveform array.

    Args:
        file_path : Path to the .wav audio file
        duration  : How many seconds to load (clips or pads to this length)

    Returns:
        y : 1D NumPy array of audio samples (float32)
    """
    try:
        # Load audio; mono=True converts stereo to mono
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=duration)

        # Ensure exact length: pad with zeros if shorter, trim if longer
        target_length = SAMPLE_RATE * duration
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))   # Pad with silence
        else:
            y = y[:target_length]                         # Trim excess

        return y

    except Exception as e:
        print(f"[ERROR] Could not load audio file '{file_path}': {e}")
        return np.zeros(SAMPLE_RATE * duration, dtype=np.float32)


def audio_to_melspectrogram(y: np.ndarray) -> np.ndarray:
    """
    Convert a raw audio waveform into a 2D Mel Spectrogram.

    Args:
        y : 1D NumPy array of audio samples

    Returns:
        mel_db : 2D NumPy array of shape (N_MELS, time_steps) in dB scale
    """
    # Compute Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )

    # Convert amplitude to decibels (log scale) — easier for CNN to learn
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to range [0, 1] for neural network input
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

    return mel_db.astype(np.float32)


def wav_to_spectrogram_array(file_path: str) -> np.ndarray:
    """
    Full pipeline: .wav file → 2D Mel Spectrogram array.
    This is the main function used by the prediction module.

    Args:
        file_path : Path to the .wav audio file

    Returns:
        mel_db : 2D NumPy array (128, time_steps)
    """
    y = load_audio(file_path)
    return audio_to_melspectrogram(y)


def save_spectrogram_image(mel_db: np.ndarray, save_path: str):
    """
    Save the spectrogram as a PNG image (for visualization only).

    Args:
        mel_db    : 2D Mel Spectrogram array
        save_path : Where to save the image
    """
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(mel_db, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                              x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"[✓] Spectrogram image saved: {save_path}")


def generate_dummy_dataset(output_dir: str, n_samples_per_class: int = 20):
    """
    Generate a DUMMY training dataset for demo purposes.

    WHAT IT DOES:
    - Creates synthetic audio signals that mimic each animal's frequency profile
    - Elephant: low-frequency rumble (~80 Hz)
    - Tiger: mid-frequency growl (~300 Hz)
    - Background: random noise

    In a real project, replace this with recorded wildlife audio from:
    - https://www.xeno-canto.org/  (bird/wildlife sounds)
    - https://freesound.org/       (general sound library)
    - https://www.tierstimmenarchiv.de/ (animal archive)

    Args:
        output_dir          : Root folder to save .npy spectrogram files
        n_samples_per_class : Number of samples to generate per class
    """
    print("[INFO] Generating dummy training dataset...")

    classes = {
        "elephant"  : {"freq": 80,   "noise": 0.05},   # Low rumble
        "tiger"     : {"freq": 300,  "noise": 0.10},   # Mid growl
        "background": {"freq": None, "noise": 0.30},   # Random noise
    }

    t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION)   # Time axis

    for class_name, params in classes.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(n_samples_per_class):
            # Generate synthetic waveform
            if params["freq"] is not None:
                # Sinusoidal wave at animal's characteristic frequency
                y = np.sin(2 * np.pi * params["freq"] * t).astype(np.float32)
                # Add harmonics for realism
                y += 0.5 * np.sin(2 * np.pi * params["freq"] * 2 * t).astype(np.float32)
                y += 0.3 * np.sin(2 * np.pi * params["freq"] * 3 * t).astype(np.float32)
            else:
                y = np.zeros_like(t, dtype=np.float32)

            # Add random noise (so each sample is slightly different)
            y += np.random.normal(0, params["noise"], size=len(t)).astype(np.float32)

            # Clip to valid audio range [-1, 1]
            y = np.clip(y, -1.0, 1.0)

            # Convert to mel spectrogram
            mel = audio_to_melspectrogram(y)

            # Save as .npy file
            save_path = os.path.join(class_dir, f"{class_name}_{i:03d}.npy")
            np.save(save_path, mel)

        print(f"  [✓] Generated {n_samples_per_class} samples for class: '{class_name}'")

    print(f"\n[✓] Dummy dataset created at: {output_dir}")
    print(f"    Classes: {list(classes.keys())}")
    print(f"    Samples per class: {n_samples_per_class}")


if __name__ == "__main__":
    # Run this script directly to generate a dummy training dataset
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    generate_dummy_dataset(dataset_dir, n_samples_per_class=30)
