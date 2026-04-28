"""
wildnode_ai/audio_detection/train.py
======================================
PURPOSE: Train the WildNode Audio CNN on spectrogram data.

FLOW:
  1. Generate (or load) dummy spectrogram .npy files
  2. Load all spectrograms into memory with their labels
  3. Split into Train (80%) / Validation (20%)
  4. Train for N epochs, with callbacks for early stopping
  5. Save trained model as 'audio_model.h5' (or .keras)
  6. Print training report and accuracy graph

HOW TO RUN:
  python audio_detection/train.py

OUTPUT:
  audio_detection/audio_model.keras   ← Saved model weights
  audio_detection/training_plot.png   ← Accuracy/Loss curves
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Adjust path so we can import from siblings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_detection.preprocess import generate_dummy_dataset, INPUT_SHAPE_CONST
from audio_detection.model import build_audio_cnn, CLASS_NAMES

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH  = os.path.join(BASE_DIR, "audio_model.keras")
PLOT_PATH   = os.path.join(BASE_DIR, "training_plot.png")

EPOCHS      = 30
BATCH_SIZE  = 16
TEST_SIZE   = 0.2
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────


def load_dataset(dataset_dir: str):
    """
    Scan the dataset folder structure and load all .npy spectrograms with labels.

    Expected folder structure:
        dataset/
            elephant/   ← class 0
            tiger/      ← class 1
            background/ ← class 2

    Returns:
        X : NumPy array of spectrograms (N, 128, 216, 1)
        y : NumPy array of integer labels (N,)
    """
    X, y = [], []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(class_dir):
            print(f"[WARN] Missing class folder: {class_dir}")
            continue

        npy_files = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
        print(f"  Loading class '{class_name}': {len(npy_files)} samples...")

        for npy_file in npy_files:
            mel = np.load(os.path.join(class_dir, npy_file))   # Shape: (128, T)

            # Ensure consistent width (pad/trim time axis to 216 columns)
            target_w = 216
            if mel.shape[1] < target_w:
                pad_w = target_w - mel.shape[1]
                mel = np.pad(mel, ((0, 0), (0, pad_w)))
            else:
                mel = mel[:, :target_w]

            # Add channel dimension: (128, 216) → (128, 216, 1)
            mel = mel[:, :, np.newaxis]

            X.append(mel)
            y.append(label_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def plot_training_history(history: tf.keras.callbacks.History, save_path: str):
    """
    Plot and save accuracy + loss curves from training history.

    Args:
        history   : Keras training history object
        save_path : Where to save the PNG plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0d1117')

    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

    # Accuracy
    axes[0].plot(history.history['accuracy'], color='#58a6ff', linewidth=2, label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], color='#3fb950', linewidth=2, label='Val Acc')
    axes[0].set_title('Model Accuracy', color='white', fontsize=13)
    axes[0].set_xlabel('Epoch', color='white')
    axes[0].set_ylabel('Accuracy', color='white')
    axes[0].legend(facecolor='#21262d', labelcolor='white')

    # Loss
    axes[1].plot(history.history['loss'], color='#f85149', linewidth=2, label='Train Loss')
    axes[1].plot(history.history['val_loss'], color='#d29922', linewidth=2, label='Val Loss')
    axes[1].set_title('Model Loss', color='white', fontsize=13)
    axes[1].set_xlabel('Epoch', color='white')
    axes[1].set_ylabel('Loss', color='white')
    axes[1].legend(facecolor='#21262d', labelcolor='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Training plot saved: {save_path}")


def train():
    print("=" * 60)
    print("  WildNode AI – Audio Detection CNN Training")
    print("=" * 60)

    # ── Step 1: Generate dummy dataset if it doesn't exist ────────────────
    if not os.path.isdir(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
        print("\n[INFO] No dataset found. Generating dummy dataset...")
        generate_dummy_dataset(DATASET_DIR, n_samples_per_class=40)
    else:
        print(f"\n[INFO] Found existing dataset at: {DATASET_DIR}")

    # ── Step 2: Load dataset ──────────────────────────────────────────────
    print("\n[INFO] Loading spectrograms...")
    X, y = load_dataset(DATASET_DIR)
    print(f"  Dataset loaded: {X.shape[0]} samples | Shape per sample: {X.shape[1:]}")
    print(f"  Class distribution: {dict(zip(CLASS_NAMES, np.bincount(y)))}")

    # ── Step 3: Train / Validation split ─────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\n  Train samples: {len(X_train)} | Val samples: {len(X_val)}")

    # ── Step 4: Build model ───────────────────────────────────────────────
    print("\n[INFO] Building CNN model...")
    model = build_audio_cnn()
    print(f"  Total parameters: {model.count_params():,}")

    # ── Step 5: Define training callbacks ─────────────────────────────────
    callbacks = [
        # Stop early if validation loss stops improving
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=8, restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when training stalls
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6,
            verbose=1
        ),
        # Save the best model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
    ]

    # ── Step 6: Train ─────────────────────────────────────────────────────
    print(f"\n[INFO] Training for up to {EPOCHS} epochs (batch size={BATCH_SIZE})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # ── Step 7: Final evaluation ───────────────────────────────────────────
    print("\n[INFO] Evaluating on validation set...")
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n{'=' * 60}")
    print(f"  [OK] Training Complete!")
    print(f"  Final Validation Accuracy : {acc * 100:.2f}%")
    print(f"  Final Validation Loss     : {loss:.4f}")
    print(f"  Model saved to            : {MODEL_PATH}")
    print(f"{'=' * 60}")

    # ── Step 8: Save training plot ─────────────────────────────────────────
    plot_training_history(history, PLOT_PATH)


if __name__ == "__main__":
    train()
