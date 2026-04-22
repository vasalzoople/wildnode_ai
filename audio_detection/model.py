"""
wildnode_ai/audio_detection/model.py
=====================================
PURPOSE: Define the CNN (Convolutional Neural Network) architecture for
         classifying wildlife audio into 3 categories:
           0 → Elephant
           1 → Tiger
           2 → Background (noise/silence)

BEGINNER NOTE:
  A CNN is like a smart image analyzer.
  It looks at the spectrogram image and finds patterns (edges, textures)
  that correspond to each animal's unique sound signature.

ARCHITECTURE:
  Input (128×216×1)
    → Conv2D + BatchNorm + ReLU + MaxPool   # Feature extraction block 1
    → Conv2D + BatchNorm + ReLU + MaxPool   # Feature extraction block 2
    → Conv2D + BatchNorm + ReLU + MaxPool   # Feature extraction block 3
    → GlobalAveragePooling                  # Flatten features
    → Dense(128) + Dropout                  # Fully connected layer
    → Dense(3) + Softmax                    # Output: probability for each class
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np

# ─── Label Mapping ───────────────────────────────────────────────────────────
CLASS_NAMES = ["elephant", "tiger", "background"]
NUM_CLASSES = len(CLASS_NAMES)

# ─── Input Shape ─────────────────────────────────────────────────────────────
# Spectrogram dimensions: 128 Mel bands × ~216 time steps × 1 channel (grayscale)
INPUT_SHAPE = (128, 216, 1)
# ─────────────────────────────────────────────────────────────────────────────


def build_audio_cnn(input_shape: tuple = INPUT_SHAPE,
                    num_classes: int = NUM_CLASSES,
                    learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Build and compile the WildNode Audio CNN model.

    Args:
        input_shape   : (height, width, channels) of the spectrogram
        num_classes   : Number of animal categories to classify
        learning_rate : Adam optimizer learning rate

    Returns:
        model : Compiled Keras model ready for training
    """
    model = models.Sequential(name="WildNode_AudioCNN")

    # ── Block 1: Detect low-level frequency features ──────────────────────
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same',
                             kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Block 2: Detect mid-level patterns ────────────────────────────────
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                             kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Block 3: Detect high-level animal-specific features ───────────────
    model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                             kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Flatten & Classify ────────────────────────────────────────────────
    model.add(layers.GlobalAveragePooling2D())   # Reduces each feature map to one value
    model.add(layers.Dense(128, activation='relu',
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Probability for each class

    # ── Compile ───────────────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',   # Works with integer labels
        metrics=['accuracy']
    )

    return model


def get_model_summary(model: tf.keras.Model) -> str:
    """Return model architecture as a formatted string."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test: build model and print summary
    model = build_audio_cnn()
    print(get_model_summary(model))
    print(f"\n✅ Model built successfully!")
    print(f"   Input shape : {INPUT_SHAPE}")
    print(f"   Classes     : {CLASS_NAMES}")
    print(f"   Parameters  : {model.count_params():,}")
