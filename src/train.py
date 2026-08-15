from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "archive-3"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def find_dataset_dir(custom_path=None):
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)

    if DEFAULT_DATA_DIR.exists():
        return DEFAULT_DATA_DIR

    fallback_data = PROJECT_ROOT / "data" / "raw"
    if fallback_data.exists():
        subdirs = [d for d in fallback_data.iterdir() if d.is_dir()]
        for sd in subdirs:
            if (sd / "train").exists() and (sd / "test").exists():
                return sd

    raise FileNotFoundError(
        "Could not find valid FER dataset directory containing train/ and test/ folders."
    )


def compute_class_weights(train_dir, class_names):
    counts = []
    print("\nClass distribution in Training Set:")
    for cls in class_names:
        cls_dir = train_dir / cls
        count = sum(1 for f in cls_dir.iterdir() if f.is_file())
        counts.append(count)
        print(f"  - {cls:10s}: {count:5d} images")

    total = sum(counts)
    n_classes = len(class_names)

    # Calculate balanced weights: total / (n_classes * count)
    class_weights = {}
    for idx, count in enumerate(counts):
        class_weights[idx] = total / (n_classes * count) if count > 0 else 1.0

    print("\nCalculated Class Weights (to balance loss):")
    for idx, cls in enumerate(class_names):
        print(f"  - {cls:10s}: {class_weights[idx]:.3f}")

    return class_weights


def build_model(input_shape=(48, 48, 1), num_classes=7):
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
        ],
        name="data_augmentation",
    )

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            data_augmentation,
            layers.Rescaling(1.0 / 255.0),

            # Block 1
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.20),

            # Block 2
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Block 3
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.30),

            # Block 4
            layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.35),

            layers.GlobalAveragePooling2D(),

            # FC Dense layer
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.40),

            layers.Dense(num_classes, activation="softmax"),
        ],
        name="DeepFER_CNN",
    )

    return model


def plot_and_save_history(history, save_path):
    epochs = range(1, len(history.history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(epochs, history.history["accuracy"], "b-o", label="Train Accuracy")
    axes[0].plot(epochs, history.history["val_accuracy"], "r-o", label="Val Accuracy")
    axes[0].set_title("Training & Validation Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Loss
    axes[1].plot(epochs, history.history["loss"], "b-o", label="Train Loss")
    axes[1].plot(epochs, history.history["val_loss"], "r-o", label="Val Loss")
    axes[1].set_title("Training & Validation Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training history plot to {save_path}")


def plot_and_save_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("DeepFER Test Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DeepFER Emotion Recognition Model")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to FER dataset root")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--use-class-weights", action="store_true", default=True, help="Apply class weighting")
    args = parser.parse_args()

    data_dir = find_dataset_dir(args.data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    print(f"Dataset location: {data_dir}")
    image_size = (48, 48)

    train_dataset = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.20,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=args.batch_size,
        color_mode="grayscale",
        label_mode="categorical",
    )

    val_dataset = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.20,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=args.batch_size,
        color_mode="grayscale",
        label_mode="categorical",
    )

    test_dataset = keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        color_mode="grayscale",
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)

    class_weights = compute_class_weights(train_dir, class_names) if args.use_class_weights else None

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(AUTOTUNE)
    val_dataset = val_dataset.prefetch(AUTOTUNE)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    model = build_model(input_shape=(48, 48, 1), num_classes=num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    checkpoint_path = MODELS_DIR / "deepfer_best.keras"
    final_model_path = MODELS_DIR / "deepfer_final.keras"
    log_path = RESULTS_DIR / "training_log.csv"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(log_path),
    ]

    print("\nStarting Training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    plot_and_save_history(history, RESULTS_DIR / "training_history.png")

    print("\nEvaluating model on Test Set...")
    best_model = keras.models.load_model(checkpoint_path)

    y_true = []
    y_pred = []

    for images, labels in test_dataset:
        preds = best_model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:\n")
    print(report)

    with open(RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plot_and_save_confusion_matrix(cm, class_names, RESULTS_DIR / "confusion_matrix.png")

    print("Training and evaluation complete!")


if __name__ == "__main__":
    main()
