from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .evaluate import save_training_history
from .model import build_classifier_model, build_regression_model, build_vqa_model
from .preprocess import IMAGE_SIZE, load_seq2seq_data, resolve_path, split_image_features


def _configure_gpu_runtime() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected by TensorFlow. Training will run on CPU.")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:  # pragma: no cover - depends on local runtime state
            print(f"Could not enable memory growth for {gpu.name}: {exc}")

    print(f"TensorFlow GPUs: {[gpu.name for gpu in gpus]}")


def _make_training_callbacks() -> list[tf.keras.callbacks.Callback]:
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def _image_to_tensor(image_path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(label, tf.float32)


def _build_regression_dataset(
    frame: pd.DataFrame,
    data_dir: str | Path,
    batch_size: int,
    count_scale: float,
) -> tf.data.Dataset:
    data_path = resolve_path(data_dir)

    image_paths: list[str] = []
    labels: list[float] = []
    for row in frame.itertuples(index=False):
        image_path = data_path / row.subset / row.fruit_type / "images" / row.filename
        if image_path.exists():
            image_paths.append(image_path.as_posix())
            labels.append(float(row.count) / count_scale)

    if not image_paths:
        raise ValueError("No valid regression samples were found.")

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(_image_to_tensor, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(256).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_classifier(data_dir: str | Path, output_model: str | Path, figures_dir: str | Path, batch_size: int = 16, epochs: int = 20):
    _configure_gpu_runtime()
    data_path = resolve_path(data_dir)
    train_dir = data_path / "train"
    valid_dir = data_path / "valid"

    tf.keras.backend.clear_session()

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir.as_posix(),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
    )
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir.as_posix(),
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
    )

    model = build_classifier_model(num_classes=train_generator.num_classes)
    class_counts = train_generator.classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(class_counts),
        y=class_counts,
    )
    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(np.unique(class_counts), class_weights)}

    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=_make_training_callbacks(),
        verbose=1,
    )

    output_path = resolve_path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    save_training_history(history, Path(figures_dir) / "classifier_training.png", "Classifier training")
    tf.keras.backend.clear_session()
    gc.collect()
    return model, history, train_generator.class_indices


def train_regression(metadata_csv: str | Path, data_dir: str | Path, output_model: str | Path, figures_dir: str | Path, batch_size: int = 16, epochs: int = 20):
    _configure_gpu_runtime()
    frame = pd.read_csv(resolve_path(metadata_csv))
    train_frame = frame[frame["subset"] == "train"]
    valid_frame = frame[frame["subset"] == "valid"]
    count_scale = max(float(frame["count"].max()), 1.0)
    print(f"Normalizing regression targets by max count={count_scale:.0f}")

    tf.keras.backend.clear_session()
    train_dataset = _build_regression_dataset(train_frame, data_dir, batch_size, count_scale)
    valid_dataset = _build_regression_dataset(valid_frame, data_dir, batch_size, count_scale)

    model = build_regression_model()
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=_make_training_callbacks(),
        verbose=1,
    )

    output_path = resolve_path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    scale_path = output_path.with_suffix(".scale.json")
    with scale_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "normalization": "count_divided_by_max_count",
                "max_count": count_scale,
            },
            handle,
            indent=2,
        )
    save_training_history(history, Path(figures_dir) / "regression_training.png", "Regression training")
    tf.keras.backend.clear_session()
    gc.collect()
    return model, history


def train_vqa(seq2seq_dir: str | Path, output_model: str | Path, figures_dir: str | Path, use_attention: bool = False, batch_size: int = 64, epochs: int = 20):
    _configure_gpu_runtime()
    data = load_seq2seq_data(seq2seq_dir)
    question_padded = data["question_padded"]
    answer_padded = data["answer_padded"]
    image_features = data["image_features"]
    tokenizer = data["tokenizer"]
    max_length = data["max_length"]
    num_fruit_classes = data["num_fruit_classes"]

    train_qs, val_qs, train_as, val_as, train_imgs, val_imgs = train_test_split(
        question_padded,
        answer_padded,
        image_features,
        test_size=0.2,
        random_state=42,
    )

    train_fruit_classifier, train_fruit_regression = split_image_features(train_imgs)
    val_fruit_classifier, val_fruit_regression = split_image_features(val_imgs)

    decoder_input_data = train_as[:, :-1]
    decoder_target_data = train_as[:, 1:]
    val_decoder_input_data = val_as[:, :-1]
    val_decoder_target_data = val_as[:, 1:]

    model = build_vqa_model(
        vocab_size=len(tokenizer.word_index) + 1,
        max_length=max_length,
        num_fruit_classes=num_fruit_classes,
        use_attention=use_attention,
    )

    history = model.fit(
        [train_qs, train_fruit_classifier, train_fruit_regression, decoder_input_data],
        np.expand_dims(decoder_target_data, -1),
        validation_data=(
            [val_qs, val_fruit_classifier, val_fruit_regression, val_decoder_input_data],
            np.expand_dims(val_decoder_target_data, -1),
        ),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=_make_training_callbacks(),
        verbose=1,
    )

    output_path = resolve_path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    figure_name = f"{output_path.stem}_training.png"
    save_training_history(history, Path(figures_dir) / figure_name, "VQA training")
    return model, history
