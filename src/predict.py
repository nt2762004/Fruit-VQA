from __future__ import annotations

import json
import re
import unicodedata
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from .preprocess import IMAGE_SIZE, build_image_feature_vector, load_seq2seq_data, resolve_path, scan_dataset


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    normalized = "".join(character for character in normalized if unicodedata.category(character) != "Mn")
    normalized = re.sub(r"[^a-z0-9<>\s]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def load_image_lookup(data_dir: str | Path) -> pd.DataFrame:
    frame = scan_dataset(data_dir)
    frame["resolved_path"] = frame["image_path"].map(lambda value: Path(value).resolve().as_posix())
    frame["filename_lower"] = frame["filename"].str.lower()
    return frame


def resolve_label_from_dataset(image_path: str | Path, data_dir: str | Path) -> tuple[str, int]:
    lookup = load_image_lookup(data_dir)
    selected_path = Path(image_path).resolve().as_posix()
    selected_name = Path(image_path).name.lower()

    match = lookup[(lookup["resolved_path"] == selected_path) | (lookup["filename_lower"] == selected_name)]
    if match.empty:
        raise ValueError(
            f"Could not find image '{image_path}' in the dataset. Provide a dataset image path or train classifier/regression models."
        )

    row = match.iloc[0]
    return str(row["fruit_type"]), int(row["count"])


def prepare_question_input(question: str, tokenizer, max_length: int) -> np.ndarray:
    normalized_question = normalize_text(question)
    question_sequence = tokenizer.texts_to_sequences([normalized_question])
    return tf.keras.preprocessing.sequence.pad_sequences(question_sequence, maxlen=max_length, padding="post")


def load_image_tensor(image_source: bytes | str | Path) -> np.ndarray:
    if isinstance(image_source, (str, Path)):
        with Image.open(image_source) as image:
            image_array = np.asarray(image.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32) / 255.0
    else:
        with Image.open(BytesIO(image_source)) as image:
            image_array = np.asarray(image.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32) / 255.0

    return image_array[None, ...]


def load_regression_scale(regression_model_path: str | Path) -> float:
    scale_path = resolve_path(regression_model_path).with_suffix(".scale.json")
    if not scale_path.exists():
        return 1.0

    with scale_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return max(float(payload.get("max_count", 1.0)), 1.0)


def predict_fruit_attributes(
    image_source: bytes | str | Path,
    classifier_model_path: str | Path,
    regression_model_path: str | Path,
    data_dir: str | Path,
) -> dict[str, object]:
    fruit_classes = sorted(scan_dataset(data_dir)["fruit_type"].unique().tolist())
    classifier_model = tf.keras.models.load_model(resolve_path(classifier_model_path), compile=False)
    regression_model = tf.keras.models.load_model(resolve_path(regression_model_path), compile=False)
    count_scale = load_regression_scale(regression_model_path)

    return predict_fruit_attributes_from_models(
        image_source,
        classifier_model,
        regression_model,
        fruit_classes,
        count_scale,
    )


def predict_fruit_attributes_from_models(
    image_source: bytes | str | Path,
    classifier_model: tf.keras.Model,
    regression_model: tf.keras.Model,
    fruit_classes: list[str],
    count_scale: float,
) -> dict[str, object]:
    count_scale = max(float(count_scale), 1.0)

    image_batch = load_image_tensor(image_source)
    class_probs = classifier_model.predict(image_batch, verbose=0)[0]
    predicted_class_index = int(np.argmax(class_probs))
    predicted_fruit_type = fruit_classes[predicted_class_index]
    class_confidence = float(class_probs[predicted_class_index])

    predicted_count_norm = float(regression_model.predict(image_batch, verbose=0)[0][0])
    predicted_count = max(0, int(round(predicted_count_norm * count_scale)))

    return {
        "fruit_type": predicted_fruit_type,
        "count": predicted_count,
        "count_raw": predicted_count_norm,
        "count_scale": count_scale,
        "class_confidence": class_confidence,
        "feature_dim": len(fruit_classes) + 1,
        "fruit_classes": fruit_classes,
    }


def predict_answer(
    model: tf.keras.Model,
    question: str,
    image_feature_vector: np.ndarray,
    tokenizer,
    max_length: int,
) -> str:
    question_input = prepare_question_input(question, tokenizer, max_length)

    decoder_length = max_length - 1
    decoder_input = np.zeros((1, decoder_length), dtype=np.int32)

    start_token = tokenizer.word_index.get("<start>")
    end_token = tokenizer.word_index.get("<end>")
    if start_token is None or end_token is None:
        raise ValueError("Tokenizer is missing <start> or <end> tokens.")

    decoder_input[0, 0] = start_token
    predicted_ids: list[int] = []

    for time_step in range(decoder_length):
        predictions = model.predict(
            [question_input, image_feature_vector[None, :-1], image_feature_vector[None, -1:], decoder_input],
            verbose=0,
        )
        predicted_id = int(np.argmax(predictions[0, time_step, :]))
        if predicted_id == end_token:
            break
        predicted_ids.append(predicted_id)
        if time_step < decoder_length - 1:
            decoder_input[0, time_step + 1] = predicted_id

    predicted_tokens = [tokenizer.index_word.get(index, "") for index in predicted_ids]
    answer = " ".join(token for token in predicted_tokens if token and token not in {"<start>", "<end>"}).strip()
    return answer


def run_single_prediction(
    image_path: str | Path,
    question: str,
    model_path: str | Path,
    seq2seq_dir: str | Path,
    data_dir: str | Path,
) -> dict[str, object]:
    data = load_seq2seq_data(seq2seq_dir)
    tokenizer = data["tokenizer"]
    max_length = data["max_length"]
    image_features = data["image_features"]
    count_scale = float(data.get("count_scale", 1.0))
    fruit_classes = sorted(scan_dataset(data_dir)["fruit_type"].unique().tolist())

    model = tf.keras.models.load_model(resolve_path(model_path), compile=False)
    fruit_type, count = resolve_label_from_dataset(image_path, data_dir)
    feature_vector = build_image_feature_vector(fruit_type, count, fruit_classes, count_scale)
    answer = predict_answer(model, question, feature_vector, tokenizer, max_length)

    return {
        "image_path": Path(image_path).resolve().as_posix(),
        "question": question,
        "normalized_question": normalize_text(question),
        "fruit_type": fruit_type,
        "count": count,
        "answer": answer,
        "feature_dim": int(feature_vector.shape[0]),
        "num_samples": int(image_features.shape[0]),
    }