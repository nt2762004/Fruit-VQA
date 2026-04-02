from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


IMAGE_SIZE = (256, 256)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def scan_dataset(data_dir: str | Path) -> pd.DataFrame:
    data_path = resolve_path(data_dir)
    rows: list[dict[str, object]] = []

    for subset in ("train", "valid", "test"):
        subset_path = data_path / subset
        if not subset_path.exists():
            continue

        for fruit_dir in sorted(p for p in subset_path.iterdir() if p.is_dir()):
            images_dir = fruit_dir / "images"
            labels_dir = fruit_dir / "labels"
            if not images_dir.exists() or not labels_dir.exists():
                continue

            for label_path in sorted(labels_dir.glob("*.txt")):
                with label_path.open("r", encoding="utf-8") as handle:
                    count = sum(1 for line in handle if line.strip())

                image_path = images_dir / f"{label_path.stem}.jpg"
                if not image_path.exists():
                    alternate = images_dir / f"{label_path.stem}.png"
                    if alternate.exists():
                        image_path = alternate
                    else:
                        continue

                rows.append(
                    {
                        "filename": image_path.name,
                        "fruit_type": fruit_dir.name,
                        "count": int(count),
                        "subset": subset,
                        "image_path": image_path.as_posix(),
                    }
                )

    if not rows:
        raise ValueError(f"No valid samples were found under {data_path}")

    frame = pd.DataFrame(rows)
    return frame.sort_values(["subset", "fruit_type", "filename"]).reset_index(drop=True)


def save_metadata_csv(data_dir: str | Path, output_csv: str | Path) -> pd.DataFrame:
    frame = scan_dataset(data_dir)
    output_path = resolve_path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=["image_path"]).to_csv(output_path, index=False)
    return frame


def build_image_feature_vector(
    fruit_type: str,
    count: int,
    fruit_classes: list[str],
    count_scale: float = 1.0,
) -> np.ndarray:
    if count_scale <= 0:
        count_scale = 1.0

    feature = np.zeros(len(fruit_classes) + 1, dtype=np.float32)
    feature[fruit_classes.index(fruit_type)] = 1.0
    feature[-1] = float(count) / float(count_scale)
    return feature


def build_qa_pairs(fruit_type: str, count: int) -> list[tuple[str, str]]:
    return [
        ("Trong anh so luong trai cay la bao nhieu?", f"<start> anh co tong cong {count} trai cay <end>"),
        ("Anh co bao nhieu trai cay?", f"<start> anh co {count} trai cay <end>"),
        ("Co bao nhieu trai cay trong anh?", f"<start> co {count} trai cay <end>"),
        ("Anh chua loai trai cay nao?", f"<start> anh chua {fruit_type} <end>"),
        ("Trai cay trong anh la gi?", f"<start> trai cay trong anh la {fruit_type} <end>"),
        ("Loai trai cay nao trong anh?", f"<start> {fruit_type} <end>"),
        ("Anh co bao nhieu trai cay loai gi?", f"<start> anh co {count} trai {fruit_type} <end>"),
        ("Trong anh co bao nhieu trai cay va la loai nao?", f"<start> trong anh co {count} trai {fruit_type} <end>"),
    ]


def generate_qa_dataset(
    metadata_csv: str | Path,
    data_dir: str | Path,
    images_qa_dir: str | Path,
    seq2seq_dir: str | Path,
) -> dict[str, object]:
    metadata_path = resolve_path(metadata_csv)
    data_path = resolve_path(data_dir)
    images_dir = resolve_path(images_qa_dir)
    seq2seq_path = resolve_path(seq2seq_dir)

    frame = pd.read_csv(metadata_path)
    fruit_classes = sorted(frame["fruit_type"].unique().tolist())
    count_scale = max(float(frame["count"].max()), 1.0)

    images_dir.mkdir(parents=True, exist_ok=True)
    seq2seq_path.mkdir(parents=True, exist_ok=True)

    questions: list[str] = []
    answers: list[str] = []
    image_features: list[np.ndarray] = []
    image_paths: list[str] = []

    for row in frame.itertuples(index=False):
        image_path = data_path / row.subset / row.fruit_type / "images" / row.filename
        if not image_path.exists():
            continue

        feature_vector = build_image_feature_vector(row.fruit_type, int(row.count), fruit_classes, count_scale)
        qa_pairs = build_qa_pairs(row.fruit_type, int(row.count))

        json_payload = []
        for question_text, answer_text in qa_pairs:
            questions.append(question_text)
            answers.append(answer_text)
            image_features.append(feature_vector)
            image_paths.append(image_path.as_posix())
            json_payload.append(
                {
                    "image": image_path.as_posix(),
                    "question": question_text,
                    "answer": answer_text.replace("<start> ", "").replace(" <end>", ""),
                }
            )

        json_path = images_dir / f"{row.filename}.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(json_payload, handle, ensure_ascii=False, indent=2)

    if not questions:
        raise ValueError("No QA pairs were generated. Check the metadata and image paths.")

    tokenizer = Tokenizer(filters="", oov_token="<unk>")
    tokenizer.fit_on_texts(questions + answers + fruit_classes)

    question_sequences = tokenizer.texts_to_sequences(questions)
    answer_sequences = tokenizer.texts_to_sequences(answers)
    max_length = max(max(len(seq) for seq in question_sequences), max(len(seq) for seq in answer_sequences))

    question_padded = pad_sequences(question_sequences, maxlen=max_length, padding="post")
    answer_padded = pad_sequences(answer_sequences, maxlen=max_length, padding="post")

    np.save(seq2seq_path / "question_padded.npy", question_padded.astype(np.float32))
    np.save(seq2seq_path / "answer_padded.npy", answer_padded.astype(np.float32))
    np.save(seq2seq_path / "image_features.npy", np.asarray(image_features, dtype=np.float32))
    np.save(seq2seq_path / "image_paths.npy", np.asarray(image_paths))

    with (seq2seq_path / "count_scale.json").open("w", encoding="utf-8") as handle:
        json.dump({"count_scale": count_scale}, handle, indent=2)

    with (seq2seq_path / "tokenizer.json").open("w", encoding="utf-8") as handle:
        handle.write(tokenizer.to_json())

    return {
        "num_samples": len(questions),
        "num_fruit_classes": len(fruit_classes),
        "feature_dim": len(fruit_classes) + 1,
        "max_length": max_length,
        "count_scale": count_scale,
        "seq2seq_dir": seq2seq_path.as_posix(),
        "images_qa_dir": images_dir.as_posix(),
    }


def load_tokenizer(tokenizer_path: str | Path) -> Tokenizer:
    path = resolve_path(tokenizer_path)
    with path.open("r", encoding="utf-8") as handle:
        return tf.keras.preprocessing.text.tokenizer_from_json(handle.read())


def load_seq2seq_data(seq2seq_dir: str | Path) -> dict[str, object]:
    seq2seq_path = resolve_path(seq2seq_dir)
    question_padded = np.load(seq2seq_path / "question_padded.npy")
    answer_padded = np.load(seq2seq_path / "answer_padded.npy")
    image_features = np.load(seq2seq_path / "image_features.npy")
    tokenizer = load_tokenizer(seq2seq_path / "tokenizer.json")
    count_scale_path = seq2seq_path / "count_scale.json"
    count_scale = 1.0
    if count_scale_path.exists():
        with count_scale_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            count_scale = float(payload.get("count_scale", 1.0))

    return {
        "question_padded": question_padded,
        "answer_padded": answer_padded,
        "image_features": image_features,
        "tokenizer": tokenizer,
        "max_length": int(question_padded.shape[1]),
        "num_fruit_classes": int(image_features.shape[1] - 1),
        "count_scale": max(count_scale, 1.0),
    }


def split_image_features(image_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if image_features.ndim != 2 or image_features.shape[1] < 2:
        raise ValueError("image_features must be a 2D array with at least 2 columns")

    num_fruit_classes = image_features.shape[1] - 1
    return image_features[:, :num_fruit_classes], image_features[:, num_fruit_classes:num_fruit_classes + 1]
