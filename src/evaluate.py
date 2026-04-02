from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split

from .preprocess import load_seq2seq_data, resolve_path, split_image_features


def save_training_history(history, output_path: str | Path, title: str) -> None:
    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metrics = history.history
    epochs = range(1, len(metrics.get("loss", [])) + 1)

    plt.figure(figsize=(10, 4))
    if "loss" in metrics:
        plt.plot(epochs, metrics["loss"], label="loss")
    if "val_loss" in metrics:
        plt.plot(epochs, metrics["val_loss"], label="val_loss")
    if "accuracy" in metrics:
        plt.plot(epochs, metrics["accuracy"], label="accuracy")
    if "val_accuracy" in metrics:
        plt.plot(epochs, metrics["val_accuracy"], label="val_accuracy")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()

    history_frame = pd.DataFrame(metrics)
    history_frame.to_csv(output.with_suffix(".csv"), index=False)

    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def evaluate_vqa_model(model, test_questions, test_answers, test_fruit_classifier, test_fruit_regression, tokenizer, decoder_seq_length: int):
    bleu_scores: list[float] = []
    rouge_scores: list[float] = []
    rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    smoothing = SmoothingFunction().method1

    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    start_token = tokenizer.word_index["<start>"]
    end_token = tokenizer.word_index["<end>"]

    for i in range(len(test_questions)):
        question_padded = test_questions[i : i + 1]
        fruit_classifier_input = test_fruit_classifier[i : i + 1]
        fruit_regression_input = test_fruit_regression[i : i + 1]
        decoder_input = np.zeros((1, decoder_seq_length))
        decoder_input[0, 0] = start_token

        output_sentence: list[int] = []
        for t in range(decoder_seq_length):
            predictions = model.predict(
                [question_padded, fruit_classifier_input, fruit_regression_input, decoder_input],
                verbose=0,
            )
            predicted_id = int(np.argmax(predictions[0, t, :]))
            if predicted_id == end_token:
                break
            output_sentence.append(predicted_id)
            if t < decoder_seq_length - 1:
                decoder_input[0, t + 1] = predicted_id

        predicted_text = [reverse_word_index[idx] for idx in output_sentence if idx in reverse_word_index]
        ground_truth = [reverse_word_index[idx] for idx in test_answers[i] if idx in reverse_word_index and idx != end_token]

        bleu_scores.append(sentence_bleu([ground_truth], predicted_text, smoothing_function=smoothing))
        rouge_scores.append(rouge_scorer_obj.score(" ".join(ground_truth), " ".join(predicted_text))["rouge1"].fmeasure)

    return float(np.mean(bleu_scores)), float(np.mean(rouge_scores))


def measure_inference_time(model, inputs, iterations: int = 100) -> float:
    start_time = time.time()
    for _ in range(iterations):
        model.predict(inputs, verbose=0)
    total_time = time.time() - start_time
    return total_time / iterations


def evaluate_and_plot(seq2seq_dir: str | Path, attention_model_path: str | Path, no_attention_model_path: str | Path, output_figure_path: str | Path):
    data = load_seq2seq_data(seq2seq_dir)
    question_padded = data["question_padded"]
    answer_padded = data["answer_padded"]
    image_features = data["image_features"]
    tokenizer = data["tokenizer"]
    decoder_seq_length = question_padded.shape[1] - 1

    _, test_qs, _, test_as, _, test_imgs = train_test_split(
        question_padded,
        answer_padded,
        image_features,
        test_size=0.2,
        random_state=42,
    )
    test_fruit_classifier, test_fruit_regression = split_image_features(test_imgs)

    attention_model = tf.keras.models.load_model(resolve_path(attention_model_path))
    no_attention_model = tf.keras.models.load_model(resolve_path(no_attention_model_path))

    bleu_att, rouge_att = evaluate_vqa_model(
        attention_model,
        test_qs,
        test_as,
        test_fruit_classifier,
        test_fruit_regression,
        tokenizer,
        decoder_seq_length,
    )
    bleu_no_att, rouge_no_att = evaluate_vqa_model(
        no_attention_model,
        test_qs,
        test_as,
        test_fruit_classifier,
        test_fruit_regression,
        tokenizer,
        decoder_seq_length,
    )

    sample_inputs = [test_qs[0:1], test_fruit_classifier[0:1], test_fruit_regression[0:1], np.zeros((1, decoder_seq_length))]
    time_att = measure_inference_time(attention_model, sample_inputs)
    time_no_att = measure_inference_time(no_attention_model, sample_inputs)

    metrics = {
        "Model": ["Attention", "No Attention"],
        "BLEU": [bleu_att, bleu_no_att],
        "ROUGE-1": [rouge_att, rouge_no_att],
        "Time (s)": [time_att, time_no_att],
    }

    output_path = resolve_path(output_figure_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(metrics["Model"], metrics["BLEU"], color=["#2E86AB", "#F18F01"])
    axes[0].set_title("BLEU Score")
    axes[0].set_ylim(0, 1)

    axes[1].bar(metrics["Model"], metrics["ROUGE-1"], color=["#2E86AB", "#F18F01"])
    axes[1].set_title("ROUGE-1 Score")
    axes[1].set_ylim(0, 1)

    axes[2].bar(metrics["Model"], metrics["Time (s)"], color=["#2E86AB", "#F18F01"])
    axes[2].set_title("Prediction Time")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output_path.with_suffix(".csv"), index=False)

    with output_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics
