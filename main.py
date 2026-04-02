from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluate import evaluate_and_plot
from src.preprocess import generate_qa_dataset, save_metadata_csv
from src.predict import run_single_prediction
from src.train import train_classifier, train_regression, train_vqa


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fruit VQA project entry point")
    subparsers = parser.add_subparsers(dest="command")

    preprocess = subparsers.add_parser("preprocess", help="Build metadata and QA artifacts")
    preprocess.add_argument("--data-dir", default="data")
    preprocess.add_argument("--metadata-csv", default="data/FruitDataFrame.csv")
    preprocess.add_argument("--images-qa-dir", default="Images_QA")
    preprocess.add_argument("--seq2seq-dir", default="seq2seqData")

    train = subparsers.add_parser("train", help="Train a model")
    train.add_argument("task", choices=["vqa-no-attention", "vqa-attention"], help="Task to train")
    train.add_argument("--seq2seq-dir", default="seq2seqData")
    train.add_argument("--figures-dir", default="reports/figures")
    train.add_argument("--output-model", default=None)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=64)

    train_classifier_cmd = subparsers.add_parser("train-classifier", help="Train the fruit classification model")
    train_classifier_cmd.add_argument("--data-dir", default="data")
    train_classifier_cmd.add_argument("--figures-dir", default="reports/figures")
    train_classifier_cmd.add_argument("--output-model", default="fruit_classifier.keras")
    train_classifier_cmd.add_argument("--epochs", type=int, default=20)
    train_classifier_cmd.add_argument("--batch-size", type=int, default=16)

    train_regression_cmd = subparsers.add_parser("train-regression", help="Train the fruit count regression model")
    train_regression_cmd.add_argument("--metadata-csv", default="data/FruitDataFrame.csv")
    train_regression_cmd.add_argument("--data-dir", default="data")
    train_regression_cmd.add_argument("--figures-dir", default="reports/figures")
    train_regression_cmd.add_argument("--output-model", default="fruit_regression.keras")
    train_regression_cmd.add_argument("--epochs", type=int, default=20)
    train_regression_cmd.add_argument("--batch-size", type=int, default=16)

    train_all_cmd = subparsers.add_parser("train-all", help="Train the classifier and regression models in sequence")
    train_all_cmd.add_argument("--data-dir", default="data")
    train_all_cmd.add_argument("--metadata-csv", default="data/FruitDataFrame.csv")
    train_all_cmd.add_argument("--figures-dir", default="reports/figures")
    train_all_cmd.add_argument("--classifier-model", default="fruit_classifier.keras")
    train_all_cmd.add_argument("--regression-model", default="fruit_regression.keras")
    train_all_cmd.add_argument("--epochs", type=int, default=20)
    train_all_cmd.add_argument("--batch-size", type=int, default=16)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate the trained VQA models")
    evaluate.add_argument("--seq2seq-dir", default="seq2seqData")
    evaluate.add_argument("--attention-model", default="seq2seq_with_attention.keras")
    evaluate.add_argument("--no-attention-model", default="seq2seq_no_attention.keras")
    evaluate.add_argument("--figure-path", default="reports/figures/vqa_evaluation.png")

    predict = subparsers.add_parser("predict", help="Predict an answer for a single image and question")
    predict.add_argument("--image", default=None, help="Path to an image inside the dataset")
    predict.add_argument("--question", default=None, help="Question to ask about the image")
    predict.add_argument("--data-dir", default="data")
    predict.add_argument("--seq2seq-dir", default="seq2seqData")
    predict.add_argument("--model", default="seq2seq_attention_full.keras")

    all_cmd = subparsers.add_parser("all", help="Run preprocessing, training, and evaluation in order")
    all_cmd.add_argument("--data-dir", default="data")
    all_cmd.add_argument("--metadata-csv", default="data/FruitDataFrame.csv")
    all_cmd.add_argument("--images-qa-dir", default="Images_QA")
    all_cmd.add_argument("--seq2seq-dir", default="seq2seqData")
    all_cmd.add_argument("--figures-dir", default="reports/figures")
    all_cmd.add_argument("--epochs", type=int, default=20)
    all_cmd.add_argument("--batch-size", type=int, default=64)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        metadata = save_metadata_csv(args.data_dir, args.metadata_csv)
        stats = generate_qa_dataset(args.metadata_csv, args.data_dir, args.images_qa_dir, args.seq2seq_dir)
        print(f"Metadata rows: {len(metadata)}")
        print(stats)
        return

    if args.command == "train":
        output_model = args.output_model
        if args.task == "vqa-no-attention":
            output_model = output_model or "seq2seq_no_attention.keras"
            train_vqa(args.seq2seq_dir, output_model, args.figures_dir, use_attention=False, batch_size=args.batch_size, epochs=args.epochs)
        else:
            output_model = output_model or "seq2seq_with_attention.keras"
            train_vqa(args.seq2seq_dir, output_model, args.figures_dir, use_attention=True, batch_size=args.batch_size, epochs=args.epochs)
        return

    if args.command == "train-classifier":
        train_classifier(args.data_dir, args.output_model, args.figures_dir, batch_size=args.batch_size, epochs=args.epochs)
        return

    if args.command == "train-regression":
        train_regression(args.metadata_csv, args.data_dir, args.output_model, args.figures_dir, batch_size=args.batch_size, epochs=args.epochs)
        return

    if args.command == "train-all":
        train_classifier(args.data_dir, args.classifier_model, args.figures_dir, batch_size=args.batch_size, epochs=args.epochs)
        train_regression(args.metadata_csv, args.data_dir, args.regression_model, args.figures_dir, batch_size=args.batch_size, epochs=args.epochs)
        return

    if args.command == "evaluate":
        metrics = evaluate_and_plot(args.seq2seq_dir, args.attention_model, args.no_attention_model, args.figure_path)
        print(metrics)
        return

    if args.command == "predict":
        image_path = args.image or input("Nhap duong dan anh: ").strip()
        question = args.question or input("Nhap cau hoi: ").strip()
        result = run_single_prediction(image_path, question, args.model, args.seq2seq_dir, args.data_dir)
        print(f"Question: {result['question']}")
        print(f"Image: {result['image_path']}")
        print(f"Detected fruit: {result['fruit_type']}")
        print(f"Detected count: {result['count']}")
        print(f"Predicted answer: {result['answer']}")
        return

    if args.command == "all":
        metadata = save_metadata_csv(args.data_dir, args.metadata_csv)
        stats = generate_qa_dataset(args.metadata_csv, args.data_dir, args.images_qa_dir, args.seq2seq_dir)
        print(f"Metadata rows: {len(metadata)}")
        print(stats)
        train_vqa(args.seq2seq_dir, "seq2seq_no_attention.keras", args.figures_dir, use_attention=False, batch_size=args.batch_size, epochs=args.epochs)
        train_vqa(args.seq2seq_dir, "seq2seq_with_attention.keras", args.figures_dir, use_attention=True, batch_size=args.batch_size, epochs=args.epochs)
        evaluate_and_plot(args.seq2seq_dir, "seq2seq_with_attention.keras", "seq2seq_no_attention.keras", "reports/figures/vqa_evaluation.png")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
