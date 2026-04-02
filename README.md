# Fruit VQA

Fruit VQA is a visual question answering project for fruit images. It combines dataset metadata, image-based fruit attribute extraction, and a seq2seq attention model to answer questions such as fruit type, count, and simple scene queries. The Streamlit app uses the default VQA model automatically, so there is no model selector in the UI.

Link Deployed: https://fruit-visual-question-answering.streamlit.app/

## Key Features

- Browse fruit samples from the built-in test gallery.
- Upload an image and let the app infer fruit type and count before answering.
- Run preprocessing, training, evaluation, and single-image prediction from one CLI entry point.
- Use the attention-based VQA model saved as `seq2seq_attention_full.keras` by default.
- Save training and evaluation artifacts as `.png`, `.csv`, and `.json` files under `reports/figures/`.

## Language & Libraries

- Python 3.10+
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- Pillow
- Matplotlib
- scikit-learn
- NLTK
- rouge-score
- tqdm

## Dataset

The project uses a fruit image dataset stored under `data/` with train, validation, and test splits.

```text
data/
  train/
  valid/
  test/
```

The preprocessing step scans the dataset and generates the supporting assets used by training and prediction:

- `data/FruitDataFrame.csv`
- `Images_QA/`
- `seq2seqData/`

The project also includes pre-trained model files at the repository root:

- `fruit_classifier.keras`
- `fruit_regression.keras`
- `fruit_regression.scale.json`
- `seq2seq_attention_full.keras`

## Methodology (Core AI)

The pipeline is built around two stages:

1. Fruit attribute extraction.
   - For dataset images, fruit type and count come from the dataset labels.
   - For uploaded images, the app first uses the classifier and regression models to estimate fruit type and count.
2. VQA answer generation.
   - The fruit attributes are converted into a feature vector.
   - A seq2seq model with attention consumes the question text and the fruit feature vector to generate the answer.

In practice, the workflow is:

- preprocess dataset metadata and QA pairs,
- train or load the fruit classifier and regression model,
- train or load the VQA attention model,
- feed the question plus fruit features into the VQA model.

## Evaluation & Results

The latest attention-model run is summarized in `reports/training_summary.md` and the exported artifacts are stored in `reports/figures/`.

Latest reported results:

| Metric | Value |
| --- | ---: |
| Training accuracy | 0.9916 |
| Validation accuracy | 0.9818 |
| Training loss | 0.0255 |
| Validation loss | 0.0463 |
| BLEU | 0.6074 |
| ROUGE-1 | 0.8547 |

Evaluation was run on a 100-sample test subset.

## Installation & Usage

Install the dependencies first:

```bash
conda activate ai_env
pip install -r requirements.txt
```

If you prefer a fresh virtual environment, create one first and then install the requirements.

Common commands:

```bash
python main.py preprocess
python main.py train vqa-attention --epochs 20 --batch-size 64 --output-model seq2seq_attention_full.keras
python main.py evaluate
streamlit run streamlit_app.py
```

Optional training commands:

```bash
python main.py train-classifier --epochs 20 --batch-size 16 --output-model fruit_classifier.keras
python main.py train-regression --epochs 20 --batch-size 16 --output-model fruit_regression.keras
python main.py train-all --epochs 20 --batch-size 16
```

Single prediction from the terminal:

```bash
python main.py predict --image "data/test/.../sample.jpg" --question "Trong ảnh có bao nhiêu trái cây?" --model seq2seq_attention_full.keras
```

## Project Structure

```
Final_version/
├── data/                         # Dataset folders
│   ├── train/                    # Training images
│   ├── valid/                    # Validation images
│   └── test/                     # Test images
├── Images_QA/                    # Images used specifically for VQA tasks
├── reports/                      # Project reports and documentation
│   ├── figures/                  # Visualization plots and charts
│   └── training_summary.md       # Summary of training results
├── seq2seqData/                  # Data specific to the Seq2Seq model
├── src/                          # Core source code
│   ├── __init__.py
│   ├── preprocess.py             # Data preprocessing logic
│   ├── predict.py                # Inference and prediction logic
│   ├── train.py                  # Model training scripts
│   ├── evaluate.py               # Evaluation metrics and scripts
│   └── model.py                  # Model architecture definitions
├── fruit_classifier.keras        # Classification model file
├── fruit_regression.keras        # Regression model file
├── fruit_regression.scale.json   # Scaling parameters for regression
├── seq2seq_attention_full.keras  # VQA Seq2Seq Attention model file
├── main.py                       # Main execution script
├── streamlit_app.py              # Streamlit Web application script
├── requirements.txt              # List of Python dependencies
└── README.md                     # Project documentation and instructions
```

Main modules:

- `main.py`: CLI entry point for preprocessing, training, evaluation, and prediction.
- `streamlit_app.py`: Streamlit interface for interactive VQA.
- `src/preprocess.py`: dataset scanning, metadata generation, and QA dataset building.
- `src/predict.py`: image loading, feature extraction, and answer generation.
- `src/train.py`: model training routines.
- `src/evaluate.py`: evaluation and plotting utilities.
