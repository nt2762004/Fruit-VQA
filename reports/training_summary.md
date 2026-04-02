# Training Summary

This summary records the latest end-to-end run for the Fruit VQA attention model.

## Saved Artifacts

- [Full attention training plot](figures/seq2seq_attention_full_training.png)
- [Full attention training metrics CSV](figures/seq2seq_attention_full_training.csv)
- [Full attention training metrics JSON](figures/seq2seq_attention_full_training.json)
- [100-sample test plot](figures/vqa_attention_full_test.png)
- [100-sample test metrics CSV](figures/vqa_attention_full_test.csv)
- [100-sample test metrics JSON](figures/vqa_attention_full_test.json)

## Final Training Results

### VQA Attention Full
- Epochs: 5
- Final training accuracy: 0.9916
- Final validation accuracy: 0.9818
- Final training loss: 0.0255
- Final validation loss: 0.0463

## Final Evaluation Results

Evaluated on a 100-sample test subset.

- BLEU: 0.6074
- ROUGE-1: 0.8547
- Inference time: 0.0790 s

## Model File

- [Attention model](../seq2seq_attention_full.keras)

## Notes

- Training history is saved as PNG, CSV, and JSON so the process is visible without opening the notebook UI.
- Evaluation metrics are saved as PNG, CSV, and JSON for quick inspection.
- This summary reflects the latest attention-only run using the dedicated full model and its separate test artifacts.
- Run a single-image prediction with `python main.py predict --image <path> --question <text> --model seq2seq_attention_full.keras`.
