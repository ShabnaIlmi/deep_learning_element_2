# Transformer Letter Counting & Language Modeling

This repository contains PyTorch implementations of transformer-based models for two tasks:
- character-level language modeling on the `text8` dataset
- synthetic letter-counting classification using fixed-length sequences

It also includes attention analysis utilities to inspect and visualize how transformer layers focus over input positions.

## Repository name suggestion
`transformer-lettercount-lm`

## Project structure
- `transformer.py` - core Transformer model and training/inference logic
- `transformer_lm.py` - language model training and evaluation driver
- `letter_counting.py` - letter counting classifier task driver
- `lm.py` - evaluation utility for language modeling
- `analyze_attention.py` - attention analysis and plotting
- `utils.py` / `_utils.py` - helper utilities and data indexing
- `data/` - training and development datasets
- `plots/` - generated visualizations
- `analysis/` - output analysis JSON
- `requirements.txt` - Python dependencies

## Setup

1. Create a Python environment (recommended Python 3.9).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run letter counting classification

```bash
python letter_counting.py --task BEFORE --train data/lettercounting-train.txt --dev data/lettercounting-dev.txt
```

### Run language model evaluation

```bash
python lm.py --model NEURAL --train_path data/text8-100k.txt --dev_path data/text8-dev.txt
```

### Analyze transformer attention patterns

```bash
python analyze_attention.py
```

## Notes
- The code is designed for academic/analysis experiments with transformers.
- `letter_counting.py` uses fixed-length 20-character examples and predicts counts of previous character occurrences.
- `lm.py` evaluates a character-level transformer on `text8` splits.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

If you want, I can also add a short `LICENSE` file or convert this into a more complete repository README.