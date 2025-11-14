"""
run_mem2_tune.py

Quick tuning script for Member 2 (letter counting) to attempt to reach >95% accuracy on dev.

Usage (PowerShell):
  python run_mem2_tune.py *> mem2_tune_run.txt

This script will:
 - Read the letter-counting train/dev files
 - Run a tiny overfit check on 5 examples
 - Run a short grid-search over small models
 - Train a final model with the best config (with more epochs) and report dev accuracy

Note: This is intended for quick local experimentation. If GPU isn't available, runs may be slow.
"""

import time
import random
import numpy as np
from transformer import *
from letter_counting import read_examples, get_letter_count_output
from utils import Indexer


def build_bundles(path, vocab_index, count_only_previous=True):
    lines = read_examples(path)
    bundles = [LetterCountingExample(l, get_letter_count_output(l, count_only_previous), vocab_index) for l in lines]
    return bundles


def train_full_model(train_bundles, dev_bundles, model_conf, train_conf):
    # build model from model_conf
    vocab_size = 27
    num_positions = 20
    model = Transformer(vocab_size, num_positions,
                        d_model=model_conf['d_model'],
                        d_internal=model_conf['d_internal'],
                        num_classes=3,
                        num_layers=model_conf['num_layers'],
                        use_causal_mask=True if train_conf.get('use_causal', False) else False)
    optimizer = optim.Adam(model.parameters(), lr=train_conf['lr'])
    loss_fcn = nn.NLLLoss()

    epochs = train_conf.get('epochs', 50)
    print(f"Training final model: {model_conf}, lr={train_conf['lr']}, epochs={epochs}")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for ex in train_bundles:
            optimizer.zero_grad()
            log_probs, _ = model(ex.input_tensor)
            loss = loss_fcn(log_probs, ex.output_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"Epoch {epoch+1}/{epochs}, avg loss {total_loss / len(train_bundles):.4f}")
    # eval
    model.eval()
    print("Final evaluation on dev set:")
    decode(model, dev_bundles, do_print=False, do_plot_attn=False)
    return model


if __name__ == '__main__':
    start = time.time()
    random.seed(0)
    np.random.seed(0)
    print("Building vocab...")
    vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
    vocab_index = Indexer()
    for c in vocab:
        vocab_index.add_and_get_index(c)
    print(repr(vocab_index))

    # Read data
    print("Loading train/dev bundles (BEFORE task)...")
    train_bundles = build_bundles('data/lettercounting-train.txt', vocab_index, count_only_previous=True)
    dev_bundles = build_bundles('data/lettercounting-dev.txt', vocab_index, count_only_previous=True)
    print(f"Train size: {len(train_bundles)}, Dev size: {len(dev_bundles)}")

    # 1) Overfit a tiny set to verify model can fit
    print("Running overfit sanity check (5 examples)...")
    overfit_model = overfit_small_set_example(train_bundles, n_examples=5, epochs=200, lr=1e-3)

    # 2) Quick grid-search (short runs)
    print("Running quick grid search (short) to find promising configs...")
    model_grid = {'d_model': [64, 128], 'd_internal': [32, 64], 'num_layers': [1, 2, 3]}
    train_grid = {'lr': [1e-3, 5e-4], 'epochs': [5]}
    results = grid_search_hyperparameters(train_bundles, dev_bundles, model_grid=model_grid, train_grid=train_grid, max_trials=6)
    print("Grid search results (top 3):")
    for r in results[:3]:
        print(r)

    if len(results) == 0:
        print("No grid results found; aborting")
    else:
        best = results[0]
        best_model_conf = best['model_conf']
        # Use a stronger train config for final training
        final_train_conf = {'lr': 5e-4, 'epochs': 50}
        print(f"Selected best model conf: {best_model_conf}, now training final model with {final_train_conf}")
        trained = train_full_model(train_bundles, dev_bundles, best_model_conf, final_train_conf)

    print(f"Done in {time.time() - start:.1f}s")
