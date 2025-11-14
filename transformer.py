# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, use_causal_mask=False, num_heads=1):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        :param use_causal_mask: whether to use causal masking (for language modeling)
        :param num_heads: number of attention heads (1 for single-head, 4/8 for multi-head)
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, num_positions)
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal, num_heads) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.use_causal_mask = use_causal_mask

    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        x = self.embedding(indices)
        x = self.pos_encoding(x)
        
        # Create causal mask if needed
        mask = None
        if self.use_causal_mask:
            seq_len = indices.shape[0]
            mask = torch.tril(torch.ones(seq_len, seq_len))
        
        attention_maps = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attention_maps.append(attn)
        
        # Apply output layer to get logits for each position
        logits = self.output_layer(x)
        # Apply log_softmax to get proper log probabilities
        log_probs = self.log_softmax(logits)
        
        return log_probs, attention_maps


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal, num_heads=1):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        :param num_heads: Number of attention heads (default 1 for single-head, use 4/8 for multi-head)
        """
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_heads = num_heads
        
        # Verify d_internal is divisible by num_heads
        if d_internal % num_heads != 0:
            raise ValueError(f"d_internal ({d_internal}) must be divisible by num_heads ({num_heads})")
        
        self.head_dim = d_internal // num_heads
        
        # Self-attention components (applied across all heads)
        self.query = nn.Linear(d_model, d_internal)
        self.key = nn.Linear(d_model, d_internal)
        self.value = nn.Linear(d_model, d_internal)
        
        # Output projection
        self.output_proj = nn.Linear(d_internal, d_model)
        
        # Feed-forward network
        self.ff1 = nn.Linear(d_model, d_model * 4)
        self.ff2 = nn.Linear(d_model * 4, d_model)
        self.relu = nn.ReLU()

    def forward(self, input_vecs, mask=None):
        # input_vecs shape: (seq_len, d_model)
        seq_len = input_vecs.shape[0]
        
        # Self-attention
        Q = self.query(input_vecs)  # (seq_len, d_internal)
        K = self.key(input_vecs)    # (seq_len, d_internal)
        V = self.value(input_vecs)  # (seq_len, d_internal)
        
        # Reshape for multi-head attention: (seq_len, num_heads, head_dim)
        Q = Q.view(seq_len, self.num_heads, self.head_dim)
        K = K.view(seq_len, self.num_heads, self.head_dim)
        V = V.view(seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (num_heads, seq_len, head_dim) for batch matmul
        Q = Q.transpose(0, 1)  # (num_heads, seq_len, head_dim)
        K = K.transpose(0, 1)  # (num_heads, seq_len, head_dim)
        V = V.transpose(0, 1)  # (num_heads, seq_len, head_dim)
        
        # Compute attention scores for each head
        # (num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)
        
        # Apply causal mask if provided
        if mask is not None:
            # mask shape: (seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax over the key dimension (dim=2)
        attention_weights = torch.softmax(scores, dim=2)  # (num_heads, seq_len, seq_len)
        
        # Apply attention to values
        # (num_heads, seq_len, head_dim)
        attended = torch.matmul(attention_weights, V)
        
        # Transpose back to (seq_len, num_heads, head_dim)
        attended = attended.transpose(0, 1).contiguous()  # (seq_len, num_heads, head_dim)
        
        # Reshape back: (seq_len, d_internal)
        attended = attended.view(seq_len, self.num_heads * self.head_dim)
        
        # Output projection
        attended = self.output_proj(attended)
        
        # First residual connection
        x = input_vecs + attended
        
        # Feed-forward network
        ff_out = self.ff2(self.relu(self.ff1(x)))
        
        # Second residual connection
        output = x + ff_out
        
        # Return attention weights for the first head (for visualization)
        attention_viz = attention_weights[0]  # (seq_len, seq_len)
        
        return output, attention_viz


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


def train_classifier(args, train, dev):
    # Model parameters — increased for better accuracy
    vocab_size = 27  # a-z + space
    num_positions = 20
    d_model = 128  # Increased from 64
    d_internal = 64  # Increased from 32
    num_classes = 3
    num_layers = 2
    
    # Determine if we need causal mask based on task
    use_causal_mask = (args.task == "BEFORE")
    
    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, use_causal_mask)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)  # Reduced learning rate for stability
    loss_fcn = nn.NLLLoss()
    
    num_epochs = 30  # Increased from 15
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        random.seed(epoch)
        
        ex_idxs = list(range(len(train)))
        random.shuffle(ex_idxs)
        
        for ex_idx in ex_idxs:
            example = train[ex_idx]
            
            model.zero_grad()
            log_probs, _ = model(example.input_tensor)
            loss = loss_fcn(log_probs, example.output_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    model.eval()
    return model



def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))


##############################################
#  Training / Tuning / Debug utilities
##############################################


class TrainingMonitor(object):
    """Lightweight monitor to store losses and attention maps during training."""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.attention_maps = []  # list of attention maps snapshots

    def update_losses(self, train_loss, val_loss):
        self.train_losses.append(float(train_loss))
        self.val_losses.append(float(val_loss))

    def store_attention(self, attn_maps):
        # attn_maps is a list of per-layer attention tensors
        self.attention_maps.append([a.detach().cpu().numpy() for a in attn_maps])


class DebugHelper(object):
    """Utilities for debugging model behavior and training process."""
    @staticmethod
    def analyze_gradients(model: nn.Module) -> dict:
        grad_stats = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_stats[name] = {
                    'mean': float(p.grad.mean().item()),
                    'std': float(p.grad.std().item()),
                    'min': float(p.grad.min().item()),
                    'max': float(p.grad.max().item())
                }
        return grad_stats

    @staticmethod
    def summarize_attention_snapshot(attn_snapshot):
        # attn_snapshot: list of numpy arrays per layer
        summary = {}
        for li, layer_attn in enumerate(attn_snapshot):
            # layer_attn shape: (batch, heads?, seq_len, seq_len) or (seq_len, seq_len)
            arr = layer_attn
            # Flatten and compute simple stats
            summary[f'layer_{li}'] = {
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'max': float(arr.max()),
                'min': float(arr.min())
            }
        return summary


def grid_search_hyperparameters(train_bundles, dev_bundles, vocab_size=27, num_positions=20,
                                model_grid=None, train_grid=None, max_trials=10):
    """
    Small helper to run a grid search over a small set of model/training hyperparameters.
    This is intentionally lightweight and intended for experimentation only (not full sweeps).
    model_grid: dict of lists for model params: d_model, d_internal, num_layers
    train_grid: dict of lists for training params: lr, epochs
    Returns: list of (config, val_loss) tuples sorted by val_loss
    """
    if model_grid is None:
        model_grid = {'d_model': [64, 128], 'd_internal': [32, 64], 'num_layers': [1, 2]}
    if train_grid is None:
        train_grid = {'lr': [1e-3, 5e-4], 'epochs': [5]}

    import itertools
    results = []
    combos = list(itertools.product(*(model_grid[k] for k in model_grid), *(train_grid[k] for k in train_grid)))
    # combos ordering: model params then train params
    for idx, combo in enumerate(combos):
        if idx >= max_trials:
            break
        # unpack
        md_vals = combo[:len(model_grid)]
        tr_vals = combo[len(model_grid):]
        model_conf = dict(zip(model_grid.keys(), md_vals))
        train_conf = dict(zip(train_grid.keys(), tr_vals))

        print(f"Grid trial {idx+1}: model={model_conf}, train={train_conf}")
        # Build model
        model = Transformer(vocab_size, num_positions, model_conf['d_model'], model_conf['d_internal'], 3, model_conf['num_layers'])
        # Quick train using provided train_classifier (single-example loop). We'll adapt small epochs.
        args = type('obj', (object,), {'task': 'BEFORE'})
        # create small copies of bundles to avoid side-effects
        small_train = train_bundles[:200]
        small_dev = dev_bundles[:200]
        # Set hyperparams inside train loop by temporarily patching values (train_classifier is self-contained)
        # We'll implement a lightweight training here to control lr/epochs explicitly
        optimizer = optim.Adam(model.parameters(), lr=train_conf['lr'])
        loss_fcn = nn.NLLLoss()
        for epoch in range(train_conf['epochs']):
            model.train()
            total_loss = 0.0
            for ex in small_train:
                optimizer.zero_grad()
                log_probs, _ = model(ex.input_tensor)
                loss = loss_fcn(log_probs, ex.output_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        # Evaluate on dev
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for ex in small_dev:
                log_probs, _ = model(ex.input_tensor)
                loss = loss_fcn(log_probs, ex.output_tensor)
                total_val += loss.item()
        avg_val = total_val / max(1, len(small_dev))
        print(f"  Val loss: {avg_val:.4f}")
        results.append({'model_conf': model_conf, 'train_conf': train_conf, 'val_loss': avg_val})

    results = sorted(results, key=lambda x: x['val_loss'])
    return results


def overfit_small_set_example(train_bundles, n_examples=5, epochs=200, lr=1e-3):
    """Quick overfit test: train a small model to overfit n_examples to verify implementation."""
    small_train = train_bundles[:n_examples]
    args = type('obj', (object,), {'task': 'BEFORE'})
    # build small model
    vocab_size = 27
    model = Transformer(vocab_size, 20, d_model=32, d_internal=16, num_classes=3, num_layers=1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fcn = nn.NLLLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for ex in small_train:
            optimizer.zero_grad()
            log_probs, _ = model(ex.input_tensor)
            loss = loss_fcn(log_probs, ex.output_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 20 == 0:
            print(f"Overfit epoch {epoch+1}, loss {total_loss/len(small_train):.4f}")
    # report final accuracy
    model.eval()
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for ex in small_train:
            log_probs, _ = model(ex.input_tensor)
            preds = np.argmax(log_probs.detach().numpy(), axis=1)
            num_correct += sum(preds == ex.output.astype(int))
            num_total += len(preds)
    print(f"Overfit accuracy on {n_examples} examples: {num_correct}/{num_total} = {num_correct/num_total:.4f}")
    return model