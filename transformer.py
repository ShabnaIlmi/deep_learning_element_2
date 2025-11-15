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


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).

# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    """Data structure for letter counting examples with input/output pairs."""

    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        # Storing the raw input string
        self.input = input
        
        # Converting characters in the input string to their corresponding vocabulary indices
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        
        # Converting the indexed input into a PyTorch tensor
        self.input_tensor = torch.LongTensor(self.input_indexed)
        
        # Storing the raw output labels 
        self.output = output
        
        # Converting the output labels into a PyTorch tensor
        self.output_tensor = torch.LongTensor(self.output)

class Transformer(nn.Module):
    """Main Transformer model for sequence classification tasks."""
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
        # Character embedding layer 
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding 
        self.pos_encoding = PositionalEncoding(d_model, num_positions)
        # 
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal, num_heads) for _ in range(num_layers)])
        # Final classification layer 
        self.output_layer = nn.Linear(d_model, num_classes)
        # Log softmax for probability output
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.use_causal_mask = use_causal_mask

    def forward(self, indices):
        """
        Forward pass through the transformer model.
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # Converting indices to embeddings
        x = self.embedding(indices)
        # Adding positional information
        x = self.pos_encoding(x)
        
        # Creating causal mask if needed 
        mask = None
        if self.use_causal_mask:
            seq_len = indices.shape[0]
            mask = torch.tril(torch.ones(seq_len, seq_len))  # Lower triangular mask
        
        # Passing through transformer layers, collecting attention maps
        attention_maps = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attention_maps.append(attn)
        
        # Applying output layer to get logits for each position
        logits = self.output_layer(x)
        # Applying log_softmax to get proper log probabilities
        log_probs = self.log_softmax(logits)
        
        return log_probs, attention_maps
class TransformerLayer(nn.Module):
    """Single transformer layer with multi-head self-attention and feed-forward network."""
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
        
        # Verifying d_internal is divisible by num_heads for proper head splitting
        if d_internal % num_heads != 0:
            raise ValueError(f"d_internal ({d_internal}) must be divisible by num_heads ({num_heads})")
        
        # Dimension per attention head
        self.head_dim = d_internal // num_heads
        
        # Self-attention components: linear projections for Q, K, V
        self.query = nn.Linear(d_model, d_internal)
        self.key = nn.Linear(d_model, d_internal)
        self.value = nn.Linear(d_model, d_internal)
        
        # Output projection to map back to d_model dimension
        self.output_proj = nn.Linear(d_internal, d_model)
        
        # Feed-forward network (standard transformer FFN with expansion factor 4)
        self.ff1 = nn.Linear(d_model, d_model * 4)
        self.ff2 = nn.Linear(d_model * 4, d_model)
        self.relu = nn.ReLU()


    def forward(self, input_vecs, mask=None):
        """Forward pass through transformer layer with self-attention and FFN."""
        # input_vecs shape: (seq_len, d_model)
        seq_len = input_vecs.shape[0]
        
        # SELF-ATTENTION 
        # Project input to query, key, value spaces
        Q = self.query(input_vecs)  # (seq_len, d_internal)
        K = self.key(input_vecs)    # (seq_len, d_internal)
        V = self.value(input_vecs)  # (seq_len, d_internal)
        
        # Reshaping for multi-head attention: split d_internal across heads
        Q = Q.view(seq_len, self.num_heads, self.head_dim)
        K = K.view(seq_len, self.num_heads, self.head_dim)
        V = V.view(seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (num_heads, seq_len, head_dim) for efficient batch operations
        Q = Q.transpose(0, 1)  # (num_heads, seq_len, head_dim)
        K = K.transpose(0, 1)  # (num_heads, seq_len, head_dim)
        V = V.transpose(0, 1)  # (num_heads, seq_len, head_dim)
        
        # Computing scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)  
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=2)  
        
        attended = torch.matmul(attention_weights, V)  
        
        # Reshaping back to original format
        attended = attended.transpose(0, 1).contiguous()  # (seq_len, num_heads, head_dim)
        attended = attended.view(seq_len, self.num_heads * self.head_dim)  # (seq_len, d_internal)
        
        # Project back to model dimension
        attended = self.output_proj(attended)
        
        # First residual connection (around attention)
        x = input_vecs + attended
        
        # FEED-FORWARD BLOCK 
        # Two-layer MLP with ReLU activation
        ff_out = self.ff2(self.relu(self.ff1(x)))
        
        # Second residual connection (around FFN)
        output = x + ff_out
        
        # Return attention weights from first head for visualization
        attention_viz = attention_weights[0]  
        
        return output, attention_viz

# Implementation of positional encoding
class PositionalEncoding(nn.Module):
    """Learnable positional encoding module that adds position information to embeddings."""
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
        # Learnable position embeddings 
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        Add positional encodings to input embeddings.
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Getting the sequence length from input tensor
        input_size = x.shape[-2]
        # Creating position indices [0, 1, 2, ..., seq_len-1]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        
        if self.batched:
            # Adding batch dimension for broadcasting across batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            # Direct addition for single sequences
            return x + self.emb(indices_to_embed)


def train_classifier(args, train, dev):
    """Main training function for the transformer classifier."""
    # Model hyperparameters (tuned for better performance)
    vocab_size = 27  # a-z + space
    num_positions = 20  # Maximum sequence length
    d_model = 64  # Hidden dimension (reduced for speed)
    d_internal = 32  # Attention dimension (reduced for speed)
    num_classes = 3  # Number of output classes
    num_layers = 2  # Number of transformer layers
    
    # Task-specific configuration: use causal mask for "BEFORE" task
    use_causal_mask = (args.task == "BEFORE")
    
    # Initializing the model and training components
    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, use_causal_mask)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Higher learning rate for faster convergence
    loss_fcn = nn.NLLLoss()  # Negative log-likelihood for classification
    
    # Training loop
    num_epochs = 10  # Increased training duration
    for epoch in range(num_epochs):
        model.train()  # Set to training mode
        total_loss = 0.0
        random.seed(epoch)  # Reproducible shuffling
        
        # Shuffle training examples each epoch
        ex_idxs = list(range(len(train)))
        random.shuffle(ex_idxs)
        
        # Processing  each training example
        for ex_idx in ex_idxs:
            example = train[ex_idx]
            
            # Standard training step: forward pass, loss, backward pass, update
            model.zero_grad()
            log_probs, _ = model(example.input_tensor)
            loss = loss_fcn(log_probs, example.output_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Report epoch progress
        avg_loss = total_loss / len(train)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    model.eval()  
    return model



def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Evaluate model on development set with optional visualization and debugging output.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    
    # Disable verbose output for large datasets
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    
    # Evaluating each example
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        # Get model predictions and attention maps
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        # Converting log probabilities to class predictions
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        
        # Printing the input/output for debugging
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        
        # Saving the attention visualizations
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                # Heatmap for attention weights
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                # Label axes with input characters
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # Saving the plot to file
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        
        # Calculating the accuracy
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    
    # Report final accuracy
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
