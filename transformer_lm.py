# models.py

import numpy as np
import torch
import torch.nn as nn


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index):
        self.model = model
        self.vocab_index = vocab_index
        self.max_seq_len = 20  # Maximum sequence length the model can handle

    def get_next_char_log_probs(self, context):
        """Get log probabilities for next character given context"""
        self.model.eval()  # Set to eval mode to disable dropout
        with torch.no_grad():
            # Truncate context if too long
            if len(context) > self.max_seq_len:
                context = context[-self.max_seq_len:]
            
            # Handle empty context
            if len(context) == 0:
                context = " "  # Use space as default
            
            # Convert context to indices
            context_indices = [self.vocab_index.index_of(c) for c in context]
            context_tensor = torch.LongTensor(context_indices)
            
            # Get model predictions
            log_probs, _ = self.model(context_tensor)
            
            # Return log probabilities for the last position (predicting next char)
            result = log_probs[-1].cpu().numpy()
            
            # Verify it sums to 1 in probability space (debugging)
            # prob_sum = np.sum(np.exp(result))
            # if abs(prob_sum - 1.0) > 0.01:
            #     print(f"Warning: probs sum to {prob_sum}")
            
            return result

    def get_log_prob_sequence(self, next_chars, context):
        """
        Completely rewrote this method to call get_next_char_log_probs repeatedly
        Previous version used chunking which caused sanity check failures
        This version builds context character-by-character as required
        """
        total_log_prob = 0.0
        current_context = context
        
        for char in next_chars:
            # Get log probs for next character
            char_log_probs = self.get_next_char_log_probs(current_context)
            
            # Get the log prob for this specific character
            char_idx = self.vocab_index.index_of(char)
            total_log_prob += char_log_probs[char_idx]
            
            # Update context for next iteration
            current_context = current_context + char
            
            # Keep context within max length
            if len(current_context) > self.max_seq_len:
                current_context = current_context[-self.max_seq_len:]
        
        return float(total_log_prob)


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    import torch
    import torch.nn as nn
    from torch import optim
    import random
    from transformer import Transformer
    
    # Model parameters
    vocab_size = len(vocab_index)
    seq_len = 20
    d_model = 128 # Increased from 64 to 128 for better model capacity
    d_internal = 64  # Must be divisible by num_heads (128 / 4 = 32)
    num_classes = vocab_size  # Predict next character
    num_layers = 2
    num_heads = 4  # Multi-head attention for better LM performance
    
    # CRITICAL: Use causal mask for language modeling
    model = Transformer(vocab_size, seq_len, d_model, d_internal, num_classes, num_layers, use_causal_mask=True, num_heads=num_heads)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fcn = nn.NLLLoss()
    
    # Create training sequences - non-overlapping chunks
    num_sequences = min(len(train_text) - seq_len, 15000)
    
    print(f"Training on {num_sequences} sequences with {num_heads}-head attention...")
    
    num_epochs = 15
    batch_positions = list(range(0, len(train_text) - seq_len - 1, seq_len))
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        random.shuffle(batch_positions)
        
        for start_pos in batch_positions[:1000]:  # Train on subset for speed
            # Get sequence
            input_seq = train_text[start_pos:start_pos+seq_len]
            target_seq = train_text[start_pos+1:start_pos+seq_len+1]
            
            if len(input_seq) != seq_len or len(target_seq) != seq_len:
                continue
            
            # Convert to indices
            try:
                input_indices = torch.LongTensor([vocab_index.index_of(c) for c in input_seq])
                target_indices = torch.LongTensor([vocab_index.index_of(c) for c in target_seq])
            except:
                continue  # Skip if character not in vocab
            
            optimizer.zero_grad()
            log_probs, _ = model(input_indices)
            
            # Compute loss for all positions
            loss = loss_fcn(log_probs, target_indices)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Early stopping if loss is good enough
        if avg_loss < 1.5:
            print("Loss threshold reached, stopping early")
            break
    
    model.eval()
    return NeuralLanguageModel(model, vocab_index)