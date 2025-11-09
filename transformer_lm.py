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

    def get_next_char_log_probs(self, context):
        self.model.eval()
        with torch.no_grad():
            # Convert context to indices
            context_indices = [self.vocab_index.index_of(c) for c in context]
            context_tensor = torch.LongTensor(context_indices)
            
            # Get model predictions
            log_probs, _ = self.model(context_tensor)
            
            # Return log probabilities for the last position
            return log_probs[-1].numpy()

    def get_log_prob_sequence(self, next_chars, context):
        total_log_prob = 0.0
        current_context = context
        
        for char in next_chars:
            char_log_probs = self.get_next_char_log_probs(current_context)
            char_idx = self.vocab_index.index_of(char)
            total_log_prob += char_log_probs[char_idx]
            current_context += char
        
        return total_log_prob


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
    d_model = 64
    d_internal = 32
    num_classes = vocab_size  # Predict next character
    num_layers = 2
    
    model = Transformer(vocab_size, seq_len, d_model, d_internal, num_classes, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fcn = nn.NLLLoss()
    
    # Prepare training data
    def create_sequences(text, seq_len):
        sequences = []
        for i in range(len(text) - seq_len):
            input_seq = text[i:i+seq_len]
            target_seq = text[i+1:i+seq_len+1]
            sequences.append((input_seq, target_seq))
        return sequences
    
    train_sequences = create_sequences(train_text, seq_len)
    
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        random.shuffle(train_sequences)
        
        for i, (input_seq, target_seq) in enumerate(train_sequences[:10000]):  # Limit for faster training
            # Convert to indices
            input_indices = torch.LongTensor([vocab_index.index_of(c) for c in input_seq])
            target_indices = torch.LongTensor([vocab_index.index_of(c) for c in target_seq])
            
            optimizer.zero_grad()
            log_probs, _ = model(input_indices)
            
            # Compute loss for all positions
            loss = loss_fcn(log_probs, target_indices)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 1000 == 0:
                print(f"Epoch {epoch + 1}, Step {i}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch + 1} completed, Average Loss: {total_loss / min(10000, len(train_sequences)):.4f}")
    
    model.eval()
    return NeuralLanguageModel(model, vocab_index)
