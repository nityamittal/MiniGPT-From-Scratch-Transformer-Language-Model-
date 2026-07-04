"""
EECS 595 HW3: GPT Implementation

This file contains all the core classes and functions needed to implement a GPT-style
decoder language model using more recent techniques (e.g., RoPE, SwiGLU, etc.).

 Students should implement the TODO sections in each class and function.
"""

import os
import math
import numpy as np
import random
import logging
from typing import Optional, Callable, List, Tuple, Dict, Any

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import RMSNorm
from torch.amp import autocast, GradScaler

# Data loading imports
from torch.utils.data import Dataset, IterableDataset, TensorDataset, DataLoader
import json
import glob
import gzip
import bz2

# Arrow dataset support
from datasets import load_from_disk

# Tokenization imports
from transformers import AutoTokenizer

# Progress and timing
from tqdm.auto import tqdm, trange
import time

# RoPE imports
from rope import Rotary, apply_rotary_pos_emb


# =============================================================================
# GPT Embedding Layer (with RoPE instead of positional embeddings)
# =============================================================================

class GPTEmbedding(nn.Module):
    """
    GPT Embedding Layer.

    This layer only handles token embeddings. Positional information is handled
    by RoPE in the attention mechanism.
    """
    def __init__(self, vocab_size: int,
                 emb_dim: int = 768,
                 context_length: int = 512):
        """
        Initialize the GPT embedding layer.

        Args:
            vocab_size: Size of the vocabulary
            emb_dim: Embedding dimension
            context_length: Maximum context length (not used in RoPE version)
        """
        super().__init__()

        ###########################################################################
        #                            TODO 1.1: YOUR CODE HERE                         #
        #                                                                         #
        # 1. Create an embedding layer for tokens (token IDs from the tokenizer). #
        # 2. Note: We don't need positional embeddings since we use RoPE!        #
        ###########################################################################

        self.token_embeddings = nn.Embedding(vocab_size, emb_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embedding layer.

        Args:
            token_ids: Tensor of shape (batch_size, seq_length)
        Returns:
            token embeddings: Tensor of shape (batch_size, seq_length, hidden_dim)
        """
        ###########################################################################
        #                            TODO 1.2: YOUR CODE HERE                         #
        #                                                                         #
        # 1. Obtain token embeddings from the token embedding layer.              #
        # 2. Return the token embeddings (no positional embeddings needed!)       #
        ###########################################################################

        token_embeds = self.token_embeddings(token_ids) 
        return token_embeds


# =============================================================================
# Multi-Head Attention with RoPE
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Rotary Position Embedding (RoPE).

    This implementation uses RoPE to encode positional information directly
    in the attention mechanism instead of using separate positional embeddings.
    """
    def __init__(self, d_in, context_length, dropout, num_heads, qkv_bias=False):
        """
        Initialize Multi-Head Attention with RoPE.

        Args:
            d_in: Dimension of the input embeddings
            context_length: Maximum sequence length (used for attention masking)
            dropout: Dropout probability
            num_heads: Number of attention heads
            qkv_bias: Whether to include bias in Q, K, V projections
        """
        super().__init__()

        ########################################################################################################################
        #                                                     TODO 1.3: YOUR CODE HERE                                             #
        #                                                                                                                      #
        # 1. Figure out how many dimensions each head should have                                                              #
        # 2. Create linear layers to turn the input embeddings into the query, key, and value projections                      #
        # 3. Calculate the scale factor (1 / sqrt(per-head embedding size))                                                    #
        # 4. Define output projection that merges heads back to model width                                                    #
        # 5. Create dropout module used after attention/MLP projections                                                        #
        # 6. Initialize RoPE for positional encoding                                                                          #
        #                                                                                                                      #
        # NOTE: Each of the Q, K, V projections represents the projections of *each* of the heads as one long sequence.        #
        #       Each of the layers is implicitly representing each head in different parts of its dimensions.                  #
        ######################################################################################################################

        d_out = d_in
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.context_length = context_length


        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out, bias=True)

        self.dropout = nn.Dropout(dropout)

        self.scale = 1.0 / math.sqrt(self.head_dim)


        # Initialize RoPE for positional encoding
        # NOTE: We'll give you this code
        self.rope = Rotary(head_dim=self.head_dim, max_seq_len=context_length, cache_dtypes=(torch.float32,torch.bfloat16))

        ##########################################################################
        #             NOTE: Causal Mask Implementation                           #
        #                                                                        #
        # Implement a *causal mask* to prevent each position in the sequence     #
        # from attending to future positions during training and inference.      #
        # In GPT, this ensures token i can only "see" tokens at positions <= i,  #
        # not any future tokens (no information leakage).                        #
        #                                                                        #
        # Instructions:                                                          #
        # 1. Construct an "upper triangular" boolean mask of shape               #
        #    (context_length, context_length), where positions [i, j] are True   #
        #    if j > i (i.e., above the diagonal: future positions).              #
        #    You can use torch.triu() for this.                                  #
        # 2. Register this mask as a buffer using `self.register_buffer`.        #
        #    - Why register as a buffer?                                         #
        #        - Buffers are tensors that are part of the module's state,      #
        #          moved to the correct device automatically, and saved/loaded   #
        #          with the model, but are not learnable parameters.             #
        #        - This is ideal for constant masks that shouldn't be trained.   #
        # 3. (OPTIONAL) For speed, you may precompute and cache masks for        #
        #    different sequence lengths in a dict (e.g., for variable-length     #
        #    sequences/batching), using self.masks[length] = ... for lengths up  #
        #    to context_length.                                                  #
        #                                                                        #
        # Helpful functions:                                                     #
        #   torch.triu()                                                         #
        #   self.register_buffer()                                               #
        #                                                                        #
        # Use this mask during attention to set attention scores for future      #
        # tokens to -inf before softmax, enforcing causality.                    #
        ##########################################################################
        mask = torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)

        self.register_buffer("mask", mask, persistent=False)


    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-head attention with RoPE.

        Args:
            embeds: Input embeddings of shape (batch_size, seq_length, d_in)
        Returns:
            Output embeddings of shape (batch_size, seq_length, d_out)
        """
        #################################################################################################################################
        #                                                   TODO 1.4: YOUR CODE HERE                                                        #
        #                                                                                                                               #
        # Implement multi-headed attention with RoPE:                                                                                   #
        #                                                                                                                               #
        # 1. Project input embeddings into Q, K, and V spaces                                                                          #
        # 2. Reshape Q, K, V to separate heads                                                                                         #
        # 3. Apply RoPE to Q and K (this encodes positional information!)                                                              #
        # 4. Compute attention scores: Q @ K^T                                                                                          #
        # 5. Apply causal mask (upper-triangular mask to -inf)                                                                          #
        # 6. Scale attention scores by 1/sqrt(head_dim)                                                                                #
        # 7. Apply softmax to get attention weights                                                                                     #
        # 8. Apply dropout to attention weights                                                                                         #
        # 9. Compute weighted sum: attention_weights @ V                                                                               #
        # 10. Reshape back to original format and apply output projection                                                               #
        #                                                                                                                               #
        # Key insight: RoPE replaces the need for separate positional embeddings!                                                      #
        #################################################################################################################################

        b, num_tokens, d_in = embeds.shape

        queries = self.W_query(embeds)
        keys    = self.W_key(embeds)
        values  = self.W_value(embeds)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        rope_cos, rope_sin = self.rope(queries)
        queries, keys = apply_rotary_pos_emb(queries, keys, rope_cos, rope_sin)

        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))

        attn_scores = attn_scores * self.scale

        if self.mask is not None:
            causal_mask = self.mask[:num_tokens, :num_tokens]
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"),)


        attn_weights = torch.softmax(attn_scores, dim=-1)


        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, values)

        attn_output = attn_output.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        out = self.out_proj(attn_output)  

        return out


# =============================================================================
# SwiGLU Activation Function
# =============================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU activation function with learnable gating mechanism.

    SwiGLU(x) = (xW1) ⊙ Swish(xW2)
    where Swish(x) = x · σ(x)
    """
    def __init__(self, dimension: int):
        """
        Initialize SwiGLU activation.

        Args:
            dimension: Input and output dimension
        """
        super().__init__()
        # NOTE: More recent implementations use a up and down projection for the main and gate paths,
        #       but we'll keep it simple for now.
        self.linear_1 = nn.Linear(dimension, dimension)  # main path
        self.linear_2 = nn.Linear(dimension, dimension)  # gate path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SwiGLU activation.

        Args:
            x: Input tensor of shape [..., dimension]
        Returns:
            Tensor of same shape after SwiGLU gating
        """
        main = self.linear_1(x)
        gate = self.linear_2(x)
        swish_gate = gate * torch.sigmoid(gate)
        return main * swish_gate


# =============================================================================
# Feed-Forward Network
# =============================================================================

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (MLP) used inside a Transformer block.

    Uses SwiGLU activation for better performance than traditional ReLU/GELU.
    """
    def __init__(self, emb_dim: int, expansion=8/3):
        """
        Initialize the feed-forward network.

        Args:
            emb_dim: Model/embedding width (D)
            expansion: Width multiplier for the hidden layer
        """
        super().__init__()

        ################################################################
        #                     TODO 1.5: YOUR CODE HERE                     #
        # Implement the layers for the SwiGLU FFN:                     #
        #   1) Calculate the hidden dimension 'd_ff'. This dimension   #
        #      is (expansion * emb_dim).                               #
        #      Reference: LLaMA paper, equation (2).                   #
        #   2) For efficiency, we'll compute W1 and W2 in one step.    #
        #      Define 'self.fc1' as a Linear layer that maps:          #
        #      emb_dim -> 2 * d_ff                                     #
        #   3) Define 'self.fc2' as the output layer that maps:        #
        #      d_ff -> emb_dim                                         #
        ################################################################

        # Calculate the hidden dimension
        # NOTE: The formula for d_ff in SwiGLU is (expansion * emb_dim).
        # We multiply by 2 in the fc1 layer because we are creating
        # *two* matrices (W1 and W2) of size [emb_dim, d_ff].

        d_ff = int(expansion * emb_dim)

        # More efficient implementation using chunking
        self.fc1 = nn.Linear(emb_dim, 2 * d_ff)
        self.fc2 = nn.Linear(d_ff, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.

        Args:
            x: Input tensor of shape [..., D]
        Returns:
            Output tensor of shape [..., D]
        """
        ################################################################
        #                     TODO 1.6: YOUR CODE HERE                     #
        # Implement the forward pass for SwiGLU:                       #
        #   1) Pass the input 'x' through 'self.fc1'.                  #
        #   2) 'Chunk' the result from fc1 into two separate tensors   #
        #      (x1 and x2) along the last dimension.                   #
        #   3) Apply the SwiGLU logic: output = (silu(x1) * x2)        #
        #      (Hint: use F.silu for the Swish activation)             #
        #   4) Pass the result through the output layer 'self.fc2'.    #
        ################################################################
        x_proj = self.fc1(x)

        x1, x2 = x_proj.chunk(2, dim=-1)

        x_glu = F.silu(x1) * x2

        out = self.fc2(x_glu)
        return out

# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """
    Transformer Block (Decoder Layer) with RoPE.

    This block assembles the core pieces of a GPT-style decoder layer:
    - Multi-head attention with RoPE
    - Position-wise feed-forward network
    - Pre-LayerNorm and residual connections
    """
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize Transformer Block.

        Required cfg keys:
            - emb_dim: int
            - context_length: int
            - n_heads: int
            - n_layers: int
            - drop_rate: float
        """
        super().__init__()

        ################################################################
        #                     TODO 1.7: YOUR CODE HERE                 #
        # Implement a *decoder-style* Transformer block for GPT with   #
        # pre-norm + residual connections.                             #
        #                                                              #
        # 1) Create a MultiHeadAttention layer with RoPE               #
        # 2) Create the position-wise feed-forward (MLP)               #
        # 3) Create two RMSNorms (pre-norm):                           #
        #      - norm1 applied before attention                        #
        #      - norm2 applied before MLP                              #
        # 4) Store dropout probability; use it after attn and MLP.     #
        ################################################################
        emb_dim = cfg["emb_dim"]
        context_length = cfg["context_length"]
        n_heads = cfg["n_heads"]
        drop_rate = cfg["drop_rate"]

        self.self_attn = MultiHeadAttention(d_in=emb_dim, context_length=context_length, dropout=drop_rate, num_heads=n_heads, qkv_bias=False,)

        self.ffn = FeedForward(emb_dim=emb_dim)

        self.norm1 = RMSNorm(emb_dim)
        self.norm2 = RMSNorm(emb_dim)

        self.dropout_p = drop_rate

    def maybe_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout if dropout_p > 0.

        Args:
            x: Input tensor
        Returns:
            Tensor with dropout applied (if enabled)
        """
        ################################################################
        #                     TODO 1.8: YOUR CODE HERE                     #
        # Apply dropout if dropout_p > 0.                              #
        # - Use nn.functional.dropout(x, p=self.dropout_p,             #
        #   training=self.training)                                    #
        # - Return x unchanged if dropout_p == 0.                      #
        ################################################################
        if self.dropout_p > 0.0:
            return nn.functional.dropout(x, p=self.dropout_p, training=self.training)
        else:
            return x

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the transformer block.

        Args:
            x: Input hidden states of shape [B, T, D]
        Returns:
            Output hidden states of shape [B, T, D]
        """
        ################################################################
        #                     TODO 1.9: YOUR CODE HERE                     #
        # Implement forward pass (pre-norm residual block):            #
        #                                                              #
        # 1. Attention sub-layer (pre-norm + residual):                #
        #    - Apply LayerNorm to input                                #
        #    - Apply MultiHeadAttention with RoPE                      #
        #    - Add residual connection with dropout                    #
        # 2. Feed-forward sub-layer (pre-norm + residual):            #
        #    - Apply LayerNorm to input                                #
        #    - Apply FeedForward network                               #
        #    - Add residual connection with dropout                    #
        ################################################################

        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm)
        attn_out = self.maybe_dropout(attn_out)
        x = x + attn_out

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        ffn_out = self.maybe_dropout(ffn_out)
        x = x + ffn_out

        return x


# =============================================================================
# GPT Model
# =============================================================================

class GPTModel(nn.Module):
    """
    Complete GPT Model with RoPE.

    This model assembles all components into a unified architecture for
    autoregressive language modeling using RoPE for positional encoding.
    """
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize GPT Model.

        Required cfg keys:
            - vocab_size: int
            - emb_dim: int
            - context_length: int
            - n_heads: int
            - n_layers: int
            - drop_rate: float
        """
        super().__init__()
        self.context_length = cfg['context_length']

        ################################################################
        #                     TODO 1.10: YOUR CODE HERE                #
        # Build the GPT model components:                              #
        # 1) Use the embedding layer (token embeddings only)           #
        # 2) Dropout after embedding                                   #
        # 3) Stack of L Transformer blocks (use nn.Sequential)         #
        # 4) Final LayerNorm (pre-logit)                               #
        # 5) Output projection to vocab (nn.Linear(emb_dim, vocab))    #
        # 6) Tie output head weights to input embeddings               #
        #                                                              #
        # Hint: nn.Sequential can make things neater (but optional)    #
        ################################################################

        # NOTE: Weight tying is when we share the weights between the input embedding
        # and the output head, so there's only one set of weights (fewer parameters).


        vocab_size = cfg["vocab_size"]
        emb_dim = cfg["emb_dim"]
        n_layers = cfg["n_layers"]
        drop_rate = cfg["drop_rate"]

        self.embedding = GPTEmbedding(vocab_size=vocab_size, emb_dim=emb_dim, context_length=self.context_length,)
        self.dropout = nn.Dropout(drop_rate)
        self.trf_blocks = nn.Sequential( *[TransformerBlock(cfg) for _ in range(n_layers)])
        self.final_norm = RMSNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.out_head.weight = self.embedding.token_embeddings.weight


    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GPT model.

        Args:
            in_idx: Input token IDs of shape [B, T]
        Returns:
            logits: Output logits of shape [B, T, V]
        """
        B, T = in_idx.shape
        if T > self.context_length:
            raise ValueError(f"Sequence length {T} exceeds context_length {self.context_length}")

        ################################################################
        #                     TODO 1.11: YOUR CODE HERE                     #
        # Forward pass:                                                #
        # 1) Embed the inputs (token embeddings only)                  #
        # 2) Apply dropout                                             #
        # 3) Pass through transformer blocks                           #
        # 4) Apply final LayerNorm                                     #
        # 5) Project to logits via out_head                            #
        # 6) Return logits                                             #
        ################################################################

        x = self.embedding(in_idx)          
        x = self.dropout(x)                 
        x = self.trf_blocks(x)              
        x = self.final_norm(x)              
        logits = self.out_head(x)           

        return logits


# =============================================================================
# Text Generation Functions
# =============================================================================

def generate_new_tokens(model, idx, max_new_tokens, context_size, temperature=1.0):
    """
    Autoregressively generates `max_new_tokens` tokens from the model.

    Args:
        model: The language model
        idx: Starting tensor of shape (batch, seq)
        max_new_tokens: Number of tokens to generate
        context_size: Context window size for the model input
        temperature: Softmax temperature (>0). Lower = more greedy, higher = more random
    Returns:
        idx: The resulting sequence with new tokens appended
    """
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:].to(device)

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]  # Final token in the sequence

        if temperature <= 0:
            # Greedy decoding; avoids division by zero
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature  # Apply temperature
            probas = torch.softmax(logits, dim=-1)
            # Sample from the distribution rather than argmax for more natural randomness
            idx_next = torch.multinomial(probas, num_samples=1)
        # Keep new token on the same device as the running sequence to avoid device mismatch
        idx_next = idx_next.to(idx.device)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_text(start_context: str, tokenizer, model, max_new_tokens, context_size):
    """
    Generate text from a starting context.

    Args:
        start_context: Starting text prompt
        tokenizer: Tokenizer to use for encoding/decoding
        model: GPT model
        max_new_tokens: Number of tokens to generate
        context_size: Context window size
    Returns:
        Generated text string
    """
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    model.eval()
    out = generate_new_tokens(model=model, idx=encoded_tensor,
                              max_new_tokens=max_new_tokens,
                              context_size=context_size)
    print("Output:", out)
    print("Output length:", len(out[0]))
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    return decoded_text


# =============================================================================
# Dataset Classes
# =============================================================================

class GPTDataset(Dataset):
    """
    Dataset for GPT causal language modeling.

    Creates input/target pairs for next-token prediction by sliding a window
    over tokenized documents.
    """
    def __init__(self, docs: list[str], tokenizer: Any, max_length: int, stride: int):
        """
        Initialize GPT Dataset.

        Args:
            docs: List of raw text documents
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
            stride: Step size for sliding window
        """
        ################################################################
        #                     TODO 1.12: YOUR CODE HERE                     #
        # Goal: Build input/target pairs for next-token prediction.    #
        #                                                              #
        # 1) Store args (tokenizer, max_length, stride).               #
        # 2) Encode the entire text into integer token ids.            #
        # 3) Slide a window of size `max_length` over token_ids with   #
        #    step `stride`. For each start index i:                    #
        #       inputs  = token_ids[i : i + max_length]                #
        #       targets = token_ids[i+1 : i + max_length + 1]          #
        # 4) Keep only full windows; convert to torch.long tensors     #
        #    and append to self.input_ids / self.target_ids.           #
        # Notes: This implements causal LM: predict                    #
        #        token t using tokens < t.                             #
        ################################################################

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        # Tokenize per-document (avoids building one giant string, which
        # exhausts memory on large corpora) and separate documents with the
        # tokenizer's EOS token so the model doesn't learn spurious
        # cross-document continuations.
        eos_id = tokenizer.eos_token_id
        token_ids: List[int] = []
        for doc in tqdm(docs, desc="Tokenizing documents"):
            token_ids.extend(tokenizer.encode(doc))
            if eos_id is not None:
                token_ids.append(eos_id)

        n_tokens = len(token_ids)
        print(f"[GPTDataset] n_tokens={n_tokens}, max_length={max_length}, stride={stride}")

        # Store one flat tensor and slice windows on demand in __getitem__,
        # instead of materializing every (overlapping) window up front.
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.window_starts = list(range(0, n_tokens - max_length, stride))

        print(f"[GPTDataset] built {len(self.window_starts)} sequences")


    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample
        Returns:
            Tuple of (input_ids, target_ids)
        """
        ################################################################
        #                     TODO 1.13: YOUR CODE HERE                     #
        # Return the input and target tensors for the given index      #
        ################################################################
        i = self.window_starts[idx]
        inputs = self.token_ids[i : i + self.max_length]
        targets = self.token_ids[i + 1 : i + self.max_length + 1]
        return inputs, targets


class GPTArrowDataset(Dataset):
    """
    Dataset for GPT causal language modeling using pre-packed Arrow datasets.

    This dataset loads pre-processed Arrow datasets where sequences are already
    packed to the maximum length, providing better GPU utilization and faster
    data loading compared to the regular GPTDataset.

    The Arrow dataset should contain:
    - input_ids: List of token IDs (length = max_length)
    - labels: List of target token IDs (length = max_length)
    """
    def __init__(self, arrow_dataset_path: str):
        """
        Initialize GPT Arrow Dataset.

        Args:
            arrow_dataset_path: Path to the Arrow dataset directory
        """
        self.dataset = load_from_disk(arrow_dataset_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a sample from the Arrow dataset.

        Args:
            idx: Index of the sample
        Returns:
            Tuple of (input_ids, labels) as tensors
        """
        example = self.dataset[idx]
        input_ids = torch.tensor(example['input_ids'], dtype=torch.long)
        labels = torch.tensor(example['labels'], dtype=torch.long)
        return input_ids, labels

# =============================================================================
# DataLoader Creation
# =============================================================================

def create_dataloader(txt=None, arrow_dataset_path=None, batch_size=16, max_length=256, stride=128,
                     shuffle=True, drop_last=True, num_workers=0):
    """
    Create a DataLoader for GPT training.

    This function supports two data formats:
    1. **Raw text format**: List of text documents (txt parameter)
    2. **Arrow dataset format**: Pre-packed Arrow dataset (arrow_dataset_path parameter)

    Args:
        txt: List of text documents (for raw text format)
        arrow_dataset_path: Path to Arrow dataset directory (for Arrow format)
        batch_size: Batch size
        max_length: Maximum sequence length (only used for raw text format)
        stride: Step size for sliding window (only used for raw text format)
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of worker processes
    Returns:
        DataLoader instance
    """
    ################################################################
    #                     TODO 1.14: YOUR CODE HERE                     #
    # 1) Check if arrow_dataset_path is provided (Arrow format)     #
    # 2) If Arrow format:                                          #
    #    - Create GPTArrowDataset with arrow_dataset_path          #
    #    - Create DataLoader with the Arrow dataset                 #
    # 3) If raw text format:                                       #
    #    - Initialize GPT tokenizer                                #
    #    - Create GPTDataset with txt, tokenizer, max_length, stride#
    #    - Create DataLoader with the regular dataset              #
    # 4) Return the appropriate DataLoader                          #
    ################################################################
    if arrow_dataset_path is not None:
        dataset = GPTArrowDataset(arrow_dataset_path)
    else:
        if txt is None:
            raise ValueError(
                "create_dataloader requires either `txt` (raw text list) "
                "or `arrow_dataset_path` (Arrow dataset directory)."
            )

        if isinstance(txt, str):
            docs = [txt]
        else:
            docs = txt

        tokenizer = setup_tokenizer()
        dataset = GPTDataset(docs=docs, tokenizer=tokenizer, max_length=max_length, stride=stride)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,)

    return dataloader


# =============================================================================
# Utility Functions
# =============================================================================

def get_optimizer_param_groups(model: nn.Module, weight_decay: float):
    """
    Split model parameters into decay / no-decay groups for AdamW.

    Weight decay should not be applied to norm scales, biases, or the
    (tied) token embedding — only to the 2D projection matrices.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "token_embeddings" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def unwrap_compiled(model: nn.Module) -> nn.Module:
    """Return the underlying module if `model` was wrapped by torch.compile."""
    return getattr(model, "_orig_mod", model)


def strip_compiled_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the `_orig_mod.` prefix torch.compile adds to state-dict keys."""
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix):] if k.startswith(prefix) else k: v
                for k, v in state_dict.items()}
    return state_dict


def setup_tokenizer():
    """
    Load GPT-2 tokenizer and add special tokens.
    Returns the configured tokenizer.
    """
    ###########################################################################
    #                            TODO 1.15: YOUR CODE HERE                    #
    #                                                                         #
    # Implement tokenizer setup:                                              #
    #                                                                         #
    # 1. Load GPT-2 tokenizer using AutoTokenizer.from_pretrained()           #
    # 2. Add pad token if missing                                             #
    # 3. Add special tokens for conversations                                 #
    # 4. Test tokenizer with special tokens                                   #
    # 5. Return configured tokenizer                                          #
    #                                                                         #
    # Proper tokenizer setup is crucial for training!                         #
    ###########################################################################

    # NOTE: Use "<|pad|>" as the special token for padding, if needed
    special_tokens_dict = {
        "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    }

    # Your code here
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    add_tokens = {}

    if tokenizer.pad_token is None:
        add_tokens["pad_token"] = "<|pad|>"

    add_tokens.update(special_tokens_dict)

    if add_tokens:
        tokenizer.add_special_tokens(add_tokens)

    # Right padding matches the collators in this project (they right-pad) and
    # is safe with a causal mask: earlier real tokens can never attend to the
    # padding that follows them. Left padding would put pad tokens *before*
    # the real content, which this model cannot mask out (its forward pass
    # takes no attention mask), so it must be avoided.
    tokenizer.padding_side = "right"

    _ = tokenizer("Example: <|system|> You are a system. "
                           "<|user|> Hi. <|assistant|> Hello. <|end|>")

    return tokenizer