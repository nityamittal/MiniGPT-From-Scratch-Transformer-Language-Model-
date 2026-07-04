"""
EECS 595 HW3: Supervised Fine-Tuning (SFT) Implementation

This file contains all the core classes and functions needed to implement
Supervised Fine-Tuning (SFT) of GPT models for conversational AI.

Students should implement the TODO sections in each class and function.
"""

import os
import json
import math
import numpy as np
import random
import logging
from typing import Optional, Callable, List, Tuple, Dict, Any, Iterable
from copy import deepcopy
import gzip

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from functools import partial

# Transformers and tokenization
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, default_data_collator

# Data handling
from datasets import load_from_disk
import orjson

# Progress tracking
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import wandb

# Import GPT components from gpt.py
from gpt import (
    GPTEmbedding,
    MultiHeadAttention,
    SwiGLU,
    FeedForward,
    TransformerBlock,
    GPTModel,
    generate_new_tokens,
    generate_text,
    setup_tokenizer,
    strip_compiled_prefix,
)


# =============================================================================
# SFT Dataset Class
# =============================================================================

class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning (SFT) of GPT models on conversational data.

    Key Features:
    1. Loads conversations from jsonlines format
    2. Formats with special tokens: <|user|>content<|end|><|assistant|>content<|end|>
    3. SELECTIVE MASKING: Only trains on <|assistant|> token and first <|end|> after assistant
    4. Masks all other tokens including <|user|>, <|system|>, and other <|end|> tokens
    """
    def __init__(self, data_file: str, tokenizer, max_length: int = 1024):
        """
        Initialize SFT Dataset.

        Args:
            data_file: Path to jsonlines file with conversations
            tokenizer: Tokenizer with special tokens added
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations: List[List[Dict[str, str]]] = []

        # Precompute special token IDs
        self.SID = {
            "user": tokenizer.convert_tokens_to_ids("<|user|>"),
            "asst": tokenizer.convert_tokens_to_ids("<|assistant|>"),
            "sys": tokenizer.convert_tokens_to_ids("<|system|>"),
            "end": tokenizer.convert_tokens_to_ids("<|end|>"),
            "pad": tokenizer.pad_token_id or 0,
        }

        ofunc = gzip.open if data_file.endswith(".gz") else open
        with ofunc(data_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading dataset"):
                if not line.strip():
                    continue
                obj = orjson.loads(line)
                msgs = obj.get("messages")
                if isinstance(msgs, list) and msgs:
                    self.conversations.append(msgs)

    def __len__(self):
        return len(self.conversations)

    def _build_ids_labels(self, conversation: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build input IDs and labels for a conversation with proper masking.

        Args:
            conversation: List of message dicts

        Returns:
            Tuple of (input_ids, labels) tensors
        """
        ###########################################################################
        #                            TODO 3.1: YOUR CODE HERE                         #
        #                                                                         #
        # Implement fast tokenization with selective masking:                    #
        #                                                                         #
        # 1. Initialize empty lists for ids and labels                           #
        # 2. Iterate through each message in the conversation                     #
        # 3. For each message, extract role and content                           #
        # 4. Based on role, append tokens and labels:                             #
        #    - assistant: Add <|assistant|> token (train), content tokens (train), #
        #                first <|end|> token (train)                               #
        #    - user: Add <|user|> token (mask), content tokens (mask),           #
        #            <|end|> token (mask)                                         #
        #    - system: Add <|system|> token (mask), content tokens (mask),       #
        #              <|end|> token (mask)                                        #
        # 5. Use self.SID dictionary for special token IDs                        #
        # 6. Use tokenizer.encode(text, add_special_tokens=False) for content    #
        # 7. Truncate if sequence exceeds max_length                              #
        # 8. Return as torch tensors                                               #
        #                                                                         #
        ###########################################################################

        ids: List[int] = []
        labels: List[int] = []

        for msg in conversation:
            role = str(msg.get("role", "")).lower()
            text = msg.get("content", "")

            if role == "assistant":
                start_id = self.SID["asst"]
                train_segment = True
            elif role == "system":
                start_id = self.SID["sys"]
                train_segment = False
            else:
                start_id = self.SID["user"]
                train_segment = False

            content_ids = self.tokenizer.encode(text, add_special_tokens=False)

            seg_ids = [start_id] + content_ids + [self.SID["end"]]

            if train_segment:
                seg_labels = list(seg_ids)
            else:
                seg_labels = [-100] * len(seg_ids)

            remaining = self.max_length - len(ids)
            if remaining <= 0:
                break

            if len(seg_ids) > remaining:
                # Truncate, but keep <|end|> as the final token so the model
                # still learns that every segment terminates — otherwise
                # truncated assistant turns teach it to never emit <|end|>.
                seg_ids = seg_ids[:remaining]
                seg_labels = seg_labels[:remaining]
                seg_ids[-1] = self.SID["end"]
                seg_labels[-1] = self.SID["end"] if train_segment else -100

            ids.extend(seg_ids)
            labels.extend(seg_labels)

        if not ids:
            ids = [self.SID["pad"]]
            labels = [-100]

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return input_ids, labels_tensor

    def __getitem__(self, idx):
        return self._build_ids_labels(self.conversations[idx])


# =============================================================================
# Data Collators
# =============================================================================

def sft_data_collator(batch, pad_token_id: int = 0):
    """
    Custom data collator for SFT dataset that handles tuple format.

    Args:
        batch: List of (input_ids, labels) tuples
        pad_token_id: Token ID used to right-pad input_ids (pass the
            tokenizer's actual pad token id, e.g. via functools.partial)

    Returns:
        Dictionary with batched input_ids and labels
    """
    ###########################################################################
    #                            TODO 3.2: YOUR CODE HERE                         #
    #                                                                         #
    # Implement SFT data collator for batching:                               #
    #                                                                         #
    # 1. Separate input_ids and labels from batch tuples                      #
    # 2. Find the maximum length in the batch                                #
    # 3. Pad all sequences to the same length:                               #
    #    - Pad input_ids with pad_token_id (usually 0)                       #
    #    - Pad labels with -100 (masked)                                     #
    # 4. Stack into batch tensors                                            #
    # 5. Return dictionary with 'input_ids' and 'labels' keys                #
    #                                                                         #
    # This ensures all sequences in a batch have the same length.            #
    ###########################################################################

    input_ids_list, labels_list = zip(*batch)

    max_len = max(t.size(0) for t in input_ids_list)

    pad_label_id = -100

    padded_inputs = []
    padded_labels = []

    for ids, labs in zip(input_ids_list, labels_list):
        cur_len = ids.size(0)
        pad_len = max_len - cur_len

        if pad_len > 0:
            ids = torch.cat(
                [ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)],
                dim=0,
            )
            labs = torch.cat(
                [labs, torch.full((pad_len,), pad_label_id, dtype=torch.long)],
                dim=0,
            )

        padded_inputs.append(ids)
        padded_labels.append(labs)

    batch_input_ids = torch.stack(padded_inputs, dim=0)
    batch_labels = torch.stack(padded_labels, dim=0)

    return {
        "input_ids": batch_input_ids,
        "labels": batch_labels,
    }


def hf_collate(examples):
    """
    HuggingFace-style collator for packed datasets.

    This collator is designed for pre-packed Arrow datasets where each example
    already contains a full sequence of input_ids and labels. Unlike regular
    datasets that need padding, packed datasets have sequences that are already
    the correct length (max_length).

    The packed format provides several advantages:
    - No padding needed: sequences are already max_length
    - Better GPU utilization: every token contributes to training
    - Faster data loading: Arrow format is optimized for speed
    - Memory efficiency: supports memory mapping for large datasets

    Args:
        examples: List of examples from packed dataset, each containing:
                 - input_ids: List of token IDs (length = max_length)
                 - labels: List of labels with -100 for masked tokens

    Returns:
        Dictionary with batched data:
        - input_ids: Tensor of shape (batch_size, max_length)
        - labels: Tensor of shape (batch_size, max_length)
        - attention_mask: Tensor indicating non-padding tokens
    """
    ids = torch.tensor(np.stack([e["input_ids"] for e in examples]), dtype=torch.long)
    labs = torch.tensor(np.stack([e["labels"] for e in examples]), dtype=torch.long)
    # Packed sequences contain no padding, so the mask is all ones. Deriving
    # it from `ids != 0` would mislabel genuine tokens with id 0 ("!") as
    # padding. Note: GPTModel does not consume this mask; it is provided for
    # compatibility with HF-style training loops only.
    attn = torch.ones_like(ids)
    return {"input_ids": ids, "labels": labs, "attention_mask": attn}


# =============================================================================
# Text Generation Functions
# =============================================================================

def generate_chat_response(model, tokenizer, user_message, max_new_tokens=100, temperature=0.7, context: Optional[str] = ""):
    """
    Generate a conversational response using the fine-tuned model.

    Args:
        model: Fine-tuned GPT model
        tokenizer: Tokenizer with special tokens
        user_message: User's input message
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)

    Returns:
        Generated response text
    """
    ##############################################################################
    #                            TODO 3.3: YOUR CODE HERE                        #
    #                                                                            #
    # Implement conversational text generation:                                  #
    #                                                                            #
    # 1. Format input with special tokens: "<|user|>message<|end|><|assistant|>" #
    # 2. Tokenize and move to device                                             #
    # 3. Generate tokens autoregressively:                                       #
    #    - Get model output (logits)                                             #
    #    - Apply temperature scaling and sample next token                       #
    #    - Stop if <|end|> token or max length reached                           #
    # 4. Decode and extract assistant's response                                 #
    #                                                                            #
    # This enables conversational AI by generating responses in chat format.     #
    ##############################################################################

    if context:
        prompt = context + "<|assistant|>"
    else:
        prompt = f"<|user|>{user_message}<|end|><|assistant|>"

    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    asst_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end|>")

    context_size = getattr(model, "context_length", input_tensor.size(1))

    was_training = model.training
    model.eval()

    generated = input_tensor

    with torch.no_grad():
        for _ in range(max_new_tokens):

            idx_cond = generated[:, -context_size:]

            logits = model(idx_cond)                
            logits = logits[:, -1, :]              

            temp = temperature if temperature > 0 else 1.0
            logits = logits / temp

            probs = torch.softmax(logits, dim=-1) 
            next_token = torch.multinomial(probs, num_samples=1)  
            next_id = next_token.item()

            generated = torch.cat([generated, next_token.to(device)], dim=1)

            if next_id == end_id:
                break

    if was_training:
        model.train()

    all_ids = generated[0].tolist()

    try:
        start_idx = max(i for i, tid in enumerate(all_ids) if tid == asst_id)
    except ValueError:
        start_idx = len(input_ids) - 1

    end_idx = len(all_ids)
    for i in range(start_idx + 1, len(all_ids)):
        if all_ids[i] == end_id:
            end_idx = i
            break

    reply_ids = all_ids[start_idx + 1:end_idx]
    response = tokenizer.decode(reply_ids).strip()

    return response

def generate_multi_turn_response(model, tokenizer, conversation_history, max_new_tokens=100, temperature=0.7):
    """
    Generate a response considering the full conversation history.

    Args:
        model: Fine-tuned GPT model
        tokenizer: Tokenizer with special tokens
        conversation_history: List of dicts with 'role' and 'content' keys
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated response text
    """
    # Format the conversation history with special tokens (excluding the last user message)
    context = ""
    last_user_message = ""

    for message in conversation_history:
        role = message['role']
        content = message['content']

        if role == 'user':
            context += f"<|user|>{content}<|end|>"
            last_user_message = content  # Keep track of the last user message
        elif role == 'assistant':
            context += f"<|assistant|>{content}<|end|>"
        elif role == 'system':
            context += f"<|system|>{content}<|end|>"

    # Use generate_chat_response with the conversation context
    return generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        user_message=last_user_message,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        context=context
    )


# =============================================================================
# Utility Functions
# =============================================================================

def load_pretrained_model(model_path: str, config: Dict[str, Any]):
    """
    Load a pre-trained GPT model from checkpoint.

    Args:
        model_path: Path to the model checkpoint
        config: Model configuration dictionary

    Returns:
        Loaded GPT model
    """
    print(f"Loading pre-trained model from {model_path}...")
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

    # Accept both raw state dicts and the wrapper dicts saved by the
    # pretraining script ({"model_state_dict": ..., "optimizer_state_dict": ...}).
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    # Handle checkpoints accidentally saved from a torch.compile'd model
    state_dict = strip_compiled_prefix(state_dict)

    # Create model with correct configuration
    model = GPTModel(config)

    # Check if resizing is needed
    original_vocab_size = state_dict['embedding.token_embeddings.weight'].shape[0]
    new_vocab_size = config['vocab_size']

    if original_vocab_size != new_vocab_size:
        print(f"❌ ERROR: Vocabulary size mismatch!")
        print(f"   Model vocab size: {original_vocab_size}")
        print(f"   Expected vocab size: {new_vocab_size}")
        print(f"   Please use the corrected model from the pre-training notebook!")
        raise ValueError("Vocabulary size mismatch - use the corrected model")

    # Load the state dict strictly — a silent partial load would leave the
    # model randomly initialized while reporting success.
    model.load_state_dict(state_dict, strict=True)
    print(f"✅ Model loaded successfully!")
    print(f"✅ Model vocabulary size: {model.embedding.token_embeddings.weight.shape[0]}")

    return model


def evaluate_validation_loss(model, val_loader, loss_fn, device):
    """
    Evaluate the model's loss on the validation dataset.

    Args:
        model: The GPT model
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run evaluation on

    Returns:
        Average validation loss
    """
    ###########################################################################
    #                            TODO 3.4: YOUR CODE HERE                     #
    #                                                                         #
    # Implement validation loss evaluation for SFT:                           #
    #                                                                         #
    # 1. Set model to evaluation mode (model.eval())                          #
    # 2. Initialize loss tracking variables                                   #
    # 3. Iterate through validation batches with torch.no_grad():             #
    #    - Handle different batch formats (dict vs tuple)                     #
    #    - Move data to device                                                #
    #    - Forward pass with mixed precision (optional)                       #
    #    - Compute loss and accumulate                                        #
    # 4. Calculate average validation loss                                    #
    # 5. Set model back to training mode (model.train())                      #
    # 6. Return the average validation loss                                   #
    #                                                                         #
    # This is crucial for monitoring SFT training progress!                   #
    ###########################################################################

    model.eval()

    total_loss = 0.0
    total_tokens = 0  


    device_str = str(device)
    if device_str.startswith("cuda"):
        device_type = "cuda"
    elif device_str.startswith("mps"):
        device_type = "mps"
    else:
        device_type = "cpu"

    use_autocast = device_type != "cpu"
    if device_type == "cuda":
        amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    elif device_type == "mps":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32  

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
            elif isinstance(batch, (list, tuple)):
                input_ids, labels = batch[0], batch[1]
            else:
                input_ids = batch.input_ids
                labels = batch.labels

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            if use_autocast:
                with autocast(device_type=device_type, dtype=amp_dtype, enabled=True):
                    logits = model(input_ids)
            else:
                logits = model(input_ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )


            valid_tokens = (shift_labels != -100).sum().item()
            if valid_tokens == 0:
                continue

            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    if total_tokens == 0:
        avg_loss = float("nan")
    else:
        avg_loss = total_loss / total_tokens

    model.train()

    return avg_loss
    


def create_sft_dataloader(data_file: str, tokenizer, batch_size: int = 16,
                         max_length: int = 1024, shuffle: bool = True,
                         drop_last: bool = True, num_workers: int = 0,
                         use_packed: bool = False):
    """
    Create a DataLoader for SFT training.

    This function supports two data formats:
    1. **Regular format** (use_packed=False): Individual conversations in jsonlines format
    2. **Packed format** (use_packed=True): Pre-processed Arrow dataset with packed sequences

    Packed datasets are more efficient for training because they:
    - Maximize GPU utilization by filling sequences to max_length
    - Reduce padding overhead
    - Enable faster data loading with Arrow format
    - Support memory mapping for large datasets

    Args:
        data_file: Path to data file (jsonlines or packed dataset)
        tokenizer: Tokenizer with special tokens
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        num_workers: Number of worker processes
        use_packed: Whether to use packed dataset format

    Returns:
        DataLoader instance
    """
    ###########################################################################
    #                            TODO 3.5: YOUR CODE HERE                     #
    #                                                                         #
    # Implement SFT DataLoader creation:                                      #
    #                                                                         #
    # 1. Check if use_packed is True or False                                 #
    # 2. If use_packed=True:                                                  #
    #    - Use load_from_disk() to load packed dataset                        #
    #    - Create DataLoader with hf_collate function                         #
    # 3. If use_packed=False:                                                 #
    #    - Create SFTDataset instance with data_file, tokenizer, max_length   #
    #    - Create DataLoader with sft_data_collator function                  #
    # 4. Print success message and return DataLoader                          #
    #                                                                         #
    # This provides flexibility to use either regular or packed datasets.     #
    ###########################################################################

    if use_packed:
        # Packed Arrow-style dataset on disk
        print(f"Loading packed SFT dataset from: {data_file}")
        dataset = load_from_disk(data_file)

        # HuggingFace Dataset works directly with PyTorch DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=hf_collate,
        )

        print(
            f"Packed SFT DataLoader created: "
            f"{len(dataset)} sequences, {len(dataloader)} batches (batch_size={batch_size})"
        )
    else:
        # Regular jsonl/jsonl.gz conversational data
        print(f"Loading SFT jsonlines dataset from: {data_file}")
        dataset = SFTDataset(
            data_file=data_file,
            tokenizer=tokenizer,
            max_length=max_length,
        )

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=partial(sft_data_collator, pad_token_id=pad_id),
        )

        print(
            f"SFT DataLoader created: "
            f"{len(dataset)} conversations, {len(dataloader)} batches (batch_size={batch_size})"
        )

    return dataloader