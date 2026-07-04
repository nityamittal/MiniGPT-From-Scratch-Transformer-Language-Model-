"""
EECS 595 HW3: SFT Training Script

This script contains the complete training loop for supervised fine-tuning (SFT) of GPT models.
Students need to implement the core components in sft.py before running this script.

Usage:
    python sft_gpt.py

The script will:
1. Load a pre-trained GPT model
2. Load SFT conversation data
3. Fine-tune the model with masked loss computation
4. Save checkpoints and log to wandb

TODO: Students need to implement the following components in sft.py:
- SFTDataset: Load and format conversational data with proper token masking
- SFTDatasetFast: Fast tokenization version for better performance
- Data collators: Handle batching for SFT training
- Generation functions: Conversational text generation
- Utility functions: Model loading and validation
"""

import os
import json
import math
import numpy as np
import random
import logging
import argparse
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

# Transformers and tokenization
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, default_data_collator

# Data handling
from datasets import load_from_disk
import orjson

# Progress tracking
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import wandb

# Import our implementations
import gpt
import sft
from gpt import setup_tokenizer as gpt_setup_tokenizer

# Set CuPy/CUDA to allow TF32 computations
# This can provide a speedup on compatible GPUs (RTX 4000 series, etc.)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GPT model with SFT')

    # Data arguments
    parser.add_argument('--train_data_path', type=str,
                       default='data/smol-smoltalk-train.jsonl.gz',
                       help='Path to training data')
    parser.add_argument('--train_data_format', type=str, choices=['jsonl', 'arrow'], default='jsonl',
                       help='Format of training data: jsonl (for .jsonl/.gz files) or arrow (for arrow datasets)')
    parser.add_argument('--val_data_format', type=str, choices=['jsonl', 'arrow'], default='jsonl',
                       help='Format of validation data: jsonl (for .jsonl/.gz files) or arrow (for arrow datasets)')
    parser.add_argument('--model_path', type=str,
                       default='models/pretrained/gpt.1B-18000-step.model.pth',
                       help='Path to pre-trained model')

    # Validation arguments
    parser.add_argument('--val_data_path', type=str,
                       default='data/smol-smoltalk-dev.jsonl.gz',
                       help='Path to validation data')
    parser.add_argument('--eval_max_docs', type=int, default=None,
                       help='Maximum number of documents to load for validation (only for raw text)')
    parser.add_argument('--eval_max_docs_step', type=int, default=None,
                       help='Maximum number of validation documents to use during step evaluation (None = use all)')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                       help='Validation batch size')

    # Model arguments
    parser.add_argument('--context_length', type=int, default=1024,
                       help='Context length')
    parser.add_argument('--emb_dim', type=int, default=512,
                       help='Embedding dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--drop_rate', type=float, default=0.1,
                       help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=3,
                       help='Maximum number of epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Warmup steps')

    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Evaluation arguments
    parser.add_argument('--eval_every', type=int, default=500,
                       help='Evaluate every N steps')


    # Logging and saving
    parser.add_argument('--output_dir', type=str,
                       default='models/sft/',
                       help='Output directory for models')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='Save model every N steps')
    parser.add_argument('--wandb_project', type=str, default='gpt-sft',
                       help='Wandb project name')

    # Data arguments
    parser.add_argument('--max_train_docs', type=int, default=None,
                       help='Maximum number of documents to load (for testing)')

    return parser.parse_args()


def setup_device(device_arg):
    """Set up the device for training."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = device_arg

    print(f"Using device: {device}")
    return device


def setup_tokenizer():
    """Set up tokenizer with special tokens for SFT."""
    # Call the setup_tokenizer function from gpt.py
    tokenizer = gpt_setup_tokenizer()

    # Calculate actual vocabulary size
    special_tokens = ["<|user|>", "<|assistant|>", "<|end|>", "<|system|>", "<|pad|>"]
    max_token_id = max(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    actual_vocab_size = max_token_id + 1

    print(f"✅ Tokenizer initialized with {actual_vocab_size} tokens")
    return tokenizer, actual_vocab_size


def create_model_config(args, vocab_size):
    """Create model configuration."""
    config = {
        "vocab_size": vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": False
    }
    return config


def load_model(model_path, config):
    """Load pre-trained model."""
    # Delegates to sft.load_pretrained_model, which handles both raw state
    # dicts and pretraining wrapper checkpoints, strips torch.compile
    # prefixes, and loads strictly.
    return sft.load_pretrained_model(model_path, config)


def create_dataloaders(args, tokenizer):
    """Create training and validation dataloaders."""
    print("Creating dataloaders...")

    ###########################################################################
    #                            TODO 4.1: YOUR CODE HERE                     #
    #                                                                         #
    # Implement SFT DataLoader creation:                                      #
    #                                                                         #
    # 1. Create training DataLoader using sft.create_sft_dataloader():        #
    #    - Use args.train_data_path for data file                             #
    #    - Pass tokenizer, batch_size, context_length                         #
    #    - Set shuffle=True, drop_last=True for training                      #
    #    - Use args.num_workers and args.use_packed                           #
    # 2. Create validation DataLoader using sft.create_sft_dataloader():      #
    #    - Use args.val_data_path for data file                               #
    #    - Pass tokenizer, batch_size, context_length                         #
    #    - Set shuffle=False, drop_last=False for validation                  #
    #    - Use args.num_workers and args.use_packed                           #
    # 3. Print success messages with batch counts                             #
    # 4. Return both dataloaders                                              #
    #                                                                         #
    # This function sets up the data pipeline for SFT training.               #
    ###########################################################################
    train_use_packed = (args.train_data_format == "arrow")
    val_use_packed = (args.val_data_format == "arrow")

    train_loader = sft.create_sft_dataloader(data_file=args.train_data_path, tokenizer=tokenizer,
        batch_size=args.batch_size, max_length=args.context_length, shuffle=True, drop_last=True, 
        num_workers=args.num_workers, use_packed=train_use_packed,)

    eval_bs = getattr(args, "eval_batch_size", args.batch_size)
    val_loader = sft.create_sft_dataloader(data_file=args.val_data_path, tokenizer=tokenizer,
        batch_size=eval_bs, max_length=args.context_length, shuffle=False,
        drop_last=False, num_workers=args.num_workers, use_packed=val_use_packed,)

    print( f"Train dataloader created from {args.train_data_path} "
           f"(format={args.train_data_format}, packed={train_use_packed})")
    print(f"  → {len(train_loader)} train batches")

    print(f"Val dataloader created from {args.val_data_path} "
        f"(format={args.val_data_format}, packed={val_use_packed})")
    print(f"  → {len(val_loader)} validation batches")

    print("SFT dataloaders ready")
    return train_loader, val_loader

    


def train_model(model, train_loader, val_loader, args, device):
    """Train the model with SFT."""
    print("Starting SFT training...")

    # Move model to device
    model = model.to(device)
    # Save the underlying module, not the torch.compile wrapper — the wrapper
    # prefixes every state-dict key with `_orig_mod.`, which breaks reloading.
    model_to_save = gpt.unwrap_compiled(model)

    # Loss function - automatically ignores -100 labels
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Optimizer (no weight decay on norm scales, biases, or the tied embedding)
    optimizer = torch.optim.AdamW(
        gpt.get_optimizer_param_groups(model, args.weight_decay),
        lr=args.learning_rate,
        betas=(0.9, 0.95)
    )

    # Learning rate scheduler. The optimizer also steps on the (possibly
    # partial) final accumulation group of each epoch, so count with ceil.
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.max_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5
    )

    # Mixed precision setup. A loss scaler is only needed (and only
    # supported) for float16 on CUDA; bfloat16 needs no scaler.
    device_type = str(device).split(':')[0]
    if device_type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32
    scaler = GradScaler(enabled=(device_type == "cuda" and amp_dtype == torch.float16))

    # Training tracking
    train_losses = []
    val_losses = []
    step = 0
    opt_step = 0
    global_step = 0  # Track total steps across all epochs

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config=vars(args)
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    model.train()
    last_save_step = -1

    last_eval_step = -1
    max_grad_norm = 1.0


    for epoch in trange(args.max_epochs, desc="Epoch"):
        epoch_losses = []
        running_loss = 0.0
        running_batches = 0
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):

            ###########################################################################
            #                            TODO 4.2: YOUR CODE HERE                     #
            #                                                                         #
            # Implement SFT forward pass:                                             #
            #                                                                         #
            # 1. Handle different batch formats (dict vs tuple)                      #
            # 2. Move input_ids and labels to the correct device                      #
            # 3. Forward pass with mixed precision:                                   #
            #    - Use torch.amp.autocast() for mixed precision                       #
            #    - Call model(input_ids) to get logits                               #
            #    - Shift logits and labels for next-token prediction                  #
            #    - Compute loss using CrossEntropyLoss with ignore_index=-100         #
            #    - Scale loss by gradient accumulation steps                          #
            #                                                                         #
            # Key insight: SFT uses masked loss where only assistant tokens           #
            # contribute to the loss (labels != -100).                               #
            #                                                                         #
            # Example:                                                                #
            # input_ids: [<|user|>, "Hello", <|end|>, <|assistant|>, "Hi", <|end|>]  #
            # labels:    [-100,     -100,    -100,    50257,        50258, 50259]   #
            # Only tokens 50257, 50258, 50259 contribute to loss!                    #
            ###########################################################################

            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
            elif isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    input_ids, labels = batch
                else:
                    input_ids = batch[0]
                    labels = batch[1]
            else:
                raise TypeError(f"Unexpected batch type: {type(batch)}")

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            if device_type == "cuda":
                with autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(input_ids)  # [B, T, V]


                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()

                    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),)
            else:
                logits = model(input_ids)  # [B, T, V]

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),)

            # Size of the current accumulation group (the last group of an
            # epoch may be smaller; dividing by the full accumulation count
            # would under-weight it).
            accum_steps = args.gradient_accumulation_steps
            group_start = (batch_idx // accum_steps) * accum_steps
            group_size = min(accum_steps, num_batches - group_start)

            unscaled_loss = loss.item()
            loss = loss / group_size



            ###########################################################################
            #                            TODO 4.3: YOUR CODE HERE                     #
            #                                                                         #
            # Implement SFT backward pass and optimization:                           #
            #                                                                         #
            # 1. Compute gradients with loss.backward()                               #
            # 2. Update weights only every gradient_accumulation_steps:               #
            #    - Apply gradient clipping with torch.nn.utils.clip_grad_norm_()      #
            #    - Call optimizer.step() to update parameters                         #
            #    - Call scheduler.step() to update learning rate                      #
            #    - Call optimizer.zero_grad() to clear gradients                      #
            #    - Increment opt_step counter                                         #
            # 3. Track loss for logging                                               #
            # 4. Evaluate the model on the validation set every eval_every steps      #
            # 5. Save the model every save_every steps                                #
            # 6. Evaluate the model on the validation set at the end of each epoch    #
            # 7. Save the model at the end of each epoch                              #
            #                                                                         #
            #                                                                         #
            # Example with gradient_accumulation_steps=4:                             #
            # - Steps 1-3: Only compute gradients (no optimizer.step())               #
            # - Step 4: Apply gradients and update parameters                         #
            ###########################################################################

            # code goes here

            running_loss += unscaled_loss
            running_batches += 1

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            is_accum_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == num_batches)

            if is_accum_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                opt_step += 1
                global_step += 1

                avg_loss = running_loss / max(1, running_batches)
                epoch_losses.append(avg_loss)
                train_losses.append(avg_loss)

                running_loss = 0.0
                running_batches = 0

                wandb.log(
                    {
                        "train_loss": avg_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "opt_step": opt_step,
                        "global_step": global_step,
                        "epoch": epoch + (batch_idx + 1) / num_batches,
                    },
                    step=global_step,
                )

                if (
                    val_loader is not None
                    and args.eval_every > 0
                    and global_step % args.eval_every == 0
                    and global_step != last_eval_step
                ):
                    val_loss_step = sft.evaluate_validation_loss(
                        model, val_loader, loss_fn, device
                    )
                    print(f"[Step {global_step}] Step validation loss: {val_loss_step:.4f}")
                    wandb.log({"val_loss_step": val_loss_step}, step=global_step)
                    last_eval_step = global_step

                if (
                    args.save_every > 0
                    and global_step % args.save_every == 0
                    and global_step != last_save_step
                ):
                    ckpt_path = os.path.join(
                        args.output_dir,
                        f"sft_checkpoint_step_{global_step}.pth",
                    )
                    torch.save(model_to_save.state_dict(), ckpt_path)
                    print(f"Saved checkpoint: {ckpt_path}")
                    last_save_step = global_step


        # code goes here

        if epoch_losses:
            epoch_train_loss = sum(epoch_losses) / len(epoch_losses)
        else:
            epoch_train_loss = float("nan")

        print(f"Epoch {epoch + 1} finished. Avg train loss: {epoch_train_loss:.4f}")
        wandb.log(
            {
                "epoch_train_loss": epoch_train_loss,
                "epoch": epoch + 1,
            },
            step=global_step,
        )

        if val_loader is not None:
            val_loss = sft.evaluate_validation_loss(model, val_loader, loss_fn, device)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1} validation loss: {val_loss:.4f}")
            wandb.log(
                {
                    "val_loss": val_loss,
                    "epoch": epoch + 1,
                },
                step=global_step,
            )

        epoch_ckpt_path = os.path.join(
            args.output_dir,
            f"sft_epoch_{epoch + 1}.pth",
        )
        torch.save(model_to_save.state_dict(), epoch_ckpt_path)
        print(f"Saved end-of-epoch checkpoint: {epoch_ckpt_path}")


    # code goes here
    final_model_path = os.path.join(args.output_dir, "sft_final_model.pth")
    torch.save(model_to_save.state_dict(), final_model_path)
    print(f"Saved final SFT model to {final_model_path}")


    wandb.finish()
    print("✅ SFT training complete!")


def main():
    """Main training function."""
    args = parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set up device
    device = setup_device(args.device)

    # Set up tokenizer
    tokenizer, vocab_size = setup_tokenizer()

    # Create model configuration
    config = create_model_config(args, vocab_size)

    ###########################################################################
    #                            TODO 4.4: YOUR CODE HERE                     #
    #                                                                         #
    # Implement model loading and setup:                                      #
    #                                                                         #
    # 1. Load pre-trained GPT model from checkpoint                           #
    # 2. Move model to the correct device (CPU/GPU)                           #
    # 3. Optionally compile model for better performance                      #
    # 4. Verify model is ready for SFT training                               #
    #                                                                         #
    # This ensures your pre-trained model is properly loaded!                 #
    ###########################################################################

    model = load_model(args.model_path, config)

    print(f"Model loaded from checkpoint: {args.model_path}")
    if hasattr(torch, "compile") and device == "cuda":
        try:
            print("Compiling model with torch.compile (mode='default')...")
            model = torch.compile(model, mode="default")
            print("✅ torch.compile succeeded.")
        except Exception as e:
            print(f"⚠️ torch.compile failed, continuing without compilation: {e}")
    else:
        print(f"Skipping torch.compile (device='{device}')")

    train_loader, val_loader = create_dataloaders(args, tokenizer)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model ready for SFT: {total_params:,} parameters "
        f"({trainable_params/1e6:.2f}M trainable)"
    )

    train_model(model, train_loader, val_loader, args, device)



if __name__ == "__main__":
    main()
