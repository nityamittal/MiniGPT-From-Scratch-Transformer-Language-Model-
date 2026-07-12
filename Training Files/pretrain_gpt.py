"""
Causal language-model pretraining loop for the GPT model in gpt.py.


Usage:
    python pretrain_gpt.py

The script will:
1. Load data from the specified dataset
2. Create train/validation splits
3. Initialize the GPT model
4. Train the model with mixed precision
5. Save checkpoints and log to wandb

Depends on the components implemented in gpt.py:
- GPTEmbedding: token embeddings (RoPE supplies position information)
- MultiHeadAttention: attention mechanism with RoPE
- SwiGLU / FeedForward: position-wise MLP
- TransformerBlock, GPTModel, and dataset classes
"""

import os
import math
import numpy as np
import random
import logging
import argparse
from typing import Optional, Callable, List, Tuple, Dict, Any

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
from torch.amp import autocast, GradScaler

# Data loading imports
from torch.utils.data import Dataset, DataLoader
import json
import glob
import gzip
import bz2
import datetime

# Arrow dataset support
from datasets import load_from_disk

# Tokenization imports
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Progress and timing
from tqdm.auto import tqdm, trange
import time
import wandb

# Import our GPT implementation
import gpt

# Set CuPy/CUDA to allow TF32 computations
# This can provide a speedup on compatible GPUs (RTX 4000 series, etc.)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GPT model')

    # Data arguments
    parser.add_argument('--data_path', type=str,
                       default='data/fineweb-edu-sample-1B.jsonl.gz',
                       help='Path to the training data (JSONL.gz file or Arrow dataset directory)')
    parser.add_argument('--data_format', type=str, choices=['jsonl', 'arrow'], default='jsonl',
                       help='Format of training data: jsonl (for .jsonl/.gz files) or arrow (for arrow datasets)')
    parser.add_argument('--max_docs', type=int, default=None,
                       help='Maximum number of documents to load (for testing, only applies to raw text)')

    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=None,
                       help='Vocabulary size (auto-detected if not specified)')
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
    parser.add_argument('--learning_rate', type=float, default=6e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                       help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=2,
                       help='Maximum number of epochs')
    parser.add_argument('--target_tokens', type=int, default=1_200_000_000,
                       help='Target number of tokens to train on')

    # Validation arguments
    parser.add_argument('--eval_data_path', type=str, default=None,
                       help='Path to validation data')
    parser.add_argument('--eval_data_format', type=str, choices=['jsonl', 'arrow'], default='jsonl',
                       help='Format of validation data: jsonl (for .jsonl/.gz files) or arrow (for arrow datasets)')
    parser.add_argument('--eval_max_docs', type=int, default=None,
                       help='Maximum number of documents to load for validation (only for raw text)')
    parser.add_argument('--eval_max_docs_step', type=int, default=None,
                       help='Maximum number of validation documents to use during step evaluation (None = use all)')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                       help='Validation batch size')


    # Logging and saving
    parser.add_argument('--output_dir', type=str,
                       default='models/pretrained/',
                       help='Output directory for saving models')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='Save model every N steps')
    parser.add_argument('--eval_every', type=int, default=1000,
                       help='Evaluate model every N steps')
    parser.add_argument('--wandb_project', type=str, default='gpt-pretraining',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str,
                       default=f"gpt-pretraining-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                       help='Wandb run name')
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    #saving and resuming progress    
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint .pt file to resume training from')

    parser.add_argument('--resume_wandb_id', type=str, default=None,
                       help='Wandb run id to resume logging into')


    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg):
    """Determine the best available device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    else:
        return device_arg

def get_amp_dtype(device):
    '''Get the appropriate AMP dtype for mixed precision training on the device.'''

    if device.startswith('cuda'):
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device == 'mps':
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32  # or disable autocast on CPU
    return amp_dtype

def load_data(data_path, max_docs=None, data_format='jsonl'):
    """
    Load data from JSONL file or Arrow dataset.

    Args:
        data_path: Path to the data file or Arrow dataset directory
        max_docs: Maximum number of documents to load (only for raw text)
        data_format: Format of the data ('jsonl' or 'arrow')
    Returns:
        List of text documents (for raw text) or None (for Arrow datasets)
    """
    if data_format == 'arrow':
        print(f"Using Arrow dataset from {data_path}")
        # For Arrow datasets, we don't need to load the data here
        # The GPTArrowDataset in gpt.py will handle loading
        return None
    else:
        print(f"Loading data from {data_path}")

        ofunc = gzip.open if data_path.endswith('gz') else open
        docs = []

        with ofunc(data_path, 'rt') as f:
            for i, line in enumerate(tqdm(f, desc="Reading data from file")):
                if max_docs and i >= max_docs:
                    break
                docs.append(json.loads(line)['text'])

        print(f"Loaded {len(docs)} documents")
        return docs


def create_dataloaders(docs, tokenizer, config, args):
    """Create train and validation dataloaders."""
    print("Creating dataloaders...")

    ###########################################################################
    #                                                                         #
    # Implement dataloader creation for training:                           #
    #                                                                         #
    # 1. Check if using Arrow dataset format (args.data_format == 'arrow')   #
    # 2. If Arrow format:                                                     #
    #    - Use gpt.create_dataloader() with arrow_dataset_path=args.data_path #
    #    - Create both train and val loaders using the same Arrow dataset     #
    #    - Note: Arrow datasets are typically pre-split or can be split       #
    # 3. If raw text format:                                                  #
    #    - Split documents into trainvalidation sets (95%/5%)               #
    #    - Create training DataLoader using docs                              #
    #    - Create validation DataLoader using validation docs                 #
    # 4. Print dataset statistics                                            #
    # 5. Return both dataloaders                                              #
    #                                                                         #
    # Proper data splitting is essential for evaluation!                     #
    ###########################################################################

    # Your Code here
    if getattr(args, "data_format", "jsonl") == "arrow":
        full_dataset = gpt.GPTArrowDataset(args.data_path)

        # Hold out 5% for validation — evaluating on the training data itself
        # cannot detect overfitting.
        n_total = len(full_dataset)
        n_val = max(1, int(0.05 * n_total))
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val], generator=generator
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  drop_last=True, num_workers=args.num_workers,)

        eval_bs = getattr(args, "eval_batch_size", args.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=eval_bs, shuffle=False,
                                drop_last=False, num_workers=args.num_workers,)

        print(f"Arrow dataset loaded from: {args.data_path}")
        print(f"Total packed sequences: {n_total} (train={n_train}, val={n_val})")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    else:
        if not docs:
            raise ValueError("No documents were loaded for raw-text training.")

        n_docs = len(docs)
        split_idx = int(0.95 * n_docs)

        if split_idx == n_docs and n_docs > 1:
            split_idx = n_docs - 1

        train_docs = docs[:split_idx]
        val_docs = docs[split_idx:]

        print(f"Total documents: {n_docs}")
        print(f"Train documents: {len(train_docs)}")
        print(f"Validation documents: {len(val_docs)}")

        context_length = config.get("context_length", 1024)

        train_dataset = gpt.GPTDataset(docs=train_docs, tokenizer=tokenizer, max_length=context_length, stride=context_length,)

        val_dataset = gpt.GPTDataset( docs=val_docs, tokenizer=tokenizer, max_length=context_length, stride=context_length,)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers,)

        eval_bs = getattr(args, "eval_batch_size", args.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=eval_bs, shuffle=False, drop_last=False, num_workers=args.num_workers,)

        print(
            f"Train sequences: {len(train_dataset)} "
            f"({len(train_loader)} batches)"
        )
        print(
            f"Validation sequences: {len(val_dataset)} "
            f"({len(val_loader)} batches)"
        )

    print("✅ Dataloaders created")

    return train_loader, val_loader



def evaluate_validation_loss(model, val_loader, loss_fn, device, max_docs=None):
    """Evaluate the model's loss on the validation dataset.

    Args:
        model: The GPT model to evaluate
        val_loader: Validation data loader
        loss_fn: Loss function to use
        device: Device to run evaluation on
        max_docs: Maximum number of validation batches to process (None = use all)
    """
    ###########################################################################
    #                                                                         #
    # Implement validation loss evaluation:                                   #
    #                                                                         #
    # 1. Set model to evaluation mode (model.eval())                          #
    # 2. Initialize loss tracking variables                                   #
    # 3. Iterate through validation batches with torch.no_grad():             #
    #    - Move data to device                                                #
    #    - Forward pass with mixed precision (optional but recommended)       #
    #    - Compute loss and accumulate                                        #
    #    - Stop early if max_docs limit is reached                            #
    # 4. Calculate average validation loss                                    #
    # 5. Set model back to training mode (model.train())                      #
    # 6. Return the average validation loss                                   #
    #                                                                         #
    # Note: max_docs parameter allows limiting validation batches for faster  #
    # step evaluation, while end-of-epoch evaluation uses all validation data #
    # This is crucial for monitoring overfitting during training!             #
    ###########################################################################

    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    device_type = str(device).split(':')[0]
    use_autocast = device_type != "cpu"
    amp_dtype = get_amp_dtype(device_type)

    with torch.no_grad():
        for batch_idx, (input_ids, labels) in enumerate(val_loader):
            if max_docs is not None and batch_idx >= max_docs:
                break

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            if use_autocast:
                with autocast(device_type=device_type, dtype=amp_dtype):
                    logits = model(input_ids)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            batch_tokens = labels.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            num_batches += 1

    if total_tokens == 0:
        avg_loss = float('nan')
    else:
        avg_loss = total_loss / total_tokens

    model.train()

    return avg_loss

def train_model(model, train_loader, val_loader, config, args, resume_state=None):
    """Train the GPT model."""
    device = get_device(args.device)
    device_type = str(device).split(':')[0]
    amp_dtype = get_amp_dtype(device_type)
    print(f"Using device: {device}")

    use_autocast = device_type != "cpu"

    # Move model to device
    model.to(device)
    model_to_save = gpt.unwrap_compiled(model)


    # Initialize training components
    loss_fn = nn.CrossEntropyLoss()

    # No weight decay on norm scales, biases, or the tied embedding
    optimizer = torch.optim.AdamW(
        gpt.get_optimizer_param_groups(model, args.weight_decay),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Creates a learning rate scheduler that first linearly increases the learning rate ("warmup")
    # and then smoothly decreases it following a half-cosine curve for the rest of training.
    # This approach helps stabilize training early on (warmup), then allows learning to slow down gently,
    # which can result in better convergence and prevent the optimizer from overshooting good solutions.
    # The scheduler adjusts the optimizer's learning rate at each step.
    #
    # Just like with the optimizer, we will need tell the scheduler how many warmup steps and
    # how many total steps, and then tell it when to take a step.

    # Gradient accumulation variables

    target_global_batch = 256
    micro_batch = args.batch_size
    accum = max(1, target_global_batch // micro_batch)

    print(f"Starting training...")
    print(f"Gradient accumulation steps: {accum}")


    tokens_per_step = config["context_length"] * micro_batch * accum
    total_steps = math.ceil(args.target_tokens / tokens_per_step)
    warmup_steps = min(400, int(0.02 * total_steps)) 

    print(f"Total training steps (optimizer steps): {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,  # half-cosine
    )

    # Loss scaling is only needed (and only supported) for float16 on CUDA;
    # bfloat16 has enough dynamic range and needs no scaler.
    scaler = GradScaler(enabled=(device_type == "cuda" and amp_dtype == torch.float16))

    opt_step = 0
    global_step = 0
    start_epoch = 0

    if resume_state is not None:
        if resume_state.get("optimizer") is not None:
            print("Loading optimizer state from checkpoint...")
            optimizer.load_state_dict(resume_state["optimizer"])
        if resume_state.get("scheduler") is not None:
            print("Loading scheduler state from checkpoint...")
            scheduler.load_state_dict(resume_state["scheduler"])

        global_step = resume_state.get("step", 0)
        opt_step = global_step
        start_epoch = resume_state.get("epoch", 0)
        print(f"Resuming from global_step={global_step}, epoch={start_epoch}")



    # Initialize wandb
    #
    # NOTE: If you're doing any other customization to your model design, we
    # recommend logging these configuration details to wandb for easier analysis
    # on whether the changes you made are helping or hurting performance.
    wandb_config = {
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "position_embedding": "rope",
        "emb_dim": config["emb_dim"],
        "n_heads": config["n_heads"],
        "n_layers": config["n_layers"],
        "context_length": config["context_length"],
        "drop_rate": config["drop_rate"],
    }
    wandb_kwargs = dict(
        project=args.wandb_project,
        config=wandb_config,
        name=args.wandb_run_name,
    )

    if getattr(args, "resume_wandb_id", None):
        wandb_kwargs["id"] = args.resume_wandb_id
        wandb_kwargs["resume"] = "allow"

    wandb.init(**wandb_kwargs)


    # Training loop
    model.train()
    losses = []

    # Track last executed steps to prevent duplicate evaluation/saving
    last_eval_step = -1
    last_save_step = -1

    max_grad_norm = 1.0       
    stop_training = False

    # Normally, we want to use a large batch size to get better gradient estimates.
    # However, if we use a large batch size, we will run out of memory. Therefore,
    # we'll use a technique called gradient accumulation to simulate a larger batch size.
    # We'll still use batches of a certain size, but we won't call the optimizer.step()
    # after each batch. Instead, we'll accumulate gradients over multiple batches
    # and call the optimizer.step() after a certain number of batches. You'll see the smaller batch size
    # called "micro-batch" in the code (and in practice) and the larger batch size called
    # the effective batch size or macro-batch.
    #
    # GRADIENT ACCUMULATION EXPLANATION:
    # Gradient accumulation allows us to simulate larger batch sizes by:
    # - Computing gradients on smaller "micro-batches"
    # - Accumulating gradients across multiple micro-batches
    # - Only updating parameters after accumulating gradients from 'accum' batches
    # - This enables training with effective batch size = micro_batch_size * accum
    # - Example: micro_batch=32, accum=8 → effective batch_size=256
    # - Benefits: Better gradient estimates, memory efficiency, stable training

    


    ###########################################################################
    # Core training loop:
    # 1. Computing gradients with loss.backward() (scaled by accumulation factor)
    # 2. Gradient clipping to prevent exploding gradients
    # 3. Optimizer step to update parameters (only every 'accum' steps)
    # 4. Learning rate scheduling
    # 5. Zeroing gradients for next iteration
    # 6. Gradient accumulation for larger effective batch sizes
    # 7. Saving the model model according to the save_every step
    # 8. Evaluating the model on the validation set according to
    #    the eval_every step and using the eval_max_docs_step docs (if a validation dataset is provided)
    # 9. Logging the loss to wandb
    # 10. Saving the model at the end of each epoch
    # 11. Logging the full validation loss to wandb at the end of each epoch
    #
    #
    # CORE STEPS:
    # 1. Scale loss by accumulation factor: (loss / accum).backward()
    # 2. Check if we've accumulated enough gradients: if (step + 1) % accum == 0
    # 3. Clip gradients to prevent explosion: torch.nn.utils.clip_grad_norm_()
    # 4. Update parameters: optimizer.step()
    # 5. Update learning rate: scheduler.step()
    # 6. Clear gradients: optimizer.zero_grad()
    # 7. Track optimization steps: opt_step += 1, global_step += 1
    #
    ###########################################################################

    for epoch in trange(start_epoch, args.max_epochs, desc="Epoch"):

        running_loss = torch.zeros((), device=device)
        running_batches = 0

        print(f"Starting epoch {epoch}")
        num_batches = len(train_loader)
        for step, (input_ids, labels) in enumerate(tqdm(train_loader, position=1, leave=True, desc="Step")):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            if use_autocast:
                with autocast(device_type=device_type, dtype=amp_dtype):
                    logits = model(input_ids)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Accumulate on-device; calling .item() every micro-batch forces a
            # GPU sync per step.
            running_loss += loss.detach()
            running_batches += 1

            # Size of the current accumulation group (the last group of an
            # epoch may be smaller than `accum`; dividing by the full `accum`
            # would under-weight it).
            group_start = (step // accum) * accum
            group_size = min(accum, num_batches - group_start)

            loss_to_backward = loss / group_size

            if scaler.is_enabled():
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            is_accum_step = ((step + 1) % accum == 0) or (step + 1 == num_batches)

            if is_accum_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                scheduler.step()

                opt_step += 1
                global_step += 1

                avg_loss = (running_loss / max(1, running_batches)).item()
                losses.append(avg_loss)

                running_loss = torch.zeros((), device=device)
                running_batches = 0

                wandb.log(
                    {
                        "train_loss": avg_loss,
                        "lr": scheduler.get_last_lr()[0],
                        "opt_step": opt_step,
                        "global_step": global_step,
                        "epoch": epoch + (step + 1) / len(train_loader),
                    },
                    step=global_step,
                )

                if (val_loader is not None and args.eval_every > 0 and global_step % args.eval_every == 0
                    and global_step != last_eval_step):
                    val_loss_step = evaluate_validation_loss(model, val_loader, loss_fn, device, max_docs=args.eval_max_docs_step,)

                    print(f"[Step {global_step}] Step validation loss: {val_loss_step:.4f}")
                    wandb.log({"val_loss_step": val_loss_step}, step=global_step)
                    last_eval_step = global_step

                if (args.save_every > 0 and global_step % args.save_every == 0 and global_step != last_save_step):
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint_step_{global_step}.pt")
                    torch.save(
                        {
                            "model_state_dict": model_to_save.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "config": config,
                            "step": global_step,
                            "epoch": epoch,
                        },
                        ckpt_path,
                    )
                    print(f"Saved checkpoint to {ckpt_path}")
                    last_save_step = global_step

                if global_step >= total_steps:
                    print(f"Reached target optimizer steps ({total_steps}). Stopping training.")
                    stop_training = True
                    break

        epoch_ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(
            {
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "step": global_step,
                "epoch": epoch + 1,
            },
            epoch_ckpt_path,
        )
        print(f"Saved end-of-epoch checkpoint to {epoch_ckpt_path}")

        if val_loader is not None:
            val_loss = evaluate_validation_loss(
                model, val_loader, loss_fn, device, max_docs=None
            )
            print(f"[Epoch {epoch + 1}] Full validation loss: {val_loss:.4f}")
            wandb.log(
                {"val_loss": val_loss, "epoch": epoch + 1},
                step=global_step,
            )

        if stop_training:
            break
    
    print("Training completed!")
    wandb.finish()


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    print("Setting up tokenizer...")
    tokenizer = gpt.setup_tokenizer()

    # Determine vocabulary size
    if args.vocab_size is None:
        special_tokens = ["<|user|>", "<|assistant|>", "<|end|>", "<|system|>", "<|pad|>"]
        max_token_id = max(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
        vocab_size = max_token_id + 1
    else:
        vocab_size = args.vocab_size

    print(f"Using vocabulary size: {vocab_size}")

    # Create model configuration based on the user's arguments
    config = {
        "vocab_size": vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": False
    }

    # Load data
    docs = load_data(args.data_path, args.max_docs, args.data_format)

    #loading checkpoint to pass it down 
    resume_state = None
    if args.resume_from is not None:
        print(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        resume_state = {
            "model": ckpt.get("model_state_dict"),
            "optimizer": ckpt.get("optimizer_state_dict"),
            "scheduler": ckpt.get("scheduler_state_dict"),
            "step": ckpt.get("step", 0),
            "epoch": ckpt.get("epoch", 0),
        }

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(docs, tokenizer, config, args)

    device = get_device(args.device)
    print(f"Using device (for compile): {device}")

    ###########################################################################
    #                                                                         #
    # Implement model initialization and setup:                               #
    #                                                                         #
    # 1. Create GPTModel instance with the configuration                      #
    # 2. Move model to the correct device (CPU/GPU)                           #
    # 3. Optionally compile model for better performance                      #
    # 4. Calculate and print parameter counts (optional)                      #
    # 5. Train the model                                                      #
    #                                                                         #
    # NOTE: you probably want mode="default" for the compile mode, but you    #
    #       can experiment with other modes if you want to.                   #
    ###########################################################################

    # your code here
    model = gpt.GPTModel(config)

    # If resuming, load model weights before compile
    if resume_state is not None and resume_state["model"] is not None:
        print("Loading model weights from checkpoint...")
        model.load_state_dict(gpt.strip_compiled_prefix(resume_state["model"]))

    # Compile (GPU) or skip (CPU), as you already do
    device_for_compile = get_device(args.device)
    print(f"Using device (for compile): {device_for_compile}")
    if hasattr(torch, "compile") and device_for_compile.startswith("cuda"):
        try:
            model = torch.compile(model, mode="default")
            print("Compiled model with torch.compile (mode='default').")
        except Exception as e:
            print(f"torch.compile failed, continuing without compilation: {e}")
    else:
        print("Skipping torch.compile on device=", device_for_compile)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} "
          f"({trainable_params/1e6:.2f}M trainable)")

    train_model(model, train_loader, val_loader, config, args, resume_state=resume_state)



if __name__ == "__main__":
    main()
