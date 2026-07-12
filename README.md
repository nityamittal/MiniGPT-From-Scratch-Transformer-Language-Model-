# MiniGPT: From-Scratch GPT-Style Language Model

This repo contains my implementation of a small GPT-style decoder-only Transformer in PyTorch.  
I built the model, pretrained it on a web text corpus, then did supervised fine-tuning (SFT) to turn it into a chatbot and evaluated it on a multiple-choice QA benchmark.

## Features

- **From-scratch GPT-style architecture**
  - Token & positional embeddings
  - Multi-head self-attention with causal masking
  - Transformer blocks with feed-forward MLP + residual connections + normalization
  - Final language modeling head for next-token prediction

- **Pretraining (Causal Language Modeling)**
  - Trained on a subset of the **FineWeb-Edu** corpus (≈1B tokens)
  - Standard next-token prediction objective (cross-entropy loss)

- **Supervised Fine-Tuning (SFT) for Chatbot Behavior**
  - Trained on a multi-turn conversation dataset derived from **SmolTalk**
  - Uses special formatting for system / user / assistant roles
  - Loss applied only on assistant tokens (SFT-style training)

- **Evaluation**
  - Simple multiple-choice QA evaluation script
  - Runs several passes and reports accuracy by difficulty/topic
  - Includes answer-parsing logic to extract model’s chosen option

- **Experiment Tracking & Checkpoints**
  - Training runs logged to Weights & Biases (W&B)
  - Final model weights saved as `.pt` checkpoint
  - Designed to run on Google Colab GPU

---

## Project Structure

- `gpt.py`  
  Core model:
  - Embeddings
  - Multi-head attention
  - Transformer blocks
  - GPT wrapper + forward pass

- `pretrain_gpt.py`  
  - Data loading for web-text corpus
  - Dataloaders for tokenized sequences
  - Training loop for causal LM pretraining
  - Checkpoint saving + logging

- `sft.py`  
  - Dataset classes for chat-style SFT data
  - Formatting of system / user / assistant messages
  - Logic to mask loss so it only applies to assistant tokens

- `sft_gpt.py`  
  - SFT training loop (starting from pretrained weights)
  - Dataloaders for conversation data
  - Checkpoint saving + logging

- `score_gpt.py` / `evaluate_model.sh`  
  - Evaluation script for multiple choice questions
  - Runs model multiple times and aggregates scores

- `InteractiveGeneration.ipynb`  
  - Notebook for free-form text generation from prompts using the pretrained model.

- `ChatWithGPT.ipynb`  
  - Notebook for interactive chat using the SFT’d chatbot model.

- `README.md`  
  - You’re reading it :)

---

## Requirements

Core stack:

- Python 3.x
- PyTorch
- `tokenizers` / Hugging Face-style tokenizer
- `datasets` (for Arrow / parquet-style data)
- Weights & Biases (`wandb`) – optional but recommended
- TQDM, NumPy, etc.

Install (example):

```bash
pip install torch datasets tokenizers wandb tqdm
```

---

## Pretrained Artifacts

Model checkpoints and the tokenized training data are hosted on Google Drive
(too large for git):

- **Model checkpoints:** https://drive.google.com/drive/folders/11JrmxAUYGynGHVFb_AhhyrAFQp_IVfNZ?usp=sharing
- **Training dataset:** https://drive.google.com/drive/folders/1hkoJ4LVguxVUSwXBLa6TBIwFgqM3ABxs?usp=drive_link

---

## License

MIT — see [LICENSE](LICENSE).
