# Mini GPT From Scratch 🧠

🚀 **Live Demo:** [Try Mini GPT](https://minigpt-magbwsb3uvn3mazzvkpzqk.streamlit.app/)

A miniature GPT-style language model built with PyTorch,
featuring causal self-attention and interactive text generation.
A small GPT-style language model built with PyTorch to explore the core concepts behind Transformer-based text generation.

## Overview

This project implements and trains a miniature GPT-like model on a tiny custom text dataset. It provides a simple interactive playground where users can enter a prompt and generate text.

The project is intended for educational purposes and demonstrates the basic workflow of building, training, saving, and using a language model.

## Features

* Custom word-level tokenizer
* Token and positional embeddings
* Causal multi-head self-attention
* Transformer blocks with residual connections and Layer Normalization
* Feed-forward neural networks with GELU activation
* Next-token prediction and text generation
* Model checkpoint and vocabulary saving
* Interactive Streamlit interface
* Adjustable generation length and temperature

## Tech Stack

* Python
* PyTorch
* Streamlit

## Project Structure

```text
mini-gpt-from-scratch/
├── app.py
├── mini_gpt.ipynb
├── requirements.txt
├── README.md
├── .gitignore
└── artifacts/
    ├── mini_gpt.pth
    └── vocab.json
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd mini-gpt-from-scratch
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## How It Works

1. The tokenizer converts text into token IDs.
2. Token and positional embeddings represent the input sequence.
3. Transformer blocks process the sequence using causal self-attention and feed-forward layers.
4. The model predicts logits for the next token.
5. Text generation repeatedly samples a next token and appends it to the sequence.

## Limitations

The model is trained on a very small dataset and has a tiny vocabulary. Its output may be repetitive, grammatically incorrect, or limited to patterns found in the training text.

This is an educational implementation, not a production-ready language model.

## Future Improvements

* Train on a larger text corpus
* Improve tokenization
* Add model evaluation
* Experiment with sampling strategies
* Explore larger Transformer architectures

## Author

**Aml Abdullah**

Computer Science | AI & Data Science
