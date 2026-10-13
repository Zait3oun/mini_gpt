
import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CHECKPOINT_PATH = ARTIFACTS_DIR / "mini_gpt.pth"
VOCAB_PATH = ARTIFACTS_DIR / "vocab.json"


# -----------------------------
# Model Architecture
# Must match the training notebook
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        mask = torch.triu(
            torch.ones(block_size, block_size, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("mask", mask)

        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        T = x.size(1)

        out, _ = self.attention(
            x, x, x,
            attn_mask=self.mask[:T, :T],
            need_weights=False,
        )

        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model, num_heads, block_size, dropout
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=32,
        num_heads=4,
        num_layers=2,
        block_size=8,
        dropout=0.1,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model, num_heads, block_size, dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        _, T = idx.shape

        token_emb = self.token_embedding(idx)
        positions = torch.arange(T, device=idx.device)
        position_emb = self.position_embedding(positions)

        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.final_ln(x)

        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=10, temperature=1.0):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probabilities = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx


# -----------------------------
# Load the trained model
# -----------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocabulary = json.load(f)

    stoi = vocabulary["stoi"]
    itos = {int(i): token for i, token in vocabulary["itos"].items()}

    model = MiniGPT(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, stoi, itos


# -----------------------------
# Streamlit Interface
# -----------------------------
st.set_page_config(
    page_title="Mini GPT Playground",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Mini GPT Playground")
st.caption("A tiny GPT-style language model built with PyTorch")

st.info(
    "Educational demo: this model was trained on a very small text dataset. "
    "Generated text may be repetitive or grammatically incorrect."
)

if not CHECKPOINT_PATH.exists() or not VOCAB_PATH.exists():
    st.error(
        "Model files not found. Please save the trained model "
        "in the artifacts folder first."
    )
    st.stop()

try:
    model, stoi, itos = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()


prompt = st.text_input(
    "Enter a prompt",
    value="I love",
    help="Use words that appear in the training vocabulary.",
)

col1, col2 = st.columns(2)

with col1:
    max_new_tokens = st.slider(
        "New tokens",
        min_value=1,
        max_value=20,
        value=10,
    )

with col2:
    temperature = st.slider(
        "Temperature",
        min_value=0.2,
        max_value=1.5,
        value=0.8,
        step=0.1,
    )

if st.button("✨ Generate Text", type="primary", use_container_width=True):
    import re

    prompt_tokens = re.findall(r"\w+|[^\w\s]", prompt)

    if not prompt_tokens:
        st.warning("Please enter a prompt.")
    elif any(token not in stoi for token in prompt_tokens):
        st.warning(
            "Some words are not in this model's vocabulary. "
            "Try words from the training examples, such as: I love, "
            "machine learning, deep learning."
        )
    elif temperature <= 0:
        st.warning("Temperature must be greater than zero.")
    else:
        prompt_ids = [stoi[token] for token in prompt_tokens]
        context = torch.tensor([prompt_ids], dtype=torch.long)

        generated_ids = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        generated_tokens = [
            itos[int(token_id)]
            for token_id in generated_ids[0].tolist()
        ]

        generated_text = " ".join(generated_tokens)

        st.subheader("Generated Text")
        st.write(generated_text)