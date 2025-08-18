import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from transformers import AutoTokenizer
from datasets import load_dataset


# -----------------------
# RMSNorm
# -----------------------
class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        return self.weight * x / (norm_x / math.sqrt(x.size(-1)) + self.eps)


# -----------------------
# Rotary Embeddings
# -----------------------
def apply_rotary_emb(q, k, rope_cache):
    cos, sin = rope_cache
    T = q.size(-2)  # sequence length dimension
    cos, sin = cos[:, :, :T, :], sin[:, :, :T, :]
    q1 = q * cos + rotate_half(q) * sin
    k1 = k * cos + rotate_half(k) * sin
    return q1, k1


def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def build_rope_cache(seq_len, head_dim, device):
    theta = 10000 ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    seq_idx = torch.arange(seq_len, device=device)
    idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
    cos = torch.cos(idx_theta).repeat_interleave(2, dim=-1)  # [seq_len, head_dim]
    sin = torch.sin(idx_theta).repeat_interleave(2, dim=-1)  # [seq_len, head_dim]
    # reshape to [1, 1, seq_len, head_dim] for broadcasting with [B, n_heads, T, head_dim]
    return cos[None, None, :, :], sin[None, None, :, :]


# -----------------------
# SwiGLU FFN
# -----------------------
class SwiGLU(Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# -----------------------
# Multihead Attention with RoPE
# -----------------------
class MultiheadAttention(Module):
    def __init__(self, dim, n_heads, rope_cache, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.rope_cache = rope_cache
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, self.rope_cache)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.full((T, T), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        att = att + mask
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
 
        # Ensure dtype consistency between att and v
        y = att.to(v.dtype) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


# -----------------------
# Transformer Block
# -----------------------
class TransformerBlock(Module):
    def __init__(self, dim, n_heads, ff_hidden_dim, rope_cache, dropout=0.0):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = MultiheadAttention(dim, n_heads, rope_cache, dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ff_hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# -----------------------
# LLaMA-like Model
# -----------------------
class LLaMA125M(Module):
    def __init__(self, vocab_size, dim=768, n_layers=12, n_heads=12, ff_mult=4, max_seq_len=512, dropout=0.0, device="cuda"):
        super().__init__()
        self.device_name = device
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.rope_cache = build_rope_cache(max_seq_len, dim // n_heads, device)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, int(ff_mult * dim // 2), self.rope_cache, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx):
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


# -----------------------
# Sample text generation
# -----------------------
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=50, top_p=0.95):
    model.eval()
    tokens = tokenizer(prompt, return_tensors="pt").to(model.device_name)
    idx = tokens.input_ids
    for _ in range(max_new_tokens):
        logits = model(idx)
        # Apply temperature
        logits = logits[:, -1, :] / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    return tokenizer.decode(idx[0], skip_special_tokens=True)


# -----------------------
# Training script
# -----------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("roneneldan/TinyStories", split="train[:10%]")

    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True, max_length=128, padding="max_length")

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids"])

    model = LLaMA125M(
        vocab_size=len(tokenizer),
        dim=768,
        n_layers=12,
        n_heads=12,
        ff_mult=4,
        max_seq_len=128,
        dropout=0.1,  # Add dropout for regularization
        device=device
    ).to(device).to(dtype)
    
    # Initialize weights with better initialization
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    model.apply(init_weights)

    # Improved optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # sample before training
    print("=== Sample Before Training ===")
    print(generate(model, tokenizer, "Once upon a time", max_new_tokens=50, temperature=0.8))

    model.train()
    total_steps = 500  # More training steps for better convergence
    start_time = time.time()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    
    for step, batch in enumerate(train_loader):
        if step >= total_steps:
            break
        input_ids = batch["input_ids"].to(device)
        
        # Shift inputs and targets for proper language modeling
        inputs = input_ids[:, :-1].contiguous()
        targets = input_ids[:, 1:].contiguous()
        
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=tokenizer.pad_token_id)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

        if step % 20 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    print(f"Training finished in {time.time()-start_time:.2f}s")

    # sample after training with different prompts
    print("=== Sample After Training ===")
    print("Prompt: 'Once upon a time'")
    print(generate(model, tokenizer, "Once upon a time", max_new_tokens=50, temperature=0.8))
    print("\nPrompt: 'The little girl'")
    print(generate(model, tokenizer, "The little girl", max_new_tokens=50, temperature=0.8))
    print("\nPrompt: 'In a small village'")
    print(generate(model, tokenizer, "In a small village", max_new_tokens=50, temperature=0.8))

    # save model and tokenizer
    save_path = "./llama125m_tinystories"
    tokenizer.save_pretrained(save_path)
    torch.save(model.state_dict(), f"{save_path}/pytorch_model.bin")
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()