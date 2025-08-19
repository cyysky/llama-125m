import torch
from torch import optim
from train_llama125m import LLaMA125M
from train_llama125m import main as original_train_main

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import os

def continue_pretrain(
    checkpoint_path: str,
    tokenizer,
    dataset,
    num_steps: int = 1000,
    lr: float = 5e-5,  # Lower learning rate for fine-tuning
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir="./llama125m_tinystories",
    checkpoint_name=None
):
    # Load checkpoint if it exists
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle two cases: full checkpoint dict OR raw state_dict (HF style)
    if "model_state_dict" in checkpoint:
        vocab_size = checkpoint.get("vocab_size", tokenizer.vocab_size)
        model = LLaMA125M(vocab_size).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_step = checkpoint.get("step", 0)

        # Use AdamW with weight decay for better regularization
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        # assume checkpoint itself is a raw state_dict
        vocab_size = tokenizer.vocab_size
        model = LLaMA125M(vocab_size).to(device)
        model.load_state_dict(checkpoint)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        start_step = 0

    model.train()
    
    # Add learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr/10)

    for step in range(start_step, start_step + num_steps):
        # Get batch from dataset
        input_ids, _ = next(iter(dataset))
        input_ids = input_ids.to(device)
        
        # Shift inputs and targets for proper language modeling
        inputs = input_ids[:, :-1].contiguous()
        targets = input_ids[:, 1:].contiguous()

        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1),
            ignore_index=tokenizer.pad_token_id
        )
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            print(f"Step {step} | Loss {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if step % 500 == 0 and step > 0:
            save_dict = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": vocab_size
            }
            ckpt_name = f"checkpoint_{step}.pt" if checkpoint_name is None else f"{checkpoint_name}_{step}.pt"
            torch.save(save_dict, os.path.join(output_dir, ckpt_name))
            print(f"Checkpoint saved: {ckpt_name}")
    
    # Save final model
    final_dict = {
        "step": start_step + num_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab_size": vocab_size
    }
    final_name = f"checkpoint_final.pt" if checkpoint_name is None else f"{checkpoint_name}_final.pt"
    torch.save(final_dict, os.path.join(output_dir, final_name))
    print(f"Final checkpoint saved: {final_name}")

    return model

def continue_pretrain_tinystories_slice(
    checkpoint_path: str,
    num_steps: int = 500,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "./llama125m_tinystories",
    checkpoint_name: str = None,
    batch_size: int = 8,  # Increased batch size
    lr: float = 5e-5  # Lower learning rate
):
    """
    Convenience wrapper to continue pretraining on TinyStories split.
    """
    # Use a different split for continued training to avoid overfitting
    dataset = load_dataset("roneneldan/TinyStories", split="train[32%:35%]")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128,  # Match original training
            return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "labels": tokens["input_ids"].squeeze()
        }

    dataset = dataset.map(tokenize_fn, remove_columns=["text"])
    dataset.set_format(type='torch', columns=['input_ids', 'labels'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Wrap dataloader as iterator for continue_pretrain
    def dataset_iter():
        while True:
            for batch in dataloader:
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                if isinstance(input_ids, list):
                    input_ids = torch.stack(input_ids)
                    labels = torch.stack(labels)
                yield input_ids, labels

    return continue_pretrain(
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer,
        dataset=dataset_iter(),
        num_steps=num_steps,
        lr=lr,
        device=device,
        output_dir=output_dir,
        checkpoint_name=checkpoint_name
    )

if __name__ == "__main__":
    import os
    
    # Check if checkpoint exists
    checkpoint_path = "llama125m_tinystories/pytorch_model.bin"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please run train_llama125m.py first to create the initial model.")
        exit(1)
    
    print("Starting continued pretraining...")
    continue_pretrain_tinystories_slice(
        checkpoint_path,
        output_dir="./llama125m_tinystories",
        checkpoint_name="continued",
        num_steps=500,
        batch_size=8,
        lr=5e-5
    )
    print("Continued pretraining complete!")