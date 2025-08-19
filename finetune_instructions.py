import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from train_llama125m import LLaMA125M
from continue_pretrain import continue_pretrain

def finetune_alpaca(
    checkpoint_path: str,
    num_steps: int = 500,
    dataset_fraction: str = "train[:1%]",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "./llama125m_alpaca",
    checkpoint_name: str = "sft",
    lr: float = 5e-5,
    max_length: int = 256,
    batch_size: int = 8,
):
    """
    Instruction fine-tuning on Alpaca dataset (or compatible).
    Loads only a fraction of dataset if specified.
    """

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # load alpaca dataset
    ds = load_dataset("tatsu-lab/alpaca", split=dataset_fraction)

    def format_and_tokenize(example):
        inst = example.get("instruction", "")
        inp = example.get("input", "")
        out = example.get("output", "")
        text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        tokens = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
        return {
            "input_ids": tokens["input_ids"],
            "labels": tokens["input_ids"]  # For language modeling, labels are the same as input_ids
        }

    ds = ds.map(format_and_tokenize, batched=False)
    ds.set_format(type="torch", columns=["input_ids", "labels"])

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Wrap dataloader to yield tuples instead of dictionaries
    def dataset_iter():
        while True:
            for batch in dataloader:
                input_ids = batch["input_ids"]
                labels = batch["labels"]
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
    
    # Check which checkpoint to use
    checkpoint_candidates = [
        "./llama125m_tinystories/continued_final.pt",
        "./llama125m_tinystories/pytorch_model.bin"
    ]
    
    checkpoint_path = None
    for path in checkpoint_candidates:
        if os.path.exists(path):
            checkpoint_path = path
            print(f"Using checkpoint: {checkpoint_path}")
            break
    
    if checkpoint_path is None:
        print("Error: No checkpoint found. Please run train_llama125m.py first.")
        exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = "./llama125m_alpaca"
    os.makedirs(output_dir, exist_ok=True)
    
    # Example run: fine-tune for 100 steps on 1% of Alpaca
    print("Starting instruction fine-tuning on Alpaca dataset...")
    model = finetune_alpaca(
        checkpoint_path=checkpoint_path,
        num_steps=4000,
        dataset_fraction="train[:100%]",
        output_dir=output_dir,
        checkpoint_name="sft_test",
        lr=5e-3,
        max_length=256,
        batch_size=8
    )
    print("Fine-tuning complete!")