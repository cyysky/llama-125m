import torch
import os
from transformers import AutoTokenizer
from train_llama125m import LLaMA125M, generate

def load_checkpoint(checkpoint_dir="./llama125m_tinystories", device="cuda"):
    """
    Load the most recent checkpoint from the directory.
    """
    # List of possible checkpoint names in order of preference
    checkpoint_candidates = [
        "continued_final.pt",  # Final checkpoint from continued training
        "continued_500.pt",     # Checkpoint at step 500
        "checkpoint_final.pt",  # Generic final checkpoint
        "checkpoint_500.pt",    # Generic checkpoint at step 500
        "pytorch_model.bin"     # Original model
    ]
    
    # Find the first existing checkpoint
    checkpoint_path = None
    for candidate in checkpoint_candidates:
        full_path = os.path.join(checkpoint_dir, candidate)
        if os.path.exists(full_path):
            checkpoint_path = full_path
            print(f"Loading checkpoint: {candidate}")
            break
    
    if checkpoint_path is None:
        # If no specific checkpoint found, look for any .pt file
        pt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if pt_files:
            # Sort to get the most recent (assuming naming convention)
            pt_files.sort()
            checkpoint_path = os.path.join(checkpoint_dir, pt_files[-1])
            print(f"Loading checkpoint: {pt_files[-1]}")
        else:
            # Fall back to pytorch_model.bin
            checkpoint_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Full checkpoint with metadata
        state_dict = checkpoint["model_state_dict"]
        vocab_size = checkpoint.get("vocab_size", None)
        step = checkpoint.get("step", "unknown")
        print(f"Loaded checkpoint from step: {step}")
    else:
        # Raw state dict
        state_dict = checkpoint
        vocab_size = None
        print("Loaded raw state dict")
    
    return state_dict, vocab_size

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    try:
        # Load checkpoint
        state_dict, vocab_size = load_checkpoint("./llama125m_tinystories", device)
        
        # If vocab_size not in checkpoint, use tokenizer's vocab size
        if vocab_size is None:
            vocab_size = len(tokenizer)
        
        # Initialize model
        model = LLaMA125M(vocab_size).to(device)
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode
        
        # Set device for generation
        if hasattr(model, 'device_name'):
            model.device_name = device
        
        print("\n=== Sample After Training ===")
        prompts = [
            "Once upon a time",
            "The little girl",
            "In a small village",
            "A brave knight",
            "The magic forest"
        ]
        
        # Test with different temperature settings
        temperatures = [0.7, 0.8, 0.9]
        
        for temp in temperatures:
            print(f"\n--- Temperature: {temp} ---")
            for p in prompts[:3]:  # Use first 3 prompts for each temperature
                print(f"\nPrompt: '{p}'")
                generated = generate(model, tokenizer, p, max_new_tokens=50, temperature=temp)
                print(generated)
        
        # Also test with top-k sampling
        print("\n--- With top_k=40 and temperature=0.8 ---")
        for p in prompts:
            print(f"\nPrompt: '{p}'")
            # Note: The generate function might need modification to support top_k
            # For now, using standard temperature sampling
            generated = generate(model, tokenizer, p, max_new_tokens=50, temperature=0.8)
            print(generated)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train_llama125m.py first to create the initial model,")
        print("then run continue_pretrain.py to create continued training checkpoints.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check that the checkpoint file is valid and compatible.")