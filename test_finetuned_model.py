import torch
import os
from transformers import AutoTokenizer
from train_llama125m import LLaMA125M

def load_model(checkpoint_path, device="cuda"):
    """Load the fine-tuned model."""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        vocab_size = checkpoint.get("vocab_size", None)
    else:
        state_dict = checkpoint
        vocab_size = None
    
    return state_dict, vocab_size

@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, device="cuda"):
    """Generate a response from the model."""
    model.eval()
    
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    idx = tokens.input_ids
    
    for _ in range(max_new_tokens):
        logits = model(idx)
        logits = logits[:, -1, :] / temperature
        
        # Simple sampling
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
            
        idx = torch.cat([idx, next_token], dim=1)
        
        if idx.shape[1] > 512:
            break
    
    return tokenizer.decode(idx[0], skip_special_tokens=True)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    
    # Load fine-tuned model
    checkpoint_path = "./llama125m_alpaca/sft_test_final.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model not found at {checkpoint_path}")
        print("Please run finetune_instructions.py first.")
        return
    
    state_dict, loaded_vocab_size = load_model(checkpoint_path, device)
    if loaded_vocab_size:
        vocab_size = loaded_vocab_size
    
    # Initialize model
    model = LLaMA125M(vocab_size=vocab_size, device=device).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("="*60)
    print("Testing Fine-tuned LLaMA-125M Model")
    print("="*60)
    
    # Test prompts in Alpaca format
    test_prompts = [
        "### Instruction:\nWhat is machine learning?\n\n### Response:\n",
        
        "### Instruction:\nWrite a Python function to reverse a string.\n\n### Response:\n",
        
        "### Instruction:\nExplain photosynthesis in simple terms.\n\n### Response:\n",
        
        "### Instruction:\nList three benefits of regular exercise.\n\n### Response:\n",
        
        "### Instruction:\nTranslate the following to French.\n\n### Input:\nHello, how are you today?\n\n### Response:\n",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}:")
        print(f"{'='*60}")
        
        # Show the instruction part
        instruction_part = prompt.split("### Response:")[0]
        print(instruction_part)
        
        # Generate response
        print("### Response:")
        full_response = generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, device=device)
        
        # Extract just the response part
        if "### Response:" in full_response:
            response = full_response.split("### Response:")[-1].strip()
        else:
            response = full_response.strip()
        
        print(response)
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("\nTo have an interactive conversation, run: python chat_with_model.py")
    print("="*60)

if __name__ == "__main__":
    main()