import torch
import os
from transformers import AutoTokenizer
from train_llama125m import LLaMA125M

def load_finetuned_model(checkpoint_path, device="cuda"):
    """
    Load the fine-tuned model from checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        vocab_size = checkpoint.get("vocab_size", None)
        step = checkpoint.get("step", "unknown")
        print(f"Loaded checkpoint from step: {step}")
    else:
        state_dict = checkpoint
        vocab_size = None
        print("Loaded model state dict")
    
    return state_dict, vocab_size

@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7, top_k=50, top_p=0.95, device="cuda"):
    """
    Generate a response from the model given a prompt.
    """
    model.eval()
    
    # Tokenize the prompt
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    idx = tokens.input_ids
    
    # Track where the prompt ends to extract only the generated response
    prompt_length = idx.shape[1]
    
    for _ in range(max_new_tokens):
        # Get model predictions
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
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the distribution
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Check for EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break
            
        idx = torch.cat([idx, next_token], dim=1)
        
        # Prevent infinite generation
        if idx.shape[1] > 512:
            break
    
    # Decode the full sequence with special tokens included
    full_text_with_tokens = tokenizer.decode(idx[0], skip_special_tokens=False)
    full_text_clean = tokenizer.decode(idx[0], skip_special_tokens=True)
    
    # Return both versions
    return full_text_clean, full_text_with_tokens

def format_instruction_prompt(instruction, input_text=""):
    """
    Format the prompt in the Alpaca instruction format.
    """
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt

def chat_loop(model, tokenizer, device="cuda"):
    """
    Interactive chat loop with the model.
    """
    print("\n" + "="*60)
    print("Chat with Fine-tuned LLaMA-125M Model")
    print("="*60)
    print("\nThis model has been fine-tuned on the Alpaca instruction dataset.")
    print("You can ask questions, request explanations, or give tasks.")
    print("\nCommands:")
    print("  - Type 'quit', 'exit', or 'bye' to end the conversation")
    print("  - Type 'clear' to clear the screen")
    print("  - Type 'help' for example prompts")
    print("  - Type 'settings' to adjust generation parameters")
    print("\n" + "-"*60 + "\n")
    
    # Default generation settings
    settings = {
        'temperature': 0.7,
        'top_k': 50,
        'top_p': 0.95,
        'max_tokens': 500
    }
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("\n📝 You: ").strip()
            
            # Check for special commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye! Thanks for chatting!")
                break
            
            elif user_input.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                print("\n" + "="*60)
                print("Chat with Fine-tuned LLaMA-125M Model")
                print("="*60 + "\n")
                continue
            
            elif user_input.lower() == 'help':
                print("\n📚 Example prompts you can try:")
                print("  - Explain quantum computing in simple terms")
                print("  - Write a Python function to calculate fibonacci numbers")
                print("  - What are the benefits of exercise?")
                print("  - Translate 'Hello, how are you?' to Spanish")
                print("  - Write a short story about a robot")
                print("  - List 5 tips for better sleep")
                print("  - Explain the difference between AI and machine learning")
                continue
            
            elif user_input.lower() == 'settings':
                print("\n⚙️  Current settings:")
                for key, value in settings.items():
                    print(f"  {key}: {value}")
                print("\nAdjust settings (press Enter to keep current value):")
                
                new_temp = input(f"  Temperature [{settings['temperature']}]: ").strip()
                if new_temp:
                    try:
                        settings['temperature'] = float(new_temp)
                    except ValueError:
                        print("  Invalid temperature, keeping current value")
                
                new_topk = input(f"  Top-k [{settings['top_k']}]: ").strip()
                if new_topk:
                    try:
                        settings['top_k'] = int(new_topk)
                    except ValueError:
                        print("  Invalid top-k, keeping current value")
                
                new_topp = input(f"  Top-p [{settings['top_p']}]: ").strip()
                if new_topp:
                    try:
                        settings['top_p'] = float(new_topp)
                    except ValueError:
                        print("  Invalid top-p, keeping current value")
                
                new_max = input(f"  Max tokens [{settings['max_tokens']}]: ").strip()
                if new_max:
                    try:
                        settings['max_tokens'] = int(new_max)
                    except ValueError:
                        print("  Invalid max tokens, keeping current value")
                
                print("\n✅ Settings updated!")
                continue
            
            if not user_input:
                continue
            
            # Format the instruction prompt
            formatted_prompt = format_instruction_prompt(user_input)
            
            # Generate response
            print("\n🤖 Model: ", end="", flush=True)
            
            response_clean, response_with_tokens = generate_response(
                model,
                tokenizer,
                formatted_prompt,
                max_new_tokens=settings['max_tokens'],
                temperature=settings['temperature'],
                top_k=settings['top_k'],
                top_p=settings['top_p'],
                device=device
            )
            
            # Extract just the response part (after "### Response:")
            if "### Response:" in response_clean:
                response_text = response_clean.split("### Response:")[-1].strip()
            else:
                response_text = response_clean.strip()
            
            # Extract response with special tokens
            if "### Response:" in response_with_tokens:
                response_with_tokens_text = response_with_tokens.split("### Response:")[-1].strip()
            else:
                response_with_tokens_text = response_with_tokens.strip()
            
            print(response_text)
            #print(f"\n🔍 With special tokens: {response_with_tokens_text}")
            
            # Store in conversation history
            conversation_history.append({
                'user': user_input,
                'model': response_text
            })
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'quit' to exit or continue chatting.")
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")
            continue

def main():
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    
    # Check for fine-tuned checkpoint
    checkpoint_path = "./llama125m_alpaca/sft_test_final.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Error: Fine-tuned model not found at {checkpoint_path}")
        print("Please run finetune_instructions.py first to create the fine-tuned model.")
        return
    
    try:
        # Load the fine-tuned model
        print(f"Loading fine-tuned model from {checkpoint_path}...")
        state_dict, loaded_vocab_size = load_finetuned_model(checkpoint_path, device)
        
        # Use loaded vocab size if available, otherwise use tokenizer's
        if loaded_vocab_size is not None:
            vocab_size = loaded_vocab_size
        
        # Initialize model
        model = LLaMA125M(
            vocab_size=vocab_size,
            dim=768,
            n_layers=12,
            n_heads=12,
            ff_mult=4,
            max_seq_len=5000,
            dropout=0.0,
            device=device
        ).to(device)
        
        # Load the state dict
        model.load_state_dict(state_dict)
        model.eval()
        
        print("✅ Model loaded successfully!")
        
        # Start the chat loop
        chat_loop(model, tokenizer, device)
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you've run finetune_instructions.py successfully")
        print("2. Check that the checkpoint file exists and is not corrupted")
        print("3. Ensure you have enough GPU/CPU memory to load the model")

if __name__ == "__main__":
    main()