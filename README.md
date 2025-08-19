# LLaMA 125M Training and Fine-tuning Project

A complete implementation of a 125M parameter LLaMA-style transformer model with training, continued pretraining, and instruction fine-tuning capabilities.

## 🚀 Overview

This project implements a scaled-down version of the LLaMA architecture (125M parameters) with the following features:

- **Custom LLaMA Implementation**: Complete transformer architecture with RMSNorm, SwiGLU, and Rotary Position Embeddings (RoPE)
- **Initial Training**: Pretraining on TinyStories dataset
- **Continued Pretraining**: Resume training from checkpoints with improved data handling
- **Instruction Fine-tuning**: Fine-tune on Alpaca dataset for instruction-following capabilities
- **Interactive Chat**: Chat interface for testing the fine-tuned model
- **Comprehensive Testing**: Multiple scripts for model evaluation and sampling

## 📁 Project Structure

```
├── train_llama125m.py          # Main training script - initial pretraining
├── continue_pretrain.py        # Continued pretraining functionality
├── finetune_instructions.py    # Instruction fine-tuning on Alpaca dataset
├── test_finetuned_model.py     # Test fine-tuned model with predefined prompts
├── chat_with_model.py          # Interactive chat interface
├── sample_after_training.py    # Generate samples from trained models
├── requirements.txt            # Python dependencies (empty - see installation)
├── note.md                     # Training notes and sample outputs
├── .gitignore                  # Git ignore file
├── llama125m_tinystories/      # Directory for base model checkpoints
└── llama125m_alpaca/           # Directory for fine-tuned model checkpoints
```

## 🏗️ Model Architecture

The LLaMA 125M model implements the following components:

### Core Components
- **RMSNorm**: Root Mean Square Layer Normalization for better training stability
- **SwiGLU**: Swish-Gated Linear Unit activation function in feed-forward networks
- **Rotary Position Embeddings (RoPE)**: Relative position encoding for better sequence understanding
- **Multi-head Attention**: Standard transformer attention with causal masking

### Model Specifications
- **Parameters**: ~125M
- **Hidden Dimension**: 768
- **Layers**: 12
- **Attention Heads**: 12
- **Feed-forward Multiplier**: 4x
- **Maximum Sequence Length**: 512 (training), 5000 (inference)
- **Vocabulary Size**: GPT-2 tokenizer (~50,257 tokens)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory for training

### Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets
```

### Required packages:
- `torch` - PyTorch framework
- `transformers` - Hugging Face transformers library
- `datasets` - Hugging Face datasets library
- `math`, `time`, `os` - Standard library modules

## 🚀 Quick Start

### 1. Initial Training
Train the model from scratch on TinyStories dataset:

```bash
python train_llama125m.py
```

**Training Details:**
- Dataset: TinyStories (10% subset, ~21k samples)
- Batch Size: 32
- Learning Rate: 1e-4 with cosine annealing
- Training Steps: 500
- Sequence Length: 128 tokens
- Optimizer: AdamW with weight decay (0.01)

### 2. Continued Pretraining (Optional)
Continue training from the base checkpoint:

```bash
python continue_pretrain.py
```

**Features:**
- Resumes from existing checkpoints
- Lower learning rate (5e-5) for stability
- Automatic checkpoint saving every 500 steps
- Gradient clipping for training stability

### 3. Instruction Fine-tuning
Fine-tune the model for instruction-following:

```bash
python finetune_instructions.py
```

**Fine-tuning Details:**
- Dataset: Alpaca instruction dataset (100% by default)
- Format: Instruction → Input → Response structure
- Training Steps: 4000
- Learning Rate: 5e-3
- Batch Size: 8
- Sequence Length: 256 tokens

### 4. Test the Fine-tuned Model
Run predefined tests on the instruction-tuned model:

```bash
python test_finetuned_model.py
```

### 5. Interactive Chat
Start an interactive conversation with the fine-tuned model:

```bash
python chat_with_model.py
```

**Chat Features:**
- Interactive command-line interface
- Adjustable generation parameters (temperature, top-k, top-p)
- Built-in help and example prompts
- Conversation history tracking

## 📊 Training Pipeline

### Stage 1: Base Pretraining
```
Raw Text → Tokenization → Language Modeling → Base Model
```
- **Input**: TinyStories dataset
- **Objective**: Next token prediction
- **Output**: `llama125m_tinystories/pytorch_model.bin`

### Stage 2: Continued Pretraining (Optional)
```
Base Model → Extended Training → Improved Base Model
```
- **Input**: Base model checkpoint
- **Objective**: Continued language modeling
- **Output**: `llama125m_tinystories/continued_final.pt`

### Stage 3: Instruction Fine-tuning
```
Base Model → Instruction Data → Instruction-Following Model
```
- **Input**: Base model + Alpaca dataset
- **Objective**: Instruction following
- **Output**: `llama125m_alpaca/sft_test_final.pt`

## 🎯 Model Capabilities

### Base Model (After TinyStories Training)
- Generates coherent short stories
- Understands basic narrative structure
- Simple language patterns

**Example Output:**
```
Prompt: "Once upon a time"
Output: "Once upon a time, there was a little boy named Timmy. Timmy loved to play outside in the park with his mommy. One day, Timmy's mommy asked him to look up..."
```

### Fine-tuned Model (After Alpaca Training)
- Follows instructions and answers questions
- Provides explanations and how-to guides
- Handles various task types (coding, explanations, creative writing)

**Example Capabilities:**
- Code generation and explanation
- Question answering
- Creative writing
- Educational content
- Problem-solving assistance

## 🔧 Configuration Options

### Training Parameters
- `batch_size`: Training batch size (default: 32 for pretraining, 8 for fine-tuning)
- `learning_rate`: Learning rate (1e-4 for pretraining, 5e-5 for continued, 5e-3 for fine-tuning)
- `num_steps`: Number of training steps
- `max_length`: Maximum sequence length
- `temperature`: Sampling temperature for generation
- `dropout`: Dropout rate for regularization

### Generation Parameters
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Controls randomness (0.1-2.0)
- `top_k`: Top-k sampling parameter
- `top_p`: Nucleus sampling parameter

## 📁 Checkpoint Management

### Checkpoint Types
1. **Base Model**: `pytorch_model.bin` - Initial trained model
2. **Continued Training**: `continued_final.pt` - Extended pretraining
3. **Fine-tuned Model**: `sft_test_final.pt` - Instruction-tuned model
4. **Intermediate Checkpoints**: Saved every 500 steps during training

### Checkpoint Format
```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "step": current_step,
    "vocab_size": vocabulary_size
}
```

## 🧪 Testing and Evaluation

### Automated Testing
```bash
python test_finetuned_model.py
```
Tests the model on predefined instruction prompts covering:
- General knowledge questions
- Code generation tasks
- Creative writing
- Problem-solving

### Manual Testing
```bash
python sample_after_training.py
```
Generates samples with different temperature settings and prompts.

### Interactive Testing
```bash
python chat_with_model.py
```
Full interactive interface with customizable parameters.

## 🎛️ Advanced Usage

### Custom Dataset Training
Modify the dataset loading in [`train_llama125m.py`](train_llama125m.py:193) to use your own dataset:

```python
dataset = load_dataset("your-dataset-name", split="train")
```

### Hyperparameter Tuning
Key parameters to experiment with:
- Learning rate schedules
- Batch sizes
- Model dimensions
- Training steps
- Dropout rates

### Model Architecture Modifications
The model architecture can be customized in the [`LLaMA125M`](train_llama125m.py:122) class:
- Change model dimensions
- Adjust number of layers/heads
- Modify feed-forward multiplier
- Update maximum sequence length

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient checkpointing
   - Enable mixed precision training

2. **Checkpoint Not Found**
   - Ensure previous training steps completed successfully
   - Check file paths in scripts
   - Verify checkpoint file integrity

3. **Poor Generation Quality**
   - Increase training steps
   - Adjust learning rate
   - Try different sampling parameters

4. **Training Instability**
   - Enable gradient clipping
   - Reduce learning rate
   - Add more regularization

### Performance Optimization
- Use mixed precision training (`torch.cuda.amp`)
- Enable gradient checkpointing for memory efficiency
- Use DataLoader with multiple workers
- Optimize batch sizes for your hardware

## 📈 Results and Performance

### Training Metrics
- **Base Training**: Converges to ~2.5 loss after 500 steps
- **Fine-tuning**: Achieves instruction-following capability
- **Generation Quality**: Coherent text generation with proper formatting

### Hardware Requirements
- **Minimum**: 8GB GPU memory
- **Recommended**: 16GB+ GPU memory
- **Training Time**: ~30 minutes for base training on RTX 3080

## 🤝 Contributing

Feel free to contribute by:
- Adding new datasets
- Implementing model improvements
- Optimizing training procedures
- Adding evaluation metrics
- Improving documentation

## 📄 License

This project is open source. Please ensure compliance with dataset licenses:
- TinyStories: Check original dataset license
- Alpaca: Stanford Alpaca license terms

## 🔗 References

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Test with provided example scripts
4. Verify your environment setup

---

**Note**: This is an educational implementation. For production use, consider using official LLaMA implementations or other established frameworks.