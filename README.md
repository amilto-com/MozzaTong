# 🍕 MozzaTong

**State-of-the-art LLM that converts natural language prompts to TONG programming language**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

MozzaTong is a cutting-edge language model specifically fine-tuned to translate natural language descriptions into [TONG programming language](https://github.com/amilto-com/Tong) code. Built on modern transformer architectures, it provides an intuitive interface for developers to generate TONG code from plain English descriptions.

## ✨ Features

🚀 **State-of-the-art LLM**: Fine-tuned from powerful coding assistant models  
🎯 **TONG-specific**: Specialized for the TONG programming language syntax and semantics  
💾 **Memory efficient**: Designed to run on 10GB GPU RAM constraint  
🔧 **Easy CLI**: Simple command-line interface for quick code generation  
⚡ **Fast inference**: Optimized for real-time code generation  
🎛️ **Flexible training**: Support for LoRA fine-tuning and custom datasets  
📊 **Dataset generation**: Built-in tools for creating training data

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/amilto-com/MozzaTong.git
cd MozzaTong

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```bash
# Generate TONG code from natural language
mozzatong generate -p "Create a function that calculates factorial"

# Start interactive session
mozzatong interactive

# Train a custom model
mozzatong train -d ./datasets/my_dataset.jsonl

# Generate training dataset
mozzatong generate-dataset -n 1000 -o ./datasets/tong_dataset
```

## 📖 TONG Language

TONG is the ultimate programming language designed for high-performance parallel and distributed computing. It combines the best syntax elements from multiple languages with features like:

- **Zero-cost abstractions**
- **Memory safety** without garbage collection
- **Automatic parallelization**
- **Heterogeneous computing** (CPU/GPU/NPU/FPGA)
- **WebAssembly ready**

### Example TONG Code

```tong
// Simple function
fn factorial(n) {
    if n <= 1 {
        return 1
    }
    return n * factorial(n - 1)
}

// Parallel processing
parallel {
    let result1 = heavy_computation1()
    let result2 = heavy_computation2()
    combine(result1, result2)
}

// GPU kernel
gpu_kernel fn matrix_multiply(a: Matrix<f32>, b: Matrix<f32>) -> Matrix<f32> {
    a * b
}
```

## 🛠️ CLI Reference

### Generate Code

```bash
mozzatong generate [OPTIONS]

Options:
  -p, --prompt TEXT       Natural language prompt to convert to TONG [required]
  -m, --model-path TEXT   Path to the trained MozzaTong model [default: ./models/mozzatong-model]
  -l, --max-length INT    Maximum length of generated TONG code [default: 512]
  -t, --temperature FLOAT Sampling temperature for generation [default: 0.7]
```

### Train Model

```bash
mozzatong train [OPTIONS]

Options:
  -d, --dataset-path TEXT    Path to the training dataset [required]
  -m, --model-name TEXT      Base model to fine-tune [default: microsoft/CodeT5-small]
  -o, --output-dir TEXT      Output directory for the trained model [default: ./models/mozzatong-model]
  -e, --epochs INT           Number of training epochs [default: 3]
  -b, --batch-size INT       Training batch size [default: 8]
  -lr, --learning-rate FLOAT Learning rate [default: 5e-5]
  --use-lora                 Use LoRA for efficient fine-tuning
```

### Generate Dataset

```bash
mozzatong generate-dataset [OPTIONS]

Options:
  -o, --output-dir TEXT    Output directory for generated dataset [default: ./datasets/tong_dataset]
  -n, --num-samples INT    Number of training samples to generate [default: 1000]
  -d, --difficulty TEXT    Difficulty level [basic|intermediate|advanced|mixed] [default: mixed]
```

### Interactive Mode

```bash
mozzatong interactive
```

Start an interactive session where you can continuously generate TONG code from natural language prompts.

## 🏗️ Architecture

MozzaTong is built on a modular architecture:

```
src/mozzatong/
├── __init__.py          # Package initialization
├── cli.py               # Command-line interface
├── inference.py         # Code generation and inference
├── trainer.py           # Model training and fine-tuning
└── dataset.py           # Dataset generation tools
```

### Core Components

- **TongGenerator**: Handles inference and code generation
- **TongTrainer**: Manages model training and fine-tuning
- **TongDatasetGenerator**: Creates training datasets
- **TongCodeValidator**: Validates generated TONG code

## 🎯 Training Your Own Model

### 1. Generate Dataset

```bash
# Generate a mixed difficulty dataset with 5000 samples
mozzatong generate-dataset -n 5000 -d mixed -o ./datasets/my_tong_dataset
```

### 2. Train Model

```bash
# Fine-tune with LoRA for efficiency
mozzatong train \
  -d ./datasets/my_tong_dataset/train_dataset.jsonl \
  -m microsoft/CodeT5-small \
  -o ./models/my-mozzatong-model \
  -e 5 \
  -b 8 \
  --use-lora
```

### 3. Test Your Model

```bash
# Generate code with your trained model
mozzatong generate \
  -p "Write a function to calculate fibonacci numbers" \
  -m ./models/my-mozzatong-model
```

## 📊 Dataset Format

Training datasets should be in JSONL format with the following structure:

```json
{
  "description": "Create a function that adds two numbers",
  "tong_code": "fn add(a, b) {\n    return a + b\n}\n\nlet result = add(5, 3)\nprint(\"Result:\", result)"
}
```

## 🔧 Advanced Configuration

### Memory-Efficient Training

For systems with limited GPU memory:

```bash
# Use smaller batch size and enable LoRA
mozzatong train \
  -d ./datasets/dataset.jsonl \
  -b 4 \
  --use-lora \
  -m microsoft/CodeT5-small
```

### Custom Model Configuration

You can modify training parameters in the code:

```python
from mozzatong import TongTrainer

trainer = TongTrainer(
    model_name="microsoft/CodeT5-base",
    use_lora=True,
    lora_rank=16,
    lora_alpha=32
)

trainer.train(
    dataset_path="./datasets/dataset.jsonl",
    output_dir="./models/custom-model",
    epochs=5,
    batch_size=8,
    learning_rate=3e-5
)
```

## 🧪 Examples

### Basic Code Generation

```python
from mozzatong import TongGenerator

generator = TongGenerator("./models/mozzatong-model")

# Generate TONG code
tong_code = generator.generate(
    prompt="Create a function that calculates the area of a circle",
    temperature=0.7
)

print(tong_code)
```

### Batch Generation

```python
prompts = [
    "Write a function to sort an array",
    "Create a parallel matrix multiplication function",
    "Implement a binary search algorithm"
]

results = generator.generate_batch(prompts)
for prompt, code in zip(prompts, results):
    print(f"Prompt: {prompt}")
    print(f"Code:\n{code}\n")
```

## 🤝 Contributing

We welcome contributions to MozzaTong! Here's how you can help:

1. **Report Issues**: Found a bug? [Open an issue](https://github.com/amilto-com/MozzaTong/issues)
2. **Suggest Features**: Have an idea? [Start a discussion](https://github.com/amilto-com/MozzaTong/discussions)
3. **Submit PRs**: 
   - Fork the repository
   - Create a feature branch
   - Make your changes
   - Add tests if applicable
   - Submit a pull request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/amilto-com/MozzaTong.git
cd MozzaTong

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
isort src/
```

## 📚 Resources

- [TONG Language Repository](https://github.com/amilto-com/Tong)
- [TONG Language Specification](https://github.com/amilto-com/Tong/blob/main/tong_spec.md)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 🙏 Acknowledgments

- Built on top of [Transformers](https://github.com/huggingface/transformers) by Hugging Face
- Inspired by the [TONG programming language](https://github.com/amilto-com/Tong)
- Uses [LoRA](https://github.com/microsoft/LoRA) for efficient fine-tuning

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏢 About AMILTO

MozzaTong is developed by [AMILTO SARL](https://amilto.com), a company focused on advancing programming languages and developer tools.

---

**MozzaTong** - Making TONG programming accessible through natural language! 🍕✨
