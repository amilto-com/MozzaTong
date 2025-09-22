"""
CLI interface for MozzaTong - Natural Language to TONG converter.
"""

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import sys
import os

from .inference import TongGenerator
from .trainer import TongTrainer
from .dataset import TongDatasetGenerator

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """MozzaTong: Convert natural language to TONG programming language."""
    console.print(
        Panel.fit(
            Text("🍕 MozzaTong - Natural Language to TONG Converter", style="bold blue"),
            border_style="blue"
        )
    )


@main.command()
@click.option(
    "--prompt", "-p", 
    required=True, 
    help="Natural language prompt to convert to TONG"
)
@click.option(
    "--model-path", "-m",
    default="./models/mozzatong-model",
    help="Path to the trained MozzaTong model"
)
@click.option(
    "--max-length", "-l",
    default=512,
    help="Maximum length of generated TONG code"
)
@click.option(
    "--temperature", "-t",
    default=0.7,
    help="Sampling temperature for generation"
)
def generate(prompt: str, model_path: str, max_length: int, temperature: float):
    """Generate TONG code from natural language prompt."""
    try:
        if not os.path.exists(model_path):
            console.print(f"❌ Model not found at {model_path}", style="red")
            console.print("💡 Train a model first using: mozzatong train", style="yellow")
            sys.exit(1)
        
        generator = TongGenerator(model_path)
        
        console.print(f"🔍 Prompt: {prompt}", style="cyan")
        console.print("🤖 Generating TONG code...", style="yellow")
        
        tong_code = generator.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        console.print("\n📝 Generated TONG Code:", style="green bold")
        console.print(Panel(tong_code, title="TONG Output", border_style="green"))
        
    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="red")
        sys.exit(1)


@main.command()
@click.option(
    "--output-dir", "-o",
    default="./datasets/tong_dataset",
    help="Output directory for generated dataset"
)
@click.option(
    "--num-samples", "-n",
    default=1000,
    help="Number of training samples to generate"
)
@click.option(
    "--difficulty", "-d",
    type=click.Choice(["basic", "intermediate", "advanced", "mixed"]),
    default="mixed",
    help="Difficulty level of generated examples"
)
def generate_dataset(output_dir: str, num_samples: int, difficulty: str):
    """Generate training dataset for MozzaTong."""
    try:
        console.print("🏗️  Generating training dataset...", style="yellow")
        
        dataset_generator = TongDatasetGenerator()
        dataset_generator.generate_dataset(
            output_dir=output_dir,
            num_samples=num_samples,
            difficulty=difficulty
        )
        
        console.print(f"✅ Dataset generated successfully at {output_dir}", style="green")
        console.print(f"📊 Generated {num_samples} samples", style="cyan")
        
    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="red")
        sys.exit(1)


@main.command()
@click.option(
    "--dataset-path", "-d",
    required=True,
    help="Path to the training dataset"
)
@click.option(
    "--model-name", "-m",
    default="microsoft/CodeT5-small",
    help="Base model to fine-tune"
)
@click.option(
    "--output-dir", "-o",
    default="./models/mozzatong-model",
    help="Output directory for the trained model"
)
@click.option(
    "--epochs", "-e",
    default=3,
    help="Number of training epochs"
)
@click.option(
    "--batch-size", "-b",
    default=8,
    help="Training batch size"
)
@click.option(
    "--learning-rate", "-lr",
    default=5e-5,
    help="Learning rate"
)
@click.option(
    "--use-lora", 
    is_flag=True,
    help="Use LoRA for efficient fine-tuning"
)
def train(
    dataset_path: str, 
    model_name: str, 
    output_dir: str, 
    epochs: int, 
    batch_size: int, 
    learning_rate: float,
    use_lora: bool
):
    """Train MozzaTong model on dataset."""
    try:
        console.print("🚀 Starting MozzaTong training...", style="yellow")
        console.print(f"📚 Dataset: {dataset_path}", style="cyan")
        console.print(f"🤖 Base model: {model_name}", style="cyan")
        console.print(f"💾 Output: {output_dir}", style="cyan")
        
        trainer = TongTrainer(
            model_name=model_name,
            use_lora=use_lora
        )
        
        trainer.train(
            dataset_path=dataset_path,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        console.print("✅ Training completed successfully!", style="green")
        console.print(f"💾 Model saved to {output_dir}", style="cyan")
        
    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="red")
        sys.exit(1)


@main.command()
def interactive():
    """Start interactive MozzaTong session."""
    try:
        model_path = "./models/mozzatong-model"
        
        if not os.path.exists(model_path):
            console.print(f"❌ Model not found at {model_path}", style="red")
            console.print("💡 Train a model first using: mozzatong train", style="yellow")
            sys.exit(1)
        
        generator = TongGenerator(model_path)
        
        console.print("🔧 Starting interactive MozzaTong session...", style="green")
        console.print("💡 Type 'exit' to quit\n", style="yellow")
        
        while True:
            try:
                prompt = input("🔍 Enter your prompt: ")
                
                if prompt.lower() in ['exit', 'quit', 'q']:
                    console.print("👋 Goodbye!", style="cyan")
                    break
                
                if not prompt.strip():
                    continue
                
                console.print("🤖 Generating...", style="yellow")
                tong_code = generator.generate(prompt)
                
                console.print("\n📝 TONG Code:", style="green bold")
                console.print(Panel(tong_code, border_style="green"))
                print()
                
            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!", style="cyan")
                break
            except Exception as e:
                console.print(f"❌ Error: {str(e)}", style="red")
                
    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()