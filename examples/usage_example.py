"""
Example usage of MozzaTong for generating TONG code from natural language.
"""

from mozzatong import TongGenerator, TongTrainer, TongDatasetGenerator
import os


def main():
    """Demonstrate MozzaTong capabilities."""
    
    print("🍕 MozzaTong Example Usage")
    print("=" * 50)
    
    # Example 1: Generate a small dataset
    print("\n1. Generating sample dataset...")
    dataset_generator = TongDatasetGenerator()
    
    dataset_path = "./example_dataset"
    dataset_generator.generate_dataset(
        output_dir=dataset_path,
        num_samples=100,
        difficulty="basic"
    )
    print(f"✅ Generated dataset at {dataset_path}")
    
    # Example 2: Show some example prompts and expected outputs
    print("\n2. Example natural language to TONG conversions:")
    
    examples = [
        "Create a function that adds two numbers",
        "Write a function to check if a number is even", 
        "Make a function that calculates factorial",
        "Create a parallel function to square array elements"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. Prompt: {example}")
        print("   Expected TONG output:")
        
        if "adds two numbers" in example:
            print("""   fn add(a, b) {
       return a + b
   }
   
   let result = add(5, 3)
   print("Result:", result)""")
        
        elif "even" in example:
            print("""   fn is_even(n) {
       return n % 2 == 0
   }
   
   let num = 42
   if is_even(num) {
       print(num, "is even")
   } else {
       print(num, "is odd")
   }""")
        
        elif "factorial" in example:
            print("""   fn factorial(n) {
       if n <= 1 {
           return 1
       }
       return n * factorial(n - 1)
   }
   
   let fact = factorial(5)
   print("Factorial of 5:", fact)""")
        
        elif "parallel" in example:
            print("""   fn parallel_square(numbers) {
       return parallel {
           map(numbers, |x| x * x)
       }
   }
   
   let data = [1, 2, 3, 4, 5]
   let squared = parallel_square(data)
   print("Squared:", squared)""")
    
    print("\n3. Training workflow:")
    print("""
   Step 1: Generate training dataset
   $ mozzatong generate-dataset -n 1000 -o ./datasets/tong_dataset
   
   Step 2: Train the model
   $ mozzatong train -d ./datasets/tong_dataset/train_dataset.jsonl --use-lora
   
   Step 3: Generate code with trained model
   $ mozzatong generate -p "Create a function to sort an array"
   
   Step 4: Start interactive session
   $ mozzatong interactive
   """)
    
    print("\n4. Model specifications:")
    print("""
   • Base model: microsoft/CodeT5-small (fits in 10GB GPU RAM)
   • Fine-tuning: LoRA for memory efficiency
   • Specialization: Natural language to TONG code conversion
   • Architecture: Encoder-decoder transformer
   • Context length: 512 tokens
   • Generation: Beam search with temperature sampling
   """)
    
    print("\n🎯 MozzaTong is ready to convert your ideas into TONG code!")


if __name__ == "__main__":
    main()