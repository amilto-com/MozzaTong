"""
Dataset generation module for MozzaTong training data.
"""

import json
import os
import random
from typing import List, Dict, Any, Tuple
import jsonlines
from tqdm import tqdm


class TongDatasetGenerator:
    """Generator for creating training datasets with natural language to TONG code pairs."""
    
    def __init__(self):
        """Initialize the dataset generator."""
        self.basic_patterns = self._load_basic_patterns()
        self.intermediate_patterns = self._load_intermediate_patterns()
        self.advanced_patterns = self._load_advanced_patterns()
    
    def _load_basic_patterns(self) -> List[Dict[str, str]]:
        """Load basic TONG code patterns."""
        return [
            {
                "description": "Create a function that prints hello world",
                "tong_code": """fn hello_world() {
    print("Hello, World!")
}

hello_world()"""
            },
            {
                "description": "Write a function to add two numbers",
                "tong_code": """fn add(a, b) {
    return a + b
}

let result = add(5, 3)
print("Result:", result)"""
            },
            {
                "description": "Create a function to calculate factorial",
                "tong_code": """fn factorial(n) {
    if n <= 1 {
        return 1
    }
    return n * factorial(n - 1)
}

let fact = factorial(5)
print("Factorial of 5:", fact)"""
            },
            {
                "description": "Write a function to check if a number is even",
                "tong_code": """fn is_even(n) {
    return n % 2 == 0
}

let num = 42
if is_even(num) {
    print(num, "is even")
} else {
    print(num, "is odd")
}"""
            },
            {
                "description": "Create a function to find the maximum of two numbers",
                "tong_code": """fn max(a, b) {
    if a > b {
        return a
    } else {
        return b
    }
}

let maximum = max(10, 20)
print("Maximum:", maximum)"""
            },
            {
                "description": "Write a simple loop that prints numbers from 1 to 5",
                "tong_code": """for i in 1..6 {
    print("Number:", i)
}"""
            },
            {
                "description": "Create a function that calculates the square of a number",
                "tong_code": """fn square(x) {
    return x * x
}

let num = 7
let squared = square(num)
print("Square of", num, "is", squared)"""
            },
            {
                "description": "Write a function to calculate the area of a rectangle",
                "tong_code": """fn rectangle_area(width, height) {
    return width * height
}

let area = rectangle_area(5, 3)
print("Rectangle area:", area)"""
            },
        ]
    
    def _load_intermediate_patterns(self) -> List[Dict[str, str]]:
        """Load intermediate TONG code patterns."""
        return [
            {
                "description": "Create a function that works with arrays and calculates the sum",
                "tong_code": """fn array_sum(numbers) {
    let total = 0
    for num in numbers {
        total = total + num
    }
    return total
}

let data = [1, 2, 3, 4, 5]
let sum_result = array_sum(data)
print("Sum of array:", sum_result)"""
            },
            {
                "description": "Write a function to find the average of numbers in an array",
                "tong_code": """fn average(numbers) {
    if len(numbers) == 0 {
        return 0
    }
    let total = sum(numbers)
    return total / len(numbers)
}

let data = [10, 20, 30, 40, 50]
let avg = average(data)
print("Average:", avg)"""
            },
            {
                "description": "Create a function that filters even numbers from an array",
                "tong_code": """fn filter_even(numbers) {
    let evens = []
    for num in numbers {
        if num % 2 == 0 {
            evens.push(num)
        }
    }
    return evens
}

let data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
let even_numbers = filter_even(data)
print("Even numbers:", even_numbers)"""
            },
            {
                "description": "Write a function to calculate Fibonacci sequence up to n terms",
                "tong_code": """fn fibonacci(n) {
    if n <= 0 {
        return []
    }
    if n == 1 {
        return [0]
    }
    if n == 2 {
        return [0, 1]
    }
    
    let fib = [0, 1]
    for i in 2..n {
        let next = fib[i-1] + fib[i-2]
        fib.push(next)
    }
    return fib
}

let fib_sequence = fibonacci(10)
print("Fibonacci sequence:", fib_sequence)"""
            },
            {
                "description": "Create a function that reverses a string",
                "tong_code": """fn reverse_string(s) {
    let chars = s.chars()
    let reversed = []
    for i in (len(chars)-1)..0 {
        reversed.push(chars[i])
    }
    return reversed.join("")
}

let text = "Hello"
let reversed = reverse_string(text)
print("Reversed:", reversed)"""
            },
            {
                "description": "Write a function to check if a string is a palindrome",
                "tong_code": """fn is_palindrome(s) {
    let clean = s.lower().replace(" ", "")
    let reversed = ""
    for i in (len(clean)-1)..0 {
        reversed = reversed + clean[i]
    }
    return clean == reversed
}

let word = "racecar"
if is_palindrome(word) {
    print(word, "is a palindrome")
} else {
    print(word, "is not a palindrome")
}"""
            },
        ]
    
    def _load_advanced_patterns(self) -> List[Dict[str, str]]:
        """Load advanced TONG code patterns."""
        return [
            {
                "description": "Create a parallel function that processes large arrays efficiently",
                "tong_code": """fn parallel_square(numbers) {
    return parallel {
        map(numbers, |x| x * x)
    }
}

let large_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
let squared = parallel_square(large_array)
print("Squared numbers:", squared)"""
            },
            {
                "description": "Write a GPU kernel for matrix multiplication",
                "tong_code": """gpu_kernel fn matrix_multiply(a: Matrix<f32>, b: Matrix<f32>) -> Matrix<f32> {
    let rows = a.rows()
    let cols = b.cols()
    let result = Matrix::zeros(rows, cols)
    
    parallel {
        for i in 0..rows {
            for j in 0..cols {
                let sum = 0.0
                for k in 0..a.cols() {
                    sum += a[i][k] * b[k][j]
                }
                result[i][j] = sum
            }
        }
    }
    
    return result
}"""
            },
            {
                "description": "Create an async function for network data processing",
                "tong_code": """async fn fetch_and_process(urls: Array<String>) -> Array<ProcessedData> {
    let futures = []
    
    for url in urls {
        let future = async {
            let data = await http_get(url)
            return process_data(data)
        }
        futures.push(future)
    }
    
    let results = await futures
    return results
}"""
            },
            {
                "description": "Write a distributed function for big data processing",
                "tong_code": """distributed fn process_big_data(data: DistributedArray<f64>) -> f64 {
    let chunk_results = parallel {
        data.chunks().map(|chunk| {
            chunk.filter(|x| x > 0.0)
                 .map(|x| x * 2.0)
                 .reduce(|a, b| a + b)
        })
    }
    
    return chunk_results.sum()
}"""
            },
            {
                "description": "Create a function with pattern matching for data processing",
                "tong_code": """fn process_result(result: Result<String, Error>) -> String {
    match result {
        Ok(value) => {
            return "Success: " + value
        },
        Err(error) => {
            return "Error: " + error.message()
        }
    }
}

fn process_option(opt: Option<i32>) -> i32 {
    match opt {
        Some(value) if value > 0 => value * 2,
        Some(value) => value,
        None => 0
    }
}"""
            },
        ]
    
    def _generate_variations(self, pattern: Dict[str, str], num_variations: int = 3) -> List[Dict[str, str]]:
        """Generate variations of a base pattern."""
        variations = [pattern]
        
        for _ in range(num_variations - 1):
            variation = self._create_variation(pattern)
            variations.append(variation)
        
        return variations
    
    def _create_variation(self, pattern: Dict[str, str]) -> Dict[str, str]:
        """Create a variation of a pattern by modifying variable names and values."""
        code = pattern["tong_code"]
        description = pattern["description"]
        
        # Simple variations: change variable names and some values
        replacements = [
            ("numbers", "values"),
            ("data", "items"),
            ("result", "output"),
            ("num", "number"),
            ("total", "sum_val"),
            ("i", "index"),
            ("x", "value"),
            ("a", "first"),
            ("b", "second"),
        ]
        
        # Apply random replacements
        for old, new in random.sample(replacements, min(3, len(replacements))):
            if old in code:
                code = code.replace(old, new)
                description = description.replace(old, new)
        
        # Change some numeric values
        import re
        numbers = re.findall(r'\b\d+\b', code)
        for num in set(numbers):
            if random.random() < 0.3:  # 30% chance to change
                new_num = str(int(num) + random.randint(-5, 5))
                if int(new_num) > 0:  # Keep positive
                    code = code.replace(num, new_num, 1)
        
        return {
            "description": description,
            "tong_code": code
        }
    
    def _generate_synthetic_examples(self, num_examples: int) -> List[Dict[str, str]]:
        """Generate synthetic examples using templates."""
        synthetic_examples = []
        
        # Function templates
        function_templates = [
            {
                "description": "Create a function that calculates {operation} of two numbers",
                "code_template": """fn {func_name}(a, b) {{
    return a {op} b
}}

let result = {func_name}({val1}, {val2})
print("Result:", result)""",
                "operations": [
                    ("sum", "+", "add"),
                    ("difference", "-", "subtract"),
                    ("product", "*", "multiply"),
                    ("quotient", "/", "divide"),
                ]
            },
            {
                "description": "Write a function to check if a number is {condition}",
                "code_template": """fn is_{condition}(n) {{
    return n {check}
}}

let num = {value}
if is_{condition}(num) {{
    print(num, "is {condition}")
}} else {{
    print(num, "is not {condition}")
}}""",
                "conditions": [
                    ("positive", "> 0", 42),
                    ("negative", "< 0", -5),
                    ("zero", "== 0", 0),
                    ("greater_than_ten", "> 10", 15),
                ]
            }
        ]
        
        for _ in range(num_examples):
            template = random.choice(function_templates)
            
            if "operations" in template:
                operation, op, func_name = random.choice(template["operations"])
                description = template["description"].format(operation=operation)
                code = template["code_template"].format(
                    func_name=func_name,
                    op=op,
                    val1=random.randint(1, 20),
                    val2=random.randint(1, 20)
                )
            else:
                condition, check, value = random.choice(template["conditions"])
                description = template["description"].format(condition=condition)
                code = template["code_template"].format(
                    condition=condition,
                    check=check,
                    value=value
                )
            
            synthetic_examples.append({
                "description": description,
                "tong_code": code
            })
        
        return synthetic_examples
    
    def generate_dataset(
        self,
        output_dir: str,
        num_samples: int = 1000,
        difficulty: str = "mixed",
        include_synthetic: bool = True,
        variations_per_pattern: int = 3
    ):
        """
        Generate a training dataset.
        
        Args:
            output_dir: Output directory for the dataset
            num_samples: Total number of samples to generate
            difficulty: Difficulty level ('basic', 'intermediate', 'advanced', 'mixed')
            include_synthetic: Whether to include synthetic examples
            variations_per_pattern: Number of variations per base pattern
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_examples = []
        
        # Select patterns based on difficulty
        if difficulty == "basic":
            patterns = self.basic_patterns
        elif difficulty == "intermediate":
            patterns = self.intermediate_patterns
        elif difficulty == "advanced":
            patterns = self.advanced_patterns
        else:  # mixed
            patterns = self.basic_patterns + self.intermediate_patterns + self.advanced_patterns
        
        # Generate variations of existing patterns
        print("Generating pattern variations...")
        for pattern in tqdm(patterns):
            variations = self._generate_variations(pattern, variations_per_pattern)
            all_examples.extend(variations)
        
        # Generate synthetic examples if requested
        if include_synthetic:
            synthetic_count = max(0, num_samples - len(all_examples))
            if synthetic_count > 0:
                print(f"Generating {synthetic_count} synthetic examples...")
                synthetic_examples = self._generate_synthetic_examples(synthetic_count)
                all_examples.extend(synthetic_examples)
        
        # Shuffle and limit to requested number
        random.shuffle(all_examples)
        all_examples = all_examples[:num_samples]
        
        # Save as JSONL file
        output_file = os.path.join(output_dir, "train_dataset.jsonl")
        print(f"Saving {len(all_examples)} examples to {output_file}")
        
        with jsonlines.open(output_file, 'w') as writer:
            for example in all_examples:
                writer.write(example)
        
        # Save metadata
        metadata = {
            "total_examples": len(all_examples),
            "difficulty": difficulty,
            "include_synthetic": include_synthetic,
            "variations_per_pattern": variations_per_pattern,
            "base_patterns": len(patterns),
        }
        
        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset generated successfully!")
        print(f"Total examples: {len(all_examples)}")
        print(f"Output directory: {output_dir}")
        
        return output_file