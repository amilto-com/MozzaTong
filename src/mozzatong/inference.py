"""
TONG code generation and inference module.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    GenerationConfig
)
from typing import Optional, List, Dict, Any
import os
import json


class TongGenerator:
    """Generator for converting natural language to TONG programming language."""
    
    def __init__(self, model_path: str):
        """
        Initialize the TONG generator.
        
        Args:
            model_path: Path to the trained MozzaTong model
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Load generation config if available
        config_path = os.path.join(model_path, "generation_config.json")
        if os.path.exists(config_path):
            self.generation_config = GenerationConfig.from_pretrained(model_path)
        else:
            self.generation_config = GenerationConfig(
                max_length=512,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
    
    def generate(
        self, 
        prompt: str, 
        max_length: int = 512,
        temperature: float = 0.7,
        num_beams: int = 4,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate TONG code from natural language prompt.
        
        Args:
            prompt: Natural language description of the desired code
            max_length: Maximum length of generated code
            temperature: Sampling temperature (higher = more creative)
            num_beams: Number of beams for beam search
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated TONG code as string
        """
        # Prepare the input prompt
        formatted_prompt = self._format_prompt(prompt)
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Create generation config
        gen_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            num_beams=num_beams,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        # Post-process the output
        tong_code = self._post_process_output(generated_text, formatted_prompt)
        
        return tong_code
    
    def generate_batch(
        self, 
        prompts: List[str], 
        max_length: int = 512,
        temperature: float = 0.7,
        num_beams: int = 4
    ) -> List[str]:
        """
        Generate TONG code for multiple prompts in batch.
        
        Args:
            prompts: List of natural language prompts
            max_length: Maximum length of generated code
            temperature: Sampling temperature
            num_beams: Number of beams for beam search
            
        Returns:
            List of generated TONG code strings
        """
        formatted_prompts = [self._format_prompt(p) for p in prompts]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Create generation config
        gen_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            num_beams=num_beams,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config
            )
        
        # Decode outputs
        results = []
        for i, output in enumerate(outputs):
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            tong_code = self._post_process_output(generated_text, formatted_prompts[i])
            results.append(tong_code)
        
        return results
    
    def _format_prompt(self, prompt: str) -> str:
        """
        Format the input prompt for the model.
        
        Args:
            prompt: Raw natural language prompt
            
        Returns:
            Formatted prompt string
        """
        # Add instruction prefix to guide the model
        formatted = f"Convert the following natural language description to TONG programming language code:\n\nDescription: {prompt}\n\nTONG Code:"
        return formatted
    
    def _post_process_output(self, generated_text: str, original_prompt: str) -> str:
        """
        Post-process the generated output to extract clean TONG code.
        
        Args:
            generated_text: Raw generated text from model
            original_prompt: Original formatted prompt
            
        Returns:
            Clean TONG code
        """
        # Remove the original prompt from the output
        if original_prompt in generated_text:
            tong_code = generated_text.replace(original_prompt, "").strip()
        else:
            tong_code = generated_text.strip()
        
        # Clean up common issues
        tong_code = tong_code.replace("TONG Code:", "").strip()
        
        # Remove any leading/trailing whitespace and normalize
        lines = tong_code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Convert the following"):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "model_type": self.model.config.model_type,
            "vocab_size": len(self.tokenizer),
            "max_position_embeddings": getattr(self.model.config, "max_position_embeddings", "unknown"),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }


class TongCodeValidator:
    """Validator for generated TONG code syntax and semantics."""
    
    def __init__(self):
        """Initialize the validator."""
        self.tong_keywords = {
            "fn", "let", "var", "if", "else", "return", "while", "for", 
            "parallel", "gpu_kernel", "distributed", "match", "async", "await"
        }
        
        self.tong_types = {
            "i8", "i16", "i32", "i64", "i128", 
            "u8", "u16", "u32", "u64", "u128",
            "f16", "f32", "f64", "f128",
            "bool", "char", "String", "Array"
        }
    
    def validate_syntax(self, code: str) -> Dict[str, Any]:
        """
        Validate TONG code syntax.
        
        Args:
            code: TONG code to validate
            
        Returns:
            Validation result with errors and warnings
        """
        errors = []
        warnings = []
        
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            # Check for basic syntax patterns
            if line.startswith('fn ') and not line.endswith('{') and '{' not in line:
                if not line.endswith(')'):
                    errors.append(f"Line {i}: Function declaration missing closing parenthesis")
            
            # Check for balanced braces
            open_braces = line.count('{')
            close_braces = line.count('}')
            if open_braces != close_braces and (open_braces > 0 or close_braces > 0):
                warnings.append(f"Line {i}: Unbalanced braces - check function/block structure")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "line_count": len([l for l in lines if l.strip()]),
            "has_functions": any("fn " in line for line in lines),
            "uses_tong_keywords": any(keyword in code for keyword in self.tong_keywords)
        }