"""
Basic tests for MozzaTong functionality.
"""

import pytest
import tempfile
import os
import json
from mozzatong.dataset import TongDatasetGenerator
from mozzatong.inference import TongCodeValidator


class TestTongDatasetGenerator:
    """Test the dataset generation functionality."""
    
    def test_basic_patterns_exist(self):
        """Test that basic patterns are available."""
        generator = TongDatasetGenerator()
        assert len(generator.basic_patterns) > 0
        
        # Check structure of first pattern
        pattern = generator.basic_patterns[0]
        assert "description" in pattern
        assert "tong_code" in pattern
        assert isinstance(pattern["description"], str)
        assert isinstance(pattern["tong_code"], str)
    
    def test_dataset_generation(self):
        """Test dataset generation with small sample."""
        generator = TongDatasetGenerator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = generator.generate_dataset(
                output_dir=temp_dir,
                num_samples=10,
                difficulty="basic",
                include_synthetic=False
            )
            
            # Check files were created
            assert os.path.exists(output_file)
            assert os.path.exists(os.path.join(temp_dir, "metadata.json"))
            
            # Check dataset content
            examples = []
            with open(output_file, 'r') as f:
                for line in f:
                    examples.append(json.loads(line))
            
            assert len(examples) > 0
            assert len(examples) <= 10
            
            # Check example structure
            example = examples[0]
            assert "description" in example
            assert "tong_code" in example


class TestTongCodeValidator:
    """Test the TONG code validation functionality."""
    
    def test_validator_initialization(self):
        """Test validator can be initialized."""
        validator = TongCodeValidator()
        assert len(validator.tong_keywords) > 0
        assert len(validator.tong_types) > 0
        assert "fn" in validator.tong_keywords
        assert "i32" in validator.tong_types
    
    def test_valid_function_syntax(self):
        """Test validation of valid TONG function syntax."""
        validator = TongCodeValidator()
        
        valid_code = """fn add(a, b) {
    return a + b
}

let result = add(5, 3)
print("Result:", result)"""
        
        result = validator.validate_syntax(valid_code)
        assert result["valid"] is True
        assert result["has_functions"] is True
        assert result["uses_tong_keywords"] is True
        assert len(result["errors"]) == 0
    
    def test_empty_code_validation(self):
        """Test validation of empty code."""
        validator = TongCodeValidator()
        
        result = validator.validate_syntax("")
        assert result["valid"] is True  # Empty code is technically valid
        assert result["line_count"] == 0
        assert result["has_functions"] is False
    
    def test_comments_ignored(self):
        """Test that comments are properly ignored."""
        validator = TongCodeValidator()
        
        code_with_comments = """// This is a comment
fn test() {
    // Another comment
    return 42
}"""
        
        result = validator.validate_syntax(code_with_comments)
        assert result["valid"] is True
        assert result["has_functions"] is True


class TestMozzaTongIntegration:
    """Integration tests for MozzaTong components."""
    
    def test_dataset_generator_patterns_valid(self):
        """Test that generated dataset patterns have valid TONG syntax."""
        generator = TongDatasetGenerator()
        validator = TongCodeValidator()
        
        # Test basic patterns
        for pattern in generator.basic_patterns[:3]:  # Test first 3
            result = validator.validate_syntax(pattern["tong_code"])
            assert result["valid"] is True, f"Invalid pattern: {pattern['description']}"
    
    def test_synthetic_examples_structure(self):
        """Test that synthetic examples have proper structure."""
        generator = TongDatasetGenerator()
        
        synthetic_examples = generator._generate_synthetic_examples(5)
        assert len(synthetic_examples) == 5
        
        for example in synthetic_examples:
            assert "description" in example
            assert "tong_code" in example
            assert len(example["description"]) > 0
            assert len(example["tong_code"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])