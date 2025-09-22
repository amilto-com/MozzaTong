"""
Training module for MozzaTong - Fine-tuning LLMs for natural language to TONG conversion.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset, load_dataset
import json
import os
from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TongTrainer:
    """Trainer for fine-tuning models on natural language to TONG conversion."""
    
    def __init__(
        self, 
        model_name: str = "microsoft/CodeT5-small",
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1
    ):
        """
        Initialize the trainer.
        
        Args:
            model_name: Base model to fine-tune
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_rank: LoRA rank parameter
            lora_alpha: LoRA alpha parameter  
            lora_dropout: LoRA dropout rate
        """
        self.model_name = model_name
        self.use_lora = use_lora
        self.lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        ) if use_lora else None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
    def _load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Apply LoRA if requested
        if self.use_lora:
            logger.info("Applying LoRA configuration...")
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.print_trainable_parameters()
    
    def _load_dataset(self, dataset_path: str) -> Dataset:
        """
        Load and prepare the training dataset.
        
        Args:
            dataset_path: Path to the dataset (JSON or JSONL file)
            
        Returns:
            Loaded and processed dataset
        """
        logger.info(f"Loading dataset from: {dataset_path}")
        
        # Load dataset based on file extension
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
        elif dataset_path.endswith('.jsonl'):
            data = []
            with open(dataset_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            # Try to load as a HuggingFace dataset
            dataset = load_dataset(dataset_path)
            return dataset['train'] if 'train' in dataset else dataset
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(data)
        
        logger.info(f"Loaded {len(dataset)} examples")
        return dataset
    
    def _preprocess_data(self, dataset: Dataset, max_input_length: int = 512, max_target_length: int = 512) -> Dataset:
        """
        Preprocess the dataset for training.
        
        Args:
            dataset: Raw dataset
            max_input_length: Maximum input sequence length
            max_target_length: Maximum target sequence length
            
        Returns:
            Preprocessed dataset
        """
        def preprocess_function(examples):
            # Format inputs
            inputs = [f"Convert to TONG: {desc}" for desc in examples["description"]]
            targets = examples["tong_code"]
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs, 
                max_length=max_input_length, 
                truncation=True,
                padding=False
            )
            
            # Tokenize targets
            labels = self.tokenizer(
                targets, 
                max_length=max_target_length, 
                truncation=True,
                padding=False
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        logger.info("Preprocessing dataset...")
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return processed_dataset
    
    def train(
        self,
        dataset_path: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        eval_steps: int = 500,
        save_steps: int = 1000,
        logging_steps: int = 100,
        max_grad_norm: float = 1.0,
        weight_decay: float = 0.01,
        validation_split: float = 0.1,
        early_stopping_patience: int = 3
    ):
        """
        Train the MozzaTong model.
        
        Args:
            dataset_path: Path to training dataset
            output_dir: Directory to save the trained model
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            eval_steps: Evaluation frequency
            save_steps: Save frequency
            logging_steps: Logging frequency
            max_grad_norm: Maximum gradient norm for clipping
            weight_decay: Weight decay
            validation_split: Fraction of data for validation
            early_stopping_patience: Early stopping patience
        """
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Load and preprocess dataset
        dataset = self._load_dataset(dataset_path)
        processed_dataset = self._preprocess_data(dataset)
        
        # Split into train/validation
        if validation_split > 0:
            train_dataset, eval_dataset = train_test_split(
                processed_dataset, 
                test_size=validation_split,
                random_state=42
            )
            train_dataset = Dataset.from_list(train_dataset)
            eval_dataset = Dataset.from_list(eval_dataset)
        else:
            train_dataset = processed_dataset
            eval_dataset = None
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            logging_dir=f"{output_dir}/logs",
            logging_steps=logging_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=eval_steps if eval_dataset else None,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            report_to="none",  # Disable wandb for now
            push_to_hub=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
        )
        
        # Callbacks
        callbacks = []
        if eval_dataset and early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            "base_model": self.model_name,
            "use_lora": self.use_lora,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dataset_size": len(train_dataset),
            "validation_size": len(eval_dataset) if eval_dataset else 0,
        }
        
        with open(os.path.join(output_dir, "training_info.json"), "w") as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("Training completed successfully!")
        
        return trainer.state.log_history
    
    def evaluate(
        self,
        model_path: str,
        test_dataset_path: str,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to trained model
            test_dataset_path: Path to test dataset
            batch_size: Evaluation batch size
            
        Returns:
            Evaluation metrics
        """
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # Load test dataset
        test_dataset = self._load_dataset(test_dataset_path)
        processed_test_dataset = self._preprocess_data(test_dataset)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # Training arguments for evaluation
        eval_args = TrainingArguments(
            output_dir="./eval_output",
            per_device_eval_batch_size=batch_size,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        # Initialize trainer for evaluation
        trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=processed_test_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Evaluate
        logger.info("Evaluating model...")
        eval_results = trainer.evaluate()
        
        return eval_results