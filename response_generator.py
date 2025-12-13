"""
Response Generator Module for Humanized Conversation with Personality

This module implements the Llama3-8B-Instruct based response generation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

from config import llama_config, training_config, SYSTEM_PROMPT_TEMPLATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConversationSample:
    """A single conversation sample for training"""
    system_prompt: str
    messages: List[Dict[str, str]]  # [{"role": "user/assistant", "content": "..."}]
    persona_big5: str
    selected_memory: Optional[str] = None

# =============================================================================
# Training Dataset
# =============================================================================

class PersonaConversationDataset(Dataset):
    """
    Dataset for training the response generator
    
    Each sample contains:
    - System prompt with persona
    - Conversation history
    - Target response
    """
    
    def __init__(
        self,
        samples: List[ConversationSample],
        tokenizer,
        max_length: int = 2048
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Format as chat template
        formatted = self._format_conversation(sample)
        
        # Tokenize
        encoding = self.tokenizer(
            formatted,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Labels are same as input_ids for causal LM
        labels = encoding['input_ids'].clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }
    
    def _format_conversation(self, sample: ConversationSample) -> str:
        """Format conversation for Llama3 Instruct"""
        
        # Llama3 Instruct format
        formatted = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sample.system_prompt}"
        
        # Add selected memory if available
        if sample.selected_memory:
            formatted += f"\n\nRelevant memory: {sample.selected_memory}"
        
        formatted += "<|eot_id|>"
        
        # Add conversation turns
        for msg in sample.messages:
            role = msg['role']
            content = msg['content']
            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        
        return formatted

# =============================================================================
# Response Generator Model
# =============================================================================

class ResponseGenerator:
    """Llama3-8B-Instruct based response generator"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_quantization: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.use_quantization = use_quantization
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path or llama_config.model_name
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if model_path:
            self.model = self._load_model(model_path)
        else:
            self.model = self._create_model()
        
        self.model.eval()
    
    def _create_model(self) -> AutoModelForCausalLM:
        """Create the base model with optional quantization"""
        
        if self.use_quantization and self.device == "cuda":
            # 4-bit quantization for efficient inference
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=llama_config.load_in_4bit,
                bnb_4bit_quant_type=llama_config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, llama_config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                llama_config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                llama_config.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
        
        return model
    
    def _load_model(self, model_path: str) -> AutoModelForCausalLM:
        """Load fine-tuned model from checkpoint"""
        
        # Try loading as PEFT model first
        try:
            base_model = self._create_model()
            model = PeftModel.from_pretrained(base_model, model_path)
            logger.info(f"Loaded PEFT model from {model_path}")
            return model
        except Exception as e:
            logger.warning(f"Could not load as PEFT model: {e}")
        
        # Fall back to full model loading
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        return model
    
    def generate_response(
        self,
        context: List[Dict[str, str]],
        persona: str,
        selected_memory: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None
    ) -> str:
        """
        Generate a personality-consistent response
        
        Args:
            context: Conversation history [{"role": "user/assistant", "content": "..."}]
            persona: System prompt with persona description
            selected_memory: Relevant memory for the response
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated response text
        """
        # Use config defaults if not specified
        temperature = temperature or llama_config.temperature
        top_p = top_p or llama_config.top_p
        top_k = top_k or llama_config.top_k
        
        # Format input
        input_text = self._format_input(context, persona, selected_memory)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=llama_config.max_length - max_new_tokens
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=llama_config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new response
        response = self._extract_response(full_response, input_text)
        
        return response
    
    def _format_input(
        self,
        context: List[Dict[str, str]],
        persona: str,
        selected_memory: Optional[str]
    ) -> str:
        """Format input for Llama3 Instruct"""
        
        # System prompt
        formatted = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{persona}"
        
        if selected_memory:
            formatted += f"\n\nRelevant context from memory: {selected_memory}"
        
        formatted += "<|eot_id|>"
        
        # Conversation history
        for msg in context:
            role = msg['role']
            content = msg['content']
            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        
        # Prompt for assistant response
        formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return formatted
    
    def _extract_response(self, full_response: str, input_text: str) -> str:
        """Extract just the generated response from full output"""
        
        # Remove the input portion
        if "assistant" in full_response:
            # Find the last assistant response
            parts = full_response.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
                # Clean up any remaining tags
                response = response.replace("<|eot_id|>", "").strip()
                return response
        
        return full_response

# =============================================================================
# Fine-tuning Trainer
# =============================================================================

class ResponseGeneratorTrainer:
    """
    Trainer for fine-tuning the response generator with LoRA
    
    Uses parameter-efficient fine-tuning (PEFT) with QLoRA
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name or llama_config.model_name
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization
        self.model = self._prepare_model()
    
    def _prepare_model(self):
        """Prepare model for training with QLoRA"""
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=llama_config.lora_r,
            lora_alpha=llama_config.lora_alpha,
            target_modules=llama_config.lora_target_modules,
            lora_dropout=llama_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")
        
        return model
    
    def train(
        self,
        train_samples: List[ConversationSample],
        val_samples: Optional[List[ConversationSample]] = None,
        output_dir: str = "./outputs/response_generator",
        num_epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None
    ):
        """
        Fine-tune the response generator
        
        Args:
            train_samples: Training conversation samples
            val_samples: Validation samples
            output_dir: Directory to save checkpoints
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        # Use config defaults
        num_epochs = num_epochs or training_config.num_epochs
        batch_size = batch_size or training_config.batch_size
        learning_rate = learning_rate or training_config.learning_rate
        
        # Create datasets
        train_dataset = PersonaConversationDataset(
            train_samples,
            self.tokenizer,
            max_length=llama_config.max_length
        )
        
        val_dataset = None
        if val_samples:
            val_dataset = PersonaConversationDataset(
                val_samples,
                self.tokenizer,
                max_length=llama_config.max_length
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=training_config.warmup_ratio,
            lr_scheduler_type=training_config.lr_scheduler_type,
            logging_steps=training_config.logging_steps,
            save_steps=training_config.save_steps,
            eval_steps=training_config.eval_steps if val_dataset else None,
            evaluation_strategy="steps" if val_dataset else "no",
            save_total_limit=training_config.save_total_limit,
            fp16=training_config.fp16,
            bf16=training_config.bf16,
            max_grad_norm=training_config.max_grad_norm,
            optim=training_config.optimizer,
            report_to="none"  # Disable wandb/tensorboard
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
    
    def save_model(self, output_dir: str):
        """Save the trained model"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

# =============================================================================
# Persona-based Response Generator (High-level interface)
# =============================================================================

class PersonaResponseGenerator:
    """
    High-level interface for generating persona-consistent responses
    
    Combines memory selection and response generation
    """
    
    def __init__(
        self,
        response_generator: ResponseGenerator,
        memory_selector = None
    ):
        self.generator = response_generator
        self.memory_selector = memory_selector
    
    def generate(
        self,
        user_message: str,
        persona: str,
        conversation_history: List[Dict[str, str]] = None,
        memories: List[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response considering persona and memories
        
        Implementation of the objective function from Section 6.1:
        P(y_k | C, P, M) ≈ P_θ(y_k | C, P, m_k) · P_η(m_k | C, P, M)
        
        Args:
            user_message: Current user message
            persona: Persona description/system prompt
            conversation_history: Previous conversation turns
            memories: Available memories
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        conversation_history = conversation_history or []
        
        # Build full context
        context = conversation_history + [{"role": "user", "content": user_message}]
        
        # Select relevant memory if available
        selected_memory = None
        if memories and self.memory_selector:
            context_text = " ".join([m["content"] for m in context])
            selected = self.memory_selector.select_memory(
                context_text,
                persona,
                memories,
                top_k=1
            )
            if selected:
                selected_memory = selected[0][0]
        
        # Generate response
        response = self.generator.generate_response(
            context=context,
            persona=persona,
            selected_memory=selected_memory,
            **kwargs
        )
        
        return response

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("Response Generator Module")
    print("=" * 50)
    
    # Example usage (without actual model loading for testing)
    print("\nExample conversation format:")
    
    sample_persona = """You are Elena Ramirez, a 34-year-old Nonprofit Director from Chicago.
    
Big Five Personality: {high, high, high, high, high}

You are deeply passionate about your work and constantly exploring new ways to make a difference. 
You are highly organized and meticulous in your planning, driven by a strong sense of duty and ethics.
Your extroverted nature makes you a natural leader, and you are excellent at motivating your team.
However, you often feel the weight of responsibilities intensely, leading to periods of stress and anxiety.

Respond as Elena would, maintaining consistency with her personality traits."""

    sample_context = [
        {"role": "user", "content": "Hi Elena! How's the fundraising going?"},
    ]
    
    print(f"\nPersona:\n{sample_persona[:200]}...")
    print(f"\nContext: {sample_context}")
    print("\n[Response would be generated here with actual model]")
