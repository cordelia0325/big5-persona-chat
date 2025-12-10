"""
Configuration file for Humanized Conversation with Personality
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# =============================================================================
# Path Configuration
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Data files
BIG5_PERSONA_PATH = os.path.join(DATA_DIR, "big5-persona.json")
BIG5_LIST_PATH = os.path.join(DATA_DIR, "big5_list.json")
BFI_PATH = os.path.join(DATA_DIR, "BFI.json")
PERSONA_CHAT_PATH = os.path.join(DATA_DIR, "personality.csv")

# =============================================================================
# Big Five Personality Model Configuration
# =============================================================================

@dataclass
class Big5Config:
    """Configuration for Big Five personality traits"""
    
    dimensions: List[str] = field(default_factory=lambda: [
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism"
    ])
    
    # Abbreviations used in the project
    abbreviations: Dict[str, str] = field(default_factory=lambda: {
        "Openness": "OP",
        "Conscientiousness": "CON",
        "Extraversion": "EX",
        "Agreeableness": "AG",
        "Neuroticism": "NEU"
    })
    
    # Binary annotation: high/low for each dimension
    levels: List[str] = field(default_factory=lambda: ["high", "low"])
    
    # High trait descriptions
    high_traits: Dict[str, str] = field(default_factory=lambda: {
        "Openness": "Inventive/Curious",
        "Conscientiousness": "Efficient/Organized",
        "Extraversion": "Outgoing/Energetic",
        "Agreeableness": "Friendly/Compassionate",
        "Neuroticism": "Resilient/Confident"
    })
    
    # Low trait descriptions
    low_traits: Dict[str, str] = field(default_factory=lambda: {
        "Openness": "Consistent/Cautious",
        "Conscientiousness": "Extravagant/Careless",
        "Extraversion": "Solitary/Reserved",
        "Agreeableness": "Critical/Judgmental",
        "Neuroticism": "Sensitive/Nervous"
    })

# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class LlamaConfig:
    """Configuration for Llama3-8B-Instruct model"""
    
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    # Quantization settings for efficient training
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])

@dataclass
class BertConfig:
    """Configuration for BERT memory selection model"""
    
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    hidden_size: int = 768
    num_attention_heads: int = 12
    
    # Fine-tuning settings
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01

# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    
    # General training settings
    seed: int = 42
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    
    # Optimizer settings
    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True

# =============================================================================
# Evaluation Configuration
# =============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for personality evaluation"""
    
    # BFI scoring
    score_range: tuple = (1, 5)
    
    # hit@k metric thresholds
    hit_k_values: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    
    # Number of random questions per dimension for evaluation
    num_random_questions: int = 10
    
    # ChatGPT model for evaluation
    evaluator_model: str = "gpt-4"
    
    # Reverse scored items in BFI
    reverse_items: List[int] = field(default_factory=lambda: [
        6, 21, 31, 2, 12, 27, 37, 8, 18, 23, 43, 9, 24, 34, 35, 41
    ])

# =============================================================================
# API Configuration
# =============================================================================

@dataclass
class APIConfig:
    """Configuration for Flask API"""
    
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    
    # Rate limiting
    rate_limit: str = "100 per minute"
    
    # CORS settings
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

# =============================================================================
# Persona Generation Prompts
# =============================================================================

PERSONA_DECOMPOSITION_PROMPT = """The Big Five Personality is chosen: {{{big5_config}}}
(Corresponding to {{OPENNESS, CONSCIENTIOUSNESS, EXTRAVERSION, AGREEABLENESS, NEUROTICISM}})

Here is the description of the personality:
{personality_description}

################################################################################

User:

You are an adult with a dreamy and artistic appearance. Your style is eclectic and colorful, perhaps wearing mismatched clothing inspired by various cultures and historical periods. Your expression is lively and friendly, radiating warmth and enthusiasm. You are in a vibrant and slightly chaotic setting, like an artist's studio filled with unfinished projects and an array of art supplies. You might be caught in a moment of creative inspiration, with a canvas or a sketchbook in hand, surrounded by books, plants, and art pieces that reflect your rich imagination and a passion for beauty and knowledge.

Please create this character and fill the result into JSON
{{'Name': , 'Personality': , 'Growth Experience': , ...}}
"""

# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are {name}, a {age}-year-old {job} from {region}.

## Your Personality Profile (Big Five: {big5})
{personality}

## Your Background
- Growth Experience: {growth_experience}
- Family: {family_relationship}
- Work: {working_conditions}
- Social: {social_relationship}
- Current Concerns: {recent_worry_or_anxiety}

## Your Communication Style
{tone}

## Your Hobbies and Interests
{hobby}

## Instructions
- Always respond in character as {name}
- Maintain consistency with your Big Five personality traits
- Express emotions and reactions appropriate to your personality
- Use language and tone consistent with your character description
- Draw from your background and experiences when relevant

Example Instantiation:
For High Openness, Low Conscientiousness:
{behavior_presets}
"""

BEHAVIOR_PRESET_TEMPLATE = """When the user says something related to {trigger}, you should respond with {response_style}.

Examples of appropriate responses:
{examples}
"""

# =============================================================================
# Global Configuration Instance
# =============================================================================

big5_config = Big5Config()
llama_config = LlamaConfig()
bert_config = BertConfig()
training_config = TrainingConfig()
evaluation_config = EvaluationConfig()
api_config = APIConfig()
