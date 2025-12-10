"""
Utility Functions for Humanized Conversation with Personality

This module contains utility functions used across the project.
"""

import os
import json
import random
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import re

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Big Five Utilities
# =============================================================================

BIG5_DIMENSIONS = ['Openness', 'Conscientiousness', 'Extraversion', 
                   'Agreeableness', 'Neuroticism']

BIG5_ABBREVIATIONS = {
    'Openness': 'OP',
    'Conscientiousness': 'CON',
    'Extraversion': 'EX',
    'Agreeableness': 'AG',
    'Neuroticism': 'NEU'
}

def parse_big5_string(big5_str: str) -> Dict[str, str]:
    """
    Parse Big5 string format: {high, low, high, high, low}
    
    Returns:
        Dictionary mapping dimensions to levels
    """
    match = re.search(r'\{([^}]+)\}', big5_str)
    if match:
        values = [v.strip() for v in match.group(1).split(',')]
        if len(values) == 5:
            return dict(zip(BIG5_DIMENSIONS, values))
    return {}

def format_big5_string(big5_dict: Dict[str, str]) -> str:
    """
    Format Big5 dictionary as string: {high, low, high, high, low}
    """
    values = [big5_dict.get(dim, 'high') for dim in BIG5_DIMENSIONS]
    return "{" + ", ".join(values) + "}"

def generate_all_big5_combinations() -> List[Dict[str, str]]:
    """Generate all 32 possible Big5 combinations"""
    from itertools import product
    
    combinations = []
    for combo in product(['high', 'low'], repeat=5):
        combinations.append(dict(zip(BIG5_DIMENSIONS, combo)))
    
    return combinations

def get_big5_index(big5_dict: Dict[str, str]) -> int:
    """
    Convert Big5 dictionary to unique index (0-31)
    
    Binary encoding: high=1, low=0
    """
    index = 0
    for i, dim in enumerate(BIG5_DIMENSIONS):
        if big5_dict.get(dim, 'high') == 'high':
            index |= (1 << (4 - i))
    return index

def index_to_big5(index: int) -> Dict[str, str]:
    """Convert index (0-31) to Big5 dictionary"""
    big5 = {}
    for i, dim in enumerate(BIG5_DIMENSIONS):
        if index & (1 << (4 - i)):
            big5[dim] = 'high'
        else:
            big5[dim] = 'low'
    return big5

# =============================================================================
# Text Processing Utilities
# =============================================================================

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def clean_response(response: str) -> str:
    """Clean model response by removing artifacts"""
    # Remove special tokens
    tokens_to_remove = [
        '<|begin_of_text|>', '<|end_of_text|>',
        '<|start_header_id|>', '<|end_header_id|>',
        '<|eot_id|>', '[INST]', '[/INST]',
        '<s>', '</s>'
    ]
    
    for token in tokens_to_remove:
        response = response.replace(token, '')
    
    # Clean up whitespace
    response = re.sub(r'\s+', ' ', response).strip()
    
    return response

def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON object from text"""
    try:
        # Try direct parsing
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in text
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None

def count_tokens_approx(text: str) -> int:
    """Approximate token count (roughly 4 chars per token for English)"""
    return len(text) // 4

# =============================================================================
# File Utilities
# =============================================================================

def ensure_dir(path: str) -> str:
    """Ensure directory exists and return path"""
    os.makedirs(path, exist_ok=True)
    return path

def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data to JSON file"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# =============================================================================
# Conversation Utilities
# =============================================================================

def format_conversation(
    messages: List[Dict[str, str]],
    format_type: str = "llama3"
) -> str:
    """
    Format conversation for different model types
    
    Args:
        messages: List of {"role": "user/assistant/system", "content": "..."}
        format_type: "llama3", "chatml", or "simple"
    """
    if format_type == "llama3":
        formatted = "<|begin_of_text|>"
        for msg in messages:
            role = msg['role']
            content = msg['content']
            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        return formatted
    
    elif format_type == "chatml":
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return formatted
    
    else:  # simple
        formatted = ""
        for msg in messages:
            role = msg['role'].capitalize()
            content = msg['content']
            formatted += f"{role}: {content}\n"
        return formatted

def extract_turns(text: str) -> List[Dict[str, str]]:
    """Extract conversation turns from text"""
    turns = []
    lines = text.strip().split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        role = "user" if i % 2 == 0 else "assistant"
        turns.append({"role": role, "content": line})
    
    return turns

# =============================================================================
# Evaluation Utilities
# =============================================================================

def compute_accuracy(predicted: List[Any], actual: List[Any]) -> float:
    """Compute accuracy between predicted and actual values"""
    if len(predicted) != len(actual):
        raise ValueError("Lists must have same length")
    
    if len(predicted) == 0:
        return 0.0
    
    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    return correct / len(predicted)

def compute_hit_at_k(
    predicted_levels: Dict[str, str],
    target_levels: Dict[str, str],
    k: int
) -> bool:
    """Check if at least k dimensions match"""
    matches = sum(
        1 for dim in BIG5_DIMENSIONS
        if predicted_levels.get(dim) == target_levels.get(dim)
    )
    return matches >= k

def score_to_level(score: float, threshold: float = 3.0) -> str:
    """Convert score (1-5) to level (high/low)"""
    return "high" if score >= threshold else "low"

def reverse_score(score: int, max_score: int = 5) -> int:
    """Reverse score for reverse-scored items"""
    return max_score + 1 - score

# =============================================================================
# Random & Sampling Utilities
# =============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def sample_with_distribution(
    items: List[Any],
    weights: List[float],
    n: int
) -> List[Any]:
    """Sample n items according to weight distribution"""
    if len(items) != len(weights):
        raise ValueError("Items and weights must have same length")
    
    total = sum(weights)
    probs = [w / total for w in weights]
    
    return list(np.random.choice(items, size=n, p=probs, replace=True))

def stratified_sample(
    data: List[Any],
    key_func,
    n_per_group: int
) -> List[Any]:
    """Stratified sampling based on a key function"""
    from collections import defaultdict
    
    groups = defaultdict(list)
    for item in data:
        key = key_func(item)
        groups[key].append(item)
    
    samples = []
    for key, items in groups.items():
        if len(items) <= n_per_group:
            samples.extend(items)
        else:
            samples.extend(random.sample(items, n_per_group))
    
    return samples

# =============================================================================
# Logging Utilities
# =============================================================================

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Setup logging configuration"""
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

class ProgressLogger:
    """Simple progress logger for long operations"""
    
    def __init__(self, total: int, name: str = "Progress"):
        self.total = total
        self.name = name
        self.current = 0
        self.start_time = datetime.now()
    
    def update(self, n: int = 1):
        """Update progress"""
        self.current += n
        percent = (self.current / self.total) * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
        else:
            eta = 0
        
        logger.info(
            f"{self.name}: {self.current}/{self.total} ({percent:.1f}%) "
            f"ETA: {eta:.0f}s"
        )
    
    def finish(self):
        """Mark as finished"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"{self.name}: Completed in {elapsed:.1f}s")

# =============================================================================
# Personality Description Utilities
# =============================================================================

def get_trait_description(dimension: str, level: str) -> str:
    """Get description for a personality trait"""
    
    descriptions = {
        ('Openness', 'high'): "creative, curious, open to new experiences",
        ('Openness', 'low'): "practical, conventional, prefers routine",
        ('Conscientiousness', 'high'): "organized, disciplined, goal-oriented",
        ('Conscientiousness', 'low'): "flexible, spontaneous, adaptable",
        ('Extraversion', 'high'): "outgoing, energetic, sociable",
        ('Extraversion', 'low'): "reserved, introspective, prefers solitude",
        ('Agreeableness', 'high'): "cooperative, trusting, helpful",
        ('Agreeableness', 'low'): "competitive, skeptical, direct",
        ('Neuroticism', 'high'): "emotionally sensitive, prone to stress",
        ('Neuroticism', 'low'): "emotionally stable, calm, resilient"
    }
    
    return descriptions.get((dimension, level), "")

def get_full_personality_description(big5_dict: Dict[str, str]) -> str:
    """Get full personality description from Big5 dictionary"""
    
    descriptions = []
    for dim in BIG5_DIMENSIONS:
        level = big5_dict.get(dim, 'high')
        desc = get_trait_description(dim, level)
        if desc:
            descriptions.append(f"• {dim}: {desc}")
    
    return "\n".join(descriptions)

# =============================================================================
# Validation Utilities
# =============================================================================

def validate_persona_data(data: Dict) -> Tuple[bool, List[str]]:
    """Validate persona data structure"""
    
    errors = []
    required_fields = ['index', 'big-5', 'profile']
    profile_fields = ['name', 'personality', 'job']
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if 'profile' in data:
        for field in profile_fields:
            if field not in data['profile']:
                errors.append(f"Missing profile field: {field}")
    
    if 'big-5' in data:
        big5 = parse_big5_string(data['big-5'])
        if len(big5) != 5:
            errors.append(f"Invalid Big-5 format: {data['big-5']}")
    
    return len(errors) == 0, errors

def validate_bfi_data(data: Dict) -> Tuple[bool, List[str]]:
    """Validate BFI data structure"""
    
    errors = []
    
    if 'questions' not in data:
        errors.append("Missing 'questions' field")
    
    if 'questions' in data:
        for q_id, q_data in data['questions'].items():
            if 'dimension' not in q_data:
                errors.append(f"Question {q_id} missing 'dimension'")
            if 'origin_en' not in q_data and 'rewritten_en' not in q_data:
                errors.append(f"Question {q_id} missing question text")
    
    return len(errors) == 0, errors

# =============================================================================
# Main Execution (Testing)
# =============================================================================

if __name__ == "__main__":
    print("Testing Utility Functions")
    print("=" * 50)
    
    # Test Big5 utilities
    big5_str = "{high, low, high, high, low}"
    parsed = parse_big5_string(big5_str)
    print(f"\nParsed Big5: {parsed}")
    print(f"Formatted: {format_big5_string(parsed)}")
    print(f"Index: {get_big5_index(parsed)}")
    print(f"Back to dict: {index_to_big5(get_big5_index(parsed))}")
    
    # Test all combinations
    combos = generate_all_big5_combinations()
    print(f"\nTotal Big5 combinations: {len(combos)}")
    
    # Test text utilities
    long_text = "This is a very long text that needs to be truncated for display purposes."
    print(f"\nTruncated: {truncate_text(long_text, 30)}")
    
    # Test conversation formatting
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    print(f"\nFormatted (simple):\n{format_conversation(messages, 'simple')}")
    
    # Test personality description
    print(f"\nPersonality description:")
    print(get_full_personality_description(parsed))
