"""
Data Processing Module for Humanized Conversation with Personality

This module handles:
1. Loading and preprocessing the BIG-5 dataset
2. Loading and preprocessing the PERSONA-CHAT dataset
3. Creating training samples for the personality model
"""

import json
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import random

from config import (
    BIG5_PERSONA_PATH,
    BIG5_LIST_PATH,
    BFI_PATH,
    PERSONA_CHAT_PATH,
    big5_config,
    SYSTEM_PROMPT_TEMPLATE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Persona:
    """Represents a persona with Big Five personality traits"""
    index: int
    big5: str  # e.g., "{high, high, high, high, high}"
    name: str
    gender: str
    age: str
    region: str
    tone: str
    job: str
    personality: str
    advantages_and_disadvantages: str
    hobby: str
    growth_experience: str
    family_relationship: str
    working_conditions: str
    social_relationship: str
    emotional_state: str
    living_conditions: str
    recent_worry_or_anxiety: str
    additional_information: str
    
    @classmethod
    def from_dict(cls, data: dict, fallback_index: int = 0) -> 'Persona':
        """Create Persona from dictionary"""
        profile = data.get('profile', {})
        
        # Handle index: try 'index', then parse from 'uid', then use fallback
        index = data.get('index')
        if index is None:
            uid = data.get('uid', '')
            # Parse index from uid like "train-0000" or "test-0001"
            if uid and '-' in uid:
                try:
                    index = int(uid.split('-')[-1])
                except ValueError:
                    index = fallback_index
            else:
                index = fallback_index
        
        return cls(
            index=index,
            big5=data.get('big-5', ''),
            name=profile.get('name', ''),
            gender=profile.get('gender', ''),
            age=profile.get('age', ''),
            region=profile.get('region', ''),
            tone=profile.get('tone', ''),
            job=profile.get('job', ''),
            personality=profile.get('personality', ''),
            advantages_and_disadvantages=profile.get('advantages_and_disadvantages', ''),
            hobby=profile.get('hobby', ''),
            growth_experience=profile.get('growth_experience', ''),
            family_relationship=profile.get('family_relationship', ''),
            working_conditions=profile.get('working_conditions', ''),
            social_relationship=profile.get('social_relationship', ''),
            emotional_state=profile.get('emotional_state', ''),
            living_conditions=profile.get('living_conditions', ''),
            recent_worry_or_anxiety=profile.get('recent_worry_or_anxiety', ''),
            additional_information=profile.get('additional_information', '')
        )
    
    def to_system_prompt(self) -> str:
        """Convert persona to system prompt for LLM"""
        # Generate behavior presets based on Big-5 traits
        behavior_presets = self._generate_behavior_presets()
        return SYSTEM_PROMPT_TEMPLATE.format(
            name=self.name,
            age=self.age,
            job=self.job,
            region=self.region,
            big5=self.big5,
            personality=self.personality,
            growth_experience=self.growth_experience,
            family_relationship=self.family_relationship,
            working_conditions=self.working_conditions,
            social_relationship=self.social_relationship,
            recent_worry_or_anxiety=self.recent_worry_or_anxiety,
            tone=self.tone,
            hobby=self.hobby,
            behavior_presets=behavior_presets
        )
    
    def _generate_behavior_presets(self) -> str:
        """Generate behavior presets based on Big-5 personality traits using big5_list.json"""
        import os
        
        # Load Big-5 descriptions from data file
        big5_list_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "big5_list.json")
        
        try:
            with open(big5_list_path, 'r', encoding='utf-8') as f:
                big5_descriptions = json.load(f)[0]  # First element contains the descriptions
        except (FileNotFoundError, json.JSONDecodeError, IndexError):
            # Fallback if file not found
            return ""
        
        big5 = self.get_big5_dict()
        presets = []
        
        for dimension, level in big5.items():
            if dimension in big5_descriptions and level in big5_descriptions[dimension]:
                description = big5_descriptions[dimension][level]
                presets.append(f"**{dimension} ({level.capitalize()})**: {description}")
        
        return '\n\n'.join(presets)
    
    def get_big5_dict(self) -> Dict[str, str]:
        """Parse Big5 string to dictionary"""
        # Parse "{high, low, high, high, low}" format
        match = re.search(r'\{([^}]+)\}', self.big5)
        if match:
            values = [v.strip() for v in match.group(1).split(',')]
            if len(values) == 5:
                return {
                    'Openness': values[0],
                    'Conscientiousness': values[1],
                    'Extraversion': values[2],
                    'Agreeableness': values[3],
                    'Neuroticism': values[4]
                }
        return {}

    def get_memory_bank(self) -> List[str]:
        """Extract memory-like statements from profile fields"""
        memories = []
        fields = [
            self.growth_experience,
            self.family_relationship,
            self.working_conditions,
            self.social_relationship,
            self.living_conditions,
            self.hobby,
            self.additional_information
        ]
        for field in fields:
            if field:
                # Split compound sentences into individual memory units if possible
                parts = [p.strip() for p in field.split('.') if p.strip()]
                memories.extend(parts)
        return memories

@dataclass
class DialogueTurn:
    """Represents a single turn in a dialogue"""
    role: str  # "user" or "assistant"
    content: str

@dataclass
class Conversation:
    """Represents a complete conversation with persona"""
    persona: str
    turns: List[DialogueTurn]
    
    def to_training_format(self, system_prompt: str = "") -> List[Dict]:
        """Convert to training format for LLM fine-tuning"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        for turn in self.turns:
            messages.append({"role": turn.role, "content": turn.content})
        
        return messages
    
    def get_context_before(self, target_turn: DialogueTurn) -> str:
        """Get conversation history before the target turn"""
        context_parts = []
        for turn in self.turns:
            if turn == target_turn:
                break
            context_parts.append(f"{turn.role}: {turn.content}")
        return "\n".join(context_parts)

# =============================================================================
# Data Loading Functions
# =============================================================================

class DataLoader:
    """Handles loading and preprocessing of all datasets"""
    
    def __init__(self):
        self.personas: List[Persona] = []
        self.big5_descriptions: Dict[str, str] = {}
        self.bfi_questions: Dict = {}
        self.persona_chat_data: List[Conversation] = []
    
    def load_big5_personas(self, filepath: str = BIG5_PERSONA_PATH) -> List[Persona]:
        """Load Big-5 Persona dataset"""
        logger.info(f"Loading Big-5 personas from {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Pass fallback_index to ensure unique indices
            self.personas = [Persona.from_dict(item, fallback_index=i) for i, item in enumerate(data)]
            logger.info(f"Loaded {len(self.personas)} personas")
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            self.personas = []
            
        return self.personas
    
    def load_big5_descriptions(self, filepath: str = BIG5_LIST_PATH) -> Dict[str, str]:
        """Load Big Five trait descriptions"""
        logger.info(f"Loading Big-5 descriptions from {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.big5_descriptions = {item['type']: item['description'] for item in data}
            logger.info(f"Loaded {len(self.big5_descriptions)} trait descriptions")
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            
        return self.big5_descriptions
    
    def load_bfi_questions(self, filepath: str = BFI_PATH) -> Dict:
        """Load Big Five Inventory questions for evaluation"""
        logger.info(f"Loading BFI questions from {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.bfi_questions = json.load(f)
            logger.info(f"Loaded BFI with {len(self.bfi_questions.get('questions', {}))} questions")
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            
        return self.bfi_questions
    
    def load_persona_chat(self, filepath: str = PERSONA_CHAT_PATH) -> List[Conversation]:
        """Load PERSONA-CHAT dataset"""
        logger.info(f"Loading PERSONA-CHAT from {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            self.persona_chat_data = []
            
            for _, row in df.iterrows():
                persona = row['Persona'].strip() if pd.notna(row['Persona']) else ""
                chat = row['chat'].strip() if pd.notna(row['chat']) else ""
                
                if not chat:
                    continue
                
                # Parse dialogue turns
                turns = self._parse_dialogue(chat)
                
                if turns:
                    self.persona_chat_data.append(Conversation(
                        persona=persona,
                        turns=turns
                    ))
            
            logger.info(f"Loaded {len(self.persona_chat_data)} conversations")
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            self.persona_chat_data = []
            
        return self.persona_chat_data
    
    def _parse_dialogue(self, chat_text: str) -> List[DialogueTurn]:
        """Parse raw dialogue text into structured turns"""
        turns = []
        lines = chat_text.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Alternate between user and assistant
            role = "user" if i % 2 == 0 else "assistant"
            turns.append(DialogueTurn(role=role, content=line))
        
        return turns
    
    def load_all(self) -> Dict:
        """Load all datasets"""
        return {
            'personas': self.load_big5_personas(),
            'big5_descriptions': self.load_big5_descriptions(),
            'bfi_questions': self.load_bfi_questions(),
            'persona_chat': self.load_persona_chat()
        }

# =============================================================================
# Dataset Preparation for Training
# =============================================================================

class TrainingDatasetBuilder:
    """
    Builds training datasets for the personality model
    """
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
    
    def build_persona_response_dataset(self) -> List[Dict]:
        """
        Build dataset for response generation training
        Format: {system_prompt, conversations}
        """
        dataset = []
        
        for persona in self.data_loader.personas:
            system_prompt = persona.to_system_prompt()
            
            # Find matching conversations from PERSONA-CHAT (Simple heuristic mapping for now)
            # In a real scenario, you would map Big-5 personas to specific dialogue datasets
            # or rely on synthetic generation.
            matching_convs = self._find_matching_conversations(persona)
            
            for conv in matching_convs:
                training_sample = {
                    'persona_index': persona.index,
                    'big5': persona.big5,
                    'system_prompt': system_prompt,
                    'messages': conv.to_training_format(system_prompt)
                }
                dataset.append(training_sample)
        
        return dataset
    
    def build_memory_selection_dataset(self, similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Build memory selection samples with semantic similarity labeling.
        
        This method uses a SentenceTransformer to compute cosine similarity between 
        assistant responses and candidate memories.
        
        Positive samples: embedding similarity > threshold
        Negative samples: similarity <= threshold
        """
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
        except ImportError:
            logger.warning("sentence-transformers or sklearn not found. Falling back to keyword matching.")
            return self._build_memory_selection_heuristic()

        logger.info("Building memory selection dataset using semantic similarity (BERT)...")
        
        # Load lightweight model for data preprocessing
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        samples = []
        
        # Iterate over PERSONA-CHAT conversations which contain both Memory (Persona) and Dialogue
        for conv in self.data_loader.persona_chat_data:
            # Extract memories from the persona description block of the conversation
            memories = self._extract_memories(conv.persona)
            if not memories:
                continue
                
            # Pre-compute memory embeddings for efficiency
            memory_embeddings = encoder.encode(memories)
            
            for i, turn in enumerate(conv.turns):
                # We only care about predicting memory relevance for Assistant responses
                if turn.role == "assistant":
                    
                    # Compute similarity between response and each memory
                    response_emb = encoder.encode([turn.content])
                    
                    # Similarities: (1, num_memories) -> flatten to (num_memories,)
                    similarities = cosine_similarity(response_emb, memory_embeddings)[0]
                    
                    # Get context (history before this turn)
                    context = conv.get_context_before(turn)
                    if not context:
                        continue # Skip empty context
                    
                    for mem_idx, memory in enumerate(memories):
                        score = similarities[mem_idx]
                        
                        # Labeling logic
                        label = 1 if score > similarity_threshold else 0
                        
                        # Data Balancing Strategy:
                        # 1. Always keep positive samples
                        # 2. Keep negative samples if they are "hard negatives" (somewhat similar but below threshold)
                        # 3. Or randomly sample negatives to avoid explosion
                        if label == 1 or random.random() < 0.3:
                            samples.append({
                                'context': context,
                                'memory': memory,
                                'label': label,
                                'similarity_score': float(score)
                            })
                            
        logger.info(f"Generated {len(samples)} memory selection samples.")
        return samples

    def _build_memory_selection_heuristic(self) -> List[Dict]:
        """Fallback method using keyword overlap"""
        dataset = []
        for conv in self.data_loader.persona_chat_data:
            memories = self._extract_memories(conv.persona)
            for i, turn in enumerate(conv.turns):
                if turn.role == "assistant" and i > 0:
                    context = " ".join([t.content for t in conv.turns[:i]])
                    for memory in memories:
                        is_relevant = self._is_memory_relevant(memory, turn.content)
                        dataset.append({
                            'context': context,
                            'memory': memory,
                            'label': 1 if is_relevant else 0
                        })
        return dataset
    
    def _find_matching_conversations(self, persona: Persona) -> List[Conversation]:
        """Find conversations that match a persona's traits"""
        # For simplicity, return a subset of conversations
        # In full implementation, this might use semantic matching between Big-5 profile and Persona-Chat profile
        return self.data_loader.persona_chat_data[:10]
    
    def _extract_memories(self, persona_text: str) -> List[str]:
        """Extract individual memory items from persona text"""
        # Persona-Chat often separates sentences by newlines or specific tokens
        memories = []
        # Split by common delimiters used in PERSONA-CHAT
        for delimiter in ['\n', 'your persona:', 'i am']:
            if delimiter in persona_text.lower():
                # Clean split
                raw_parts = persona_text.split(delimiter)
                for part in raw_parts:
                    cleaned = part.strip()
                    if len(cleaned) > 10: # Filter out noise
                        memories.append(cleaned)
                if memories: break
        
        # Fallback: Split by sentence
        if not memories:
            memories = [m.strip() for m in persona_text.split('.') if len(m.strip()) > 5]
        
        return memories
    
    def _is_memory_relevant(self, memory: str, response: str) -> bool:
        """Check if a memory is relevant to a response (Keyword overlap)"""
        memory_words = set(memory.lower().split())
        response_words = set(response.lower().split())
        overlap = memory_words.intersection(response_words)
        # Filter out stop words (simplified)
        stop_words = {'i', 'a', 'the', 'and', 'to', 'of', 'in', 'is', 'am', 'my'}
        meaningful_overlap = overlap - stop_words
        return len(meaningful_overlap) >= 1

# =============================================================================
# Utility Functions
# =============================================================================

def get_personality_description(big5_dict: Dict[str, str], descriptions: Dict[str, str]) -> str:
    """Get combined personality description based on Big5 traits"""
    description_parts = []
    
    for dimension, level in big5_dict.items():
        key = f"{level.capitalize()} {dimension}"
        if key in descriptions:
            description_parts.append(descriptions[key])
    
    return " ".join(description_parts)

def parse_big5_string(big5_str: str) -> Dict[str, str]:
    """Parse Big5 string format: {high, low, high, high, low}"""
    dimensions = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    
    match = re.search(r'\{([^}]+)\}', big5_str)
    if match:
        values = [v.strip() for v in match.group(1).split(',')]
        if len(values) == 5:
            return dict(zip(dimensions, values))
    
    return {}

def generate_all_big5_combinations() -> List[str]:
    """Generate all 32 possible Big5 combinations"""
    combinations = []
    levels = ['high', 'low']
    
    for o in levels:
        for c in levels:
            for e in levels:
                for a in levels:
                    for n in levels:
                        combinations.append(f"{{{o}, {c}, {e}, {a}, {n}}}")
    
    return combinations

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load_all()
    
    print(f"Loaded {len(data['personas'])} personas")
    print(f"Loaded {len(data['persona_chat'])} conversations")
    
    builder = TrainingDatasetBuilder(loader)
    
    # Test memory selection dataset building
    print("Building memory selection dataset...")
    memory_dataset = builder.build_memory_selection_dataset(similarity_threshold=0.6)
    print(f"Built {len(memory_dataset)} memory selection samples")
    
    if len(memory_dataset) > 0:
        print("Sample:", memory_dataset[0])
