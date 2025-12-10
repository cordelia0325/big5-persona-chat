"""
Personality Model Module for Humanized Conversation with Personality

This module integrates all components into a unified personality-aware 
conversation system.

The core objective function:
P(y_k | C, P, M) ≈ P_θ(y_k | C, P, m_k) · P_η(m_k | C, P, M)

Where:
- C = dialogue context
- P = persona based on Big Five
- M = memory set
- m_k = selected memory
- y_k = generated response
- P_θ = response generation model (Llama3-8B-Instruct)
- P_η = memory selection model (BERT)
"""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging
import json

from config import (
    big5_config, 
    llama_config, 
    bert_config,
    SYSTEM_PROMPT_TEMPLATE
)
from data_processing import Persona, DataLoader
from memory_selector import MemorySelector, SimpleMemorySelector
from response_generator import ResponseGenerator, PersonaResponseGenerator
from persona_generator import PersonaGenerator, BehaviorPresetGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConversationContext:
    """Represents the current conversation state"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    persona: Optional[Persona] = None
    memories: List[str] = field(default_factory=list)
    selected_memory: Optional[str] = None
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation"""
        self.messages.append({"role": role, "content": content})
    
    def get_context_text(self) -> str:
        """Get conversation as plain text"""
        return " ".join([m["content"] for m in self.messages])
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
        self.selected_memory = None

@dataclass
class PersonalityProfile:
    """Complete personality profile for an AI agent"""
    big5: Dict[str, str]  # {"Openness": "high", ...}
    persona: Optional[Persona] = None
    behavior_presets: List[Dict] = field(default_factory=list)
    system_prompt: str = ""
    
    @classmethod
    def from_big5_string(cls, big5_str: str) -> 'PersonalityProfile':
        """Create profile from Big5 string like {high, low, high, high, low}"""
        import re
        
        dimensions = ['Openness', 'Conscientiousness', 'Extraversion', 
                      'Agreeableness', 'Neuroticism']
        
        match = re.search(r'\{([^}]+)\}', big5_str)
        if match:
            values = [v.strip() for v in match.group(1).split(',')]
            if len(values) == 5:
                big5 = dict(zip(dimensions, values))
                return cls(big5=big5)
        
        # Default to all high
        return cls(big5={d: "high" for d in dimensions})

# =============================================================================
# Personality-Aware Conversation Model
# =============================================================================

class PersonalityConversationModel:
    """
    Main class integrating persona, memory, and response generation
    

    This class implements:
    1. Sense: Process user input and context
    2. Plan: Select relevant memories and plan response
    3. Act: Generate personality-consistent response
    """
    
    def __init__(
        self,
        response_generator: ResponseGenerator = None,
        memory_selector: MemorySelector = None,
        use_simple_memory: bool = True
    ):
        """
        Initialize the conversation model
        
        Args:
            response_generator: Llama3-based response generator
            memory_selector: BERT-based memory selector
            use_simple_memory: Use simple cosine similarity for memory selection
        """
        self.response_generator = response_generator
        
        if memory_selector:
            self.memory_selector = memory_selector
        elif use_simple_memory:
            self.memory_selector = SimpleMemorySelector()
        else:
            self.memory_selector = None
        
        self.behavior_preset_gen = BehaviorPresetGenerator()
        self.conversations: Dict[str, ConversationContext] = {}
    
    def create_conversation(
        self,
        conversation_id: str,
        persona: Persona,
        initial_memories: List[str] = None
    ) -> ConversationContext:
        """Create a new conversation with a persona"""
        
        context = ConversationContext(
            persona=persona,
            memories=initial_memories or []
        )
        
        self.conversations[conversation_id] = context
        
        return context
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get existing conversation context"""
        return self.conversations.get(conversation_id)
    
    def process_message(
        self,
        conversation_id: str,
        user_message: str
    ) -> str:
        """
        Process a user message and generate response
        
        Implements the objective function:
        P(y_k | C, P, M) ≈ P_θ(y_k | C, P, m_k) · P_η(m_k | C, P, M)
        """
        context = self.conversations.get(conversation_id)
        
        if not context:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Add user message to context
        context.add_message("user", user_message)
        
        # Step 1: Select relevant memory (P_η)
        selected_memory = self._select_memory(context)
        context.selected_memory = selected_memory
        
        # Step 2: Generate response (P_θ)
        response = self._generate_response(context)
        
        # Add response to context
        context.add_message("assistant", response)
        
        return response
    
    def _select_memory(self, context: ConversationContext) -> Optional[str]:
        """
        Select the most relevant memory for the current context
        
        P_η(m_k | C, P, M)
        """
        if not context.memories or not self.memory_selector:
            return None
        
        context_text = context.get_context_text()
        persona_text = context.persona.personality if context.persona else ""
        
        # Use memory selector
        if hasattr(self.memory_selector, 'select_memory'):
            if hasattr(self.memory_selector, 'encoder'):
                # SimpleMemorySelector
                selected = self.memory_selector.select_memory(
                    context_text,
                    context.memories,
                    top_k=1
                )
            else:
                # BertMemorySelector
                selected = self.memory_selector.select_memory(
                    context_text,
                    persona_text,
                    context.memories,
                    top_k=1
                )
            
            if selected:
                return selected[0][0]
        
        return None
    
    def _generate_response(self, context: ConversationContext) -> str:
        """
        Generate a personality-consistent response
        
        P_θ(y_k | C, P, m_k)
        """
        if not self.response_generator:
            return self._fallback_response(context)
        
        # Build system prompt from persona
        system_prompt = ""
        if context.persona:
            system_prompt = context.persona.to_system_prompt()
        
        # Generate response
        response = self.response_generator.generate_response(
            context=context.messages,
            persona=system_prompt,
            selected_memory=context.selected_memory
        )
        
        return response
    
    def _fallback_response(self, context: ConversationContext) -> str:
        """Generate a simple fallback response"""
        
        if context.persona:
            return f"[{context.persona.name}]: I appreciate your message. " \
                   f"As someone who is {context.persona.personality[:100]}..."
        
        return "I appreciate your message. How can I help you today?"
    
    def add_memory(self, conversation_id: str, memory: str):
        """Add a new memory to the conversation"""
        context = self.conversations.get(conversation_id)
        if context:
            context.memories.append(memory)
    
    def get_personality_profile(
        self, 
        conversation_id: str
    ) -> Optional[PersonalityProfile]:
        """Get the personality profile for a conversation"""
        context = self.conversations.get(conversation_id)
        
        if not context or not context.persona:
            return None
        
        big5_dict = context.persona.get_big5_dict()
        behavior_presets = self.behavior_preset_gen.generate_behavior_presets(big5_dict)
        
        return PersonalityProfile(
            big5=big5_dict,
            persona=context.persona,
            behavior_presets=behavior_presets,
            system_prompt=context.persona.to_system_prompt()
        )

# =============================================================================
# Role-Playing Agent
# =============================================================================

class RolePlayingAgent:
    """Role-playing agent with personality-based behavior"""
    
    def __init__(
        self,
        persona: Persona,
        conversation_model: PersonalityConversationModel
    ):
        self.persona = persona
        self.model = conversation_model
        self.conversation_id = f"agent_{persona.index}"
        
        # Create conversation context
        self.context = self.model.create_conversation(
            self.conversation_id,
            persona,
            initial_memories=self._extract_memories()
        )
        
        # Generate behavior presets
        self.behavior_presets = BehaviorPresetGenerator().generate_behavior_presets(
            persona.get_big5_dict()
        )
    
    def _extract_memories(self) -> List[str]:
        """Extract memories from persona description"""
        memories = [
            self.persona.hobby,
            self.persona.growth_experience,
            self.persona.family_relationship,
            self.persona.working_conditions,
            self.persona.social_relationship,
            self.persona.recent_worry_or_anxiety
        ]
        return [m for m in memories if m]
    
    def chat(self, user_message: str) -> str:
        """Process a chat message and return response"""
        return self.model.process_message(self.conversation_id, user_message)
    
    def get_system_prompt(self) -> str:
        """Get the full system prompt with behavior presets"""
        base_prompt = self.persona.to_system_prompt()
        
        # Add behavior presets
        if self.behavior_presets:
            preset_gen = BehaviorPresetGenerator()
            preset_text = "\n\n## Behavior Guidelines\n"
            
            for preset in self.behavior_presets:
                preset_text += preset_gen.format_preset_prompt(preset) + "\n"
            
            base_prompt += preset_text
        
        return base_prompt
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.context.clear()

# =============================================================================
# Personality Model Factory
# =============================================================================

class PersonalityModelFactory:
    """Factory for creating personality models and agents"""
    
    @staticmethod
    def create_model(
        use_gpu: bool = True,
        load_llama: bool = False,
        load_bert: bool = False,
        llama_path: str = None,
        bert_path: str = None
    ) -> PersonalityConversationModel:
        """
        Create a personality conversation model
        
        Args:
            use_gpu: Whether to use GPU if available
            load_llama: Whether to load the Llama model
            load_bert: Whether to load the BERT memory selector
            llama_path: Path to fine-tuned Llama model
            bert_path: Path to fine-tuned BERT model
        """
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # Initialize response generator
        response_generator = None
        if load_llama:
            response_generator = ResponseGenerator(
                model_path=llama_path,
                device=device
            )
        
        # Initialize memory selector
        memory_selector = None
        if load_bert:
            memory_selector = MemorySelector(
                model_path=bert_path,
                device=device
            )
        
        return PersonalityConversationModel(
            response_generator=response_generator,
            memory_selector=memory_selector,
            use_simple_memory=not load_bert
        )
    
    @staticmethod
    def create_agent(
        persona: Persona,
        model: PersonalityConversationModel = None
    ) -> RolePlayingAgent:
        """Create a role-playing agent with a persona"""
        
        if model is None:
            model = PersonalityModelFactory.create_model(
                load_llama=False,
                load_bert=False
            )
        
        return RolePlayingAgent(persona, model)
    
    @staticmethod
    def load_personas(filepath: str) -> List[Persona]:
        """Load personas from JSON file"""
        loader = DataLoader()
        return loader.load_big5_personas(filepath)

# =============================================================================
# Dialogue Manager
# =============================================================================

class DialogueManager:
    """
    Manage multi-turn dialogues with personality consistency
    
    Tracks:
    - Conversation history
    - Personality state
    - Memory utilization
    """
    
    def __init__(self, model: PersonalityConversationModel):
        self.model = model
        self.active_conversations: Dict[str, Dict] = {}
    
    def start_conversation(
        self,
        user_id: str,
        persona: Persona
    ) -> str:
        """Start a new conversation"""
        
        conversation_id = f"{user_id}_{persona.index}"
        
        context = self.model.create_conversation(
            conversation_id,
            persona,
            initial_memories=[]
        )
        
        self.active_conversations[conversation_id] = {
            'user_id': user_id,
            'persona': persona,
            'context': context,
            'turn_count': 0
        }
        
        # Generate greeting
        greeting = self._generate_greeting(persona)
        context.add_message("assistant", greeting)
        
        return greeting
    
    def _generate_greeting(self, persona: Persona) -> str:
        """Generate a personality-appropriate greeting"""
        
        big5 = persona.get_big5_dict()
        
        if big5.get('Extraversion') == 'high':
            return f"Hey there! I'm {persona.name}. Great to meet you! " \
                   f"I work as a {persona.job}. What's on your mind today?"
        else:
            return f"Hello. I'm {persona.name}, a {persona.job}. " \
                   f"How can I help you?"
    
    def send_message(
        self,
        conversation_id: str,
        message: str
    ) -> str:
        """Send a message in an existing conversation"""
        
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conv_data = self.active_conversations[conversation_id]
        conv_data['turn_count'] += 1
        
        response = self.model.process_message(conversation_id, message)
        
        return response
    
    def end_conversation(self, conversation_id: str):
        """End a conversation"""
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
        
        if conversation_id in self.model.conversations:
            del self.model.conversations[conversation_id]
    
    def get_conversation_stats(self, conversation_id: str) -> Dict:
        """Get statistics for a conversation"""
        
        if conversation_id not in self.active_conversations:
            return {}
        
        conv_data = self.active_conversations[conversation_id]
        context = conv_data['context']
        
        return {
            'user_id': conv_data['user_id'],
            'persona_name': conv_data['persona'].name,
            'turn_count': conv_data['turn_count'],
            'message_count': len(context.messages),
            'memory_count': len(context.memories),
            'last_memory_used': context.selected_memory
        }

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("Personality Model Demo")
    print("=" * 50)
    
    # Create a sample persona
    sample_persona_data =   {
      "index": 400,
      "big-5": "{low, low, low, high, high}",
      "profile": {
        "name": "Cassia Marigold Fletcher",
        "gender": "Female",
        "age": "32",
        "region": "Austin, Texas, USA",
        "tone": "Cassia's voice is melodious and expressive, often varying in pitch to convey her emotions vividly. Her speaking style is descriptive and imaginative, filled with metaphors and analogies that reflect her artistic nature.",
        "job": "Mixed Media Artist",
        "personality": "Cassia is someone who thrives on the tangible and practical aspects of life, often shying away from abstract or theoretical ideas. She shows a notable lack of organization in her personal and professional life, which she compensates for with her creativity and adaptability. Preferring solitude or the company of a few close friends, Cassia is introspective yet displays warmth and friendliness in her interactions. Despite her generally upbeat nature, she is susceptible to anxiety and stress, often feeling overwhelmed by her emotions.",
        "advantages_and_disadvantages": "Strengths include her adaptability, creativity, and ability to connect deeply with people on a personal level. Weaknesses involve her disorganization, tendency to procrastinate, and susceptibility to stress and anxiety.",
        "hobby": "Cassia is passionate about creating art that integrates materials like fabrics, paints, and reclaimed objects. She spends much of her free time in her studio, experimenting with different textures and colors. Additionally, she enjoys visiting local art galleries and antique shops to find inspiration and unique items that she can incorporate into her artwork.",
        "growth_experience": "Growing up in a small, culturally rich town, Cassia was always fascinated by the local artisans and their crafts. Her education in fine arts further fueled her passion for exploring different art forms and materials, shaping her future as a mixed media artist.",
        "family_relationship": "Cassia has a close-knit relationship with her parents, who have always supported her artistic pursuits. Although she moved away from home to pursue her career, she maintains regular contact with her family. She has a younger brother with whom she shares a competitive yet affectionate bond.",
        "working_conditions": "Cassia's studio is a vibrant, albeit slightly chaotic space filled with her creations and ongoing projects. She often works late into the night, finding it challenging to maintain a conventional work-life balance due to her spontaneous bursts of creativity.",
        "social_relationship": "She has a small group of close friends, mostly fellow artists and creative individuals, who understand and share her passion for art. They often collaborate on projects and participate in group exhibitions.",
        "emotional_state": "While Cassia generally maintains a positive outlook, her high level of neuroticism makes her prone to intense emotions. She experiences frequent mood swings, from bursts of euphoric creativity to periods of deep worry and doubt.",
        "living_conditions": "Cassia lives in a modest, bohemian-style apartment in a lively neighborhood known for its artistic community. Her home is filled with artworks, plants, and eclectic furniture, reflecting her artistic taste and personality.",
        "recent_worry_or_anxiety": "Currently, Cassia is anxious about an upcoming solo exhibition, fearing that her work may not be well-received by critics and the public. This has led to sleepless nights and a restless mind as she continues to tweak her pieces.",
        "additional_information": "Cassia is actively involved in community art programs, often volunteering to teach art classes to children and teenagers. She has a peculiar quirk of collecting vintage hats, each with its own story and place in her art studio."
      }
    }
    
    # Create persona
    from data_processing import Persona
    persona = Persona.from_dict(sample_persona_data)
    
    print(f"\nPersona: {persona.name}")
    print(f"Big Five: {persona.big5}")
    print(f"Job: {persona.job}")
    
    # Create model (without loading actual LLM)
    model = PersonalityModelFactory.create_model(
        load_llama=False,
        load_bert=False
    )
    
    # Create agent
    agent = PersonalityModelFactory.create_agent(persona, model)
    
    print(f"\nAgent created for: {agent.persona.name}")
    print(f"Memories extracted: {len(agent._extract_memories())}")
    
    # Show system prompt preview
    system_prompt = agent.get_system_prompt()
    print(f"\nSystem prompt preview:\n{system_prompt[:500]}...")
