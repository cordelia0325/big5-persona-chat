"""
Inference Module for Humanized Conversation with Personality

This module provides inference capabilities for the trained models,
including interactive chat and batch inference.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch

from config import (
    BIG5_PERSONA_PATH,
    llama_config,
    bert_config
)
from data_processing import DataLoader, Persona
from memory_selector import MemorySelector, SimpleMemorySelector
from response_generator import ResponseGenerator
from personality_model import (
    PersonalityConversationModel,
    PersonalityModelFactory,
    RolePlayingAgent,
    ConversationContext
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Inference Engine
# =============================================================================

class InferenceEngine:
    """
    Main inference engine for personality-aware response generation
    
    Implements the core objective function:
    P(y_k | C, P, M) ≈ P_θ(y_k | C, P, m_k) · P_η(m_k | C, P, M)
    """
    
    def __init__(
        self,
        llama_path: Optional[str] = None,
        bert_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the inference engine
        
        Args:
            llama_path: Path to fine-tuned Llama model
            bert_path: Path to fine-tuned BERT model
            use_gpu: Whether to use GPU if available
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize response generator
        self.response_generator = None
        if llama_path or True:  # Always try to load
            try:
                self.response_generator = ResponseGenerator(
                    model_path=llama_path,
                    use_quantization=True,
                    device=self.device
                )
                logger.info("Response generator loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load response generator: {e}")
        
        # Initialize memory selector
        self.memory_selector = None
        if bert_path:
            try:
                self.memory_selector = MemorySelector(
                    model_path=bert_path,
                    device=self.device
                )
                logger.info("Memory selector loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load memory selector: {e}")
        else:
            self.memory_selector = SimpleMemorySelector()
            logger.info("Using simple memory selector")
    
    def generate_response(
        self,
        user_message: str,
        persona: Persona,
        conversation_history: List[Dict[str, str]] = None,
        memories: List[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 512
    ) -> Tuple[str, Optional[str]]:
        """
        Generate a personality-consistent response
        
        Args:
            user_message: Current user message
            persona: Persona to embody
            conversation_history: Previous conversation turns
            memories: Available memories
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (response, selected_memory)
        """
        conversation_history = conversation_history or []
        memories = memories or []
        
        # Build context
        context = conversation_history + [{"role": "user", "content": user_message}]
        context_text = " ".join([m["content"] for m in context])
        
        # Select relevant memory
        selected_memory = None
        if memories and self.memory_selector:
            if hasattr(self.memory_selector, 'select_memory'):
                selected = self.memory_selector.select_memory(
                    context_text,
                    memories,
                    top_k=1
                )
                if selected:
                    selected_memory = selected[0][0]
        
        # Generate response
        if self.response_generator:
            system_prompt = persona.to_system_prompt()
            response = self.response_generator.generate_response(
                context=context,
                persona=system_prompt,
                selected_memory=selected_memory,
                temperature=temperature,
                max_new_tokens=max_new_tokens
            )
        else:
            # Fallback response
            response = self._generate_fallback_response(persona, user_message)
        
        return response, selected_memory
    
    def _generate_fallback_response(self, persona: Persona, user_message: str) -> str:
        """Generate a simple fallback response without the LLM"""
        
        big5 = persona.get_big5_dict()
        
        # Personality-based response style
        if big5.get('Extraversion') == 'high':
            prefix = "That's a great question! "
        else:
            prefix = "I see. "
        
        if big5.get('Openness') == 'high':
            suffix = " There are so many interesting angles to consider here."
        else:
            suffix = " Let me give you a straightforward answer."
        
        return f"{prefix}As {persona.name}, a {persona.job}, I think about this quite often.{suffix}"
    
    def batch_generate(
        self,
        messages: List[str],
        persona: Persona,
        **kwargs
    ) -> List[Tuple[str, Optional[str]]]:
        """Generate responses for multiple messages"""
        
        results = []
        for msg in messages:
            response, memory = self.generate_response(msg, persona, **kwargs)
            results.append((response, memory))
        
        return results

# =============================================================================
# Interactive Chat
# =============================================================================

class InteractiveChat:
    """
    Interactive command-line chat interface
    
    Allows users to have conversations with AI personas
    """
    
    def __init__(
        self,
        personas_path: str = BIG5_PERSONA_PATH,
        load_llama: bool = False,
        load_bert: bool = False,
        llama_path: Optional[str] = None,
        bert_path: Optional[str] = None
    ):
        """Initialize the interactive chat"""
        
        # Load personas
        self.data_loader = DataLoader()
        self.personas = {}
        
        if os.path.exists(personas_path):
            personas_list = self.data_loader.load_big5_personas(personas_path)
            self.personas = {p.index: p for p in personas_list}
            logger.info(f"Loaded {len(self.personas)} personas")
        
        # Initialize inference engine
        self.engine = None
        if load_llama or load_bert:
            self.engine = InferenceEngine(
                llama_path=llama_path,
                bert_path=bert_path
            )
        
        # Current state
        self.current_persona: Optional[Persona] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.memories: List[str] = []
    
    def select_persona(self, persona_id: int) -> bool:
        """Select a persona by ID"""
        
        if persona_id in self.personas:
            self.current_persona = self.personas[persona_id]
            self.conversation_history = []
            self.memories = self._extract_memories(self.current_persona)
            return True
        return False
    
    def _extract_memories(self, persona: Persona) -> List[str]:
        """Extract memories from persona"""
        memories = [
            persona.hobby,
            persona.growth_experience,
            persona.family_relationship,
            persona.working_conditions,
            persona.social_relationship,
            persona.recent_worry_or_anxiety
        ]
        return [m for m in memories if m]
    
    def chat(self, user_message: str) -> str:
        """Process a chat message and return response"""
        
        if not self.current_persona:
            return "No persona selected. Use 'select <id>' to choose a persona."
        
        if self.engine:
            response, memory = self.engine.generate_response(
                user_message=user_message,
                persona=self.current_persona,
                conversation_history=self.conversation_history,
                memories=self.memories
            )
        else:
            response = self._simple_response(user_message)
            memory = None
        
        # Update history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _simple_response(self, user_message: str) -> str:
        """Generate simple response without LLM"""
        
        if not self.current_persona:
            return "Hello!"
        
        p = self.current_persona
        return f"[{p.name}]: Thank you for your message. As a {p.job}, I find this topic quite interesting."
    
    def run(self):
        """Run the interactive chat loop"""
        
        self._print_welcome()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == 'quit' or user_input.lower() == 'exit':
                    print("\nGoodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self._print_help()
                    continue
                
                elif user_input.lower() == 'list':
                    self._list_personas()
                    continue
                
                elif user_input.lower().startswith('select '):
                    try:
                        persona_id = int(user_input.split()[1])
                        if self.select_persona(persona_id):
                            p = self.current_persona
                            print(f"\n✓ Selected: {p.name} ({p.job})")
                            print(f"  Big-5: {p.big5}")
                            print(f"  {p.personality[:100]}...")
                        else:
                            print(f"✗ Persona {persona_id} not found")
                    except (ValueError, IndexError):
                        print("Usage: select <persona_id>")
                    continue
                
                elif user_input.lower() == 'persona':
                    if self.current_persona:
                        self._show_current_persona()
                    else:
                        print("No persona selected")
                    continue
                
                elif user_input.lower() == 'reset':
                    self.conversation_history = []
                    print("✓ Conversation reset")
                    continue
                
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                # Regular chat
                response = self.chat(user_input)
                
                if self.current_persona:
                    print(f"\n{self.current_persona.name}: {response}")
                else:
                    print(f"\nAssistant: {response}")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    def _print_welcome(self):
        """Print welcome message"""
        print("\n" + "=" * 60)
        print("Humanized Conversation with Personality")
        print("Interactive Chat Interface")
        print("=" * 60)
        print("\nType 'help' for available commands")
        print(f"Loaded {len(self.personas)} personas")
        
        if self.engine:
            print("✓ AI models loaded")
        else:
            print("! Running in simple mode (no AI models)")
    
    def _print_help(self):
        """Print help message"""
        print("\n" + "-" * 40)
        print("Available Commands:")
        print("-" * 40)
        print("  list          - List all available personas")
        print("  select <id>   - Select a persona by ID")
        print("  persona       - Show current persona details")
        print("  reset         - Reset conversation history")
        print("  history       - Show conversation history")
        print("  help          - Show this help message")
        print("  quit/exit     - Exit the chat")
        print("-" * 40)
    
    def _list_personas(self):
        """List available personas"""
        print("\n" + "-" * 60)
        print("Available Personas:")
        print("-" * 60)
        
        for i, (idx, p) in enumerate(sorted(self.personas.items())[:20]):
            print(f"  [{idx:3d}] {p.name:20s} | {p.job:25s} | {p.big5}")
        
        if len(self.personas) > 20:
            print(f"  ... and {len(self.personas) - 20} more")
        print("-" * 60)
    
    def _show_current_persona(self):
        """Show current persona details"""
        p = self.current_persona
        print("\n" + "-" * 60)
        print(f"Current Persona: {p.name}")
        print("-" * 60)
        print(f"  Index: {p.index}")
        print(f"  Big-5: {p.big5}")
        print(f"  Age: {p.age}")
        print(f"  Region: {p.region}")
        print(f"  Job: {p.job}")
        print(f"  Tone: {p.tone}")
        print(f"\nPersonality:")
        print(f"  {p.personality}")
        print(f"\nHobbies: {p.hobby}")
        print("-" * 60)
    
    def _show_history(self):
        """Show conversation history"""
        print("\n" + "-" * 40)
        print("Conversation History:")
        print("-" * 40)
        
        if not self.conversation_history:
            print("  (empty)")
        else:
            for msg in self.conversation_history:
                role = msg['role'].capitalize()
                content = msg['content'][:80]
                if len(msg['content']) > 80:
                    content += "..."
                print(f"  {role}: {content}")
        print("-" * 40)

# =============================================================================
# Batch Inference
# =============================================================================

class BatchInference:
    """
    Batch inference for processing multiple conversations
    
    Useful for evaluation and large-scale testing
    """
    
    def __init__(
        self,
        engine: InferenceEngine,
        personas: Dict[int, Persona]
    ):
        self.engine = engine
        self.personas = personas
    
    def process_conversations(
        self,
        conversations: List[Dict],
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Process multiple conversations
        
        Args:
            conversations: List of conversation dicts with:
                - persona_id: ID of persona to use
                - messages: List of user messages
            output_path: Optional path to save results
            
        Returns:
            List of results with responses
        """
        results = []
        
        for conv in conversations:
            persona_id = conv.get('persona_id', 0)
            messages = conv.get('messages', [])
            
            if persona_id not in self.personas:
                logger.warning(f"Persona {persona_id} not found, skipping")
                continue
            
            persona = self.personas[persona_id]
            responses = []
            history = []
            
            for msg in messages:
                response, memory = self.engine.generate_response(
                    user_message=msg,
                    persona=persona,
                    conversation_history=history
                )
                
                history.append({"role": "user", "content": msg})
                history.append({"role": "assistant", "content": response})
                
                responses.append({
                    'user': msg,
                    'assistant': response,
                    'memory_used': memory
                })
            
            results.append({
                'persona_id': persona_id,
                'persona_name': persona.name,
                'big5': persona.big5,
                'conversation': responses
            })
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_path}")
        
        return results

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inference Module')
    parser.add_argument('--mode', choices=['chat', 'batch'], default='chat')
    parser.add_argument('--personas-path', default=BIG5_PERSONA_PATH)
    parser.add_argument('--persona-id', type=int, default=0)
    parser.add_argument('--load-llama', action='store_true')
    parser.add_argument('--load-bert', action='store_true')
    parser.add_argument('--llama-path', help='Path to fine-tuned Llama model')
    parser.add_argument('--bert-path', help='Path to fine-tuned BERT model')
    
    args = parser.parse_args()
    
    if args.mode == 'chat':
        chat = InteractiveChat(
            personas_path=args.personas_path,
            load_llama=args.load_llama,
            load_bert=args.load_bert,
            llama_path=args.llama_path,
            bert_path=args.bert_path
        )
        chat.run()
    else:
        print("Batch mode requires input file specification")
