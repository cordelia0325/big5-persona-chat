#!/usr/bin/env python
"""
Main Entry Point for Humanized Conversation with Personality

Usage:
    python main.py                      # Run full demo suite
    python main.py --demo basic         # Run basic data loading demo
    python main.py --demo chat          # Run interactive chat demo
    python main.py --demo api           # Start API server
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# =============================================================================
# Demo 1: Basic Data Loading
# =============================================================================

def demo_data_loading():
    """Demonstrate how to load persona data"""
    print("\n" + "=" * 60)
    print("Demo 1: Data Loading")
    print("=" * 60)
    
    from data_processing import DataLoader
    loader = DataLoader()
    
    personas_path = os.path.join(DATA_DIR, "big5-persona.json")
    if os.path.exists(personas_path):
        personas = loader.load_big5_personas(personas_path)
        print(f"\nLoaded {len(personas)} personas")
        if personas:
            p = personas[0]
            print(f"\nFirst Persona: {p.name}, {p.age} years old, {p.job}")
            print(f"Big-5: {p.big5}")
    else:
        print(f"Please place data files in {DATA_DIR}")
    
    return loader

# =============================================================================
# Demo 2: Create Agent
# =============================================================================

def demo_create_agent():
    """Demonstrate creating a Personality Agent"""
    print("\n" + "=" * 60)
    print("Demo 2: Create Personality Agent")
    print("=" * 60)
    
    from data_processing import Persona
    from personality_model import PersonalityModelFactory
    
    sample_data =   {
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
    
    persona = Persona.from_dict(sample_data)
    print(f"\nCreated Persona: {persona.name}")
    print(f"Big-5: {persona.big5}")
    
    model = PersonalityModelFactory.create_model(load_llama=False, load_bert=False)
    agent = PersonalityModelFactory.create_agent(persona, model)
    print(f"Agent created successfully")
    
    return agent

# =============================================================================
# Demo 3: Memory Selection
# =============================================================================

def demo_memory_selection():
    """Demonstrate memory retrieval/selection"""
    print("\n" + "=" * 60)
    print("Demo 3: Memory Selection")
    print("=" * 60)
    
    from memory_selector import SimpleMemorySelector
    
    selector = SimpleMemorySelector()
    context = "I love hiking and outdoor activities"
    memories = [
        "I enjoy camping in the mountains",
        "I work as a software engineer",
        "I have a pet dog named Max",
        "Weekend runs in the park",
        "I love landscape photography"
    ]
    
    print(f"\nContext: {context}")
    print(f"\nSelecting most relevant memories:")
    
    selected = selector.select_memory(context, memories, top_k=2)
    for memory, score in selected:
        print(f"  [{score:.3f}] {memory}")
    
    return selector

# =============================================================================
# Demo 4: Evaluation
# =============================================================================

def demo_evaluation():
    """Demonstrate personality evaluation metrics"""
    print("\n" + "=" * 60)
    print("Demo 4: Personality Evaluation")
    print("=" * 60)
    
    from evaluation import BFIQuestionLoader, PersonalityEvaluator, PersonalityScore
    
    bfi_path = os.path.join(DATA_DIR, "BFI.json")
    if not os.path.exists(bfi_path):
        print(f"BFI.json not found")
        return None
    
    bfi_loader = BFIQuestionLoader(bfi_path)
    evaluator = PersonalityEvaluator(bfi_loader)
    
    target = {"Openness": "high", "Conscientiousness": "high", 
              "Extraversion": "high", "Agreeableness": "high", "Neuroticism": "low"}
    
    predicted = PersonalityScore(openness=4.2, conscientiousness=3.8,
                                  extraversion=4.5, agreeableness=3.2, neuroticism=2.1)
    
    accuracy = evaluator.compute_accuracy(predicted, target)
    print(f"\nDimension Accuracy:")
    for dim, acc in accuracy.items():
        print(f"  {'✓' if acc else '✗'} {dim}: {acc*100:.0f}%")
    
    return evaluator

# =============================================================================
# Demo 5: API Server
# =============================================================================

def demo_api_server():
    """Start the API server"""
    print("\n" + "=" * 60)
    print("Starting API Server")
    print("=" * 60)
    
    from api import app, state
    
    personas_path = os.path.join(DATA_DIR, "big5-1024-persona.json")
    state.initialize(
        load_llama=False,
        load_bert=False,
        personas_path=personas_path if os.path.exists(personas_path) else None
    )
    
    print(f"\nLoaded {len(state.personas)} personas")
    print(f"\nServer running at: http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000)

# =============================================================================
# Quick Start Helper
# =============================================================================

class QuickStart:
    """Helper class for quick initialization"""
    
    def __init__(self):
        self.data_loader = None
        self.model = None
        self.agent = None
    
    def load_data(self):
        from data_processing import DataLoader
        self.data_loader = DataLoader()
        personas_path = os.path.join(DATA_DIR, "big5-1024-persona.json")
        if os.path.exists(personas_path):
            self.data_loader.load_big5_personas(personas_path)
        return self
    
    def create_model(self, load_llama=False, load_bert=False):
        from personality_model import PersonalityModelFactory
        self.model = PersonalityModelFactory.create_model(load_llama=load_llama, load_bert=load_bert)
        return self
    
    def create_agent(self, persona_index=0):
        from personality_model import PersonalityModelFactory
        if not self.data_loader or not self.data_loader.personas:
            raise ValueError("Call load_data() first")
        if not self.model:
            self.create_model()
        persona = self.data_loader.personas[persona_index]
        self.agent = PersonalityModelFactory.create_agent(persona, self.model)
        return self
    
    def chat(self, message):
        if not self.agent:
            raise ValueError("Call create_agent() first")
        return self.agent.chat(message)
    
    def run_demo(self):
        print("\nQuick Demo")
        print("=" * 40)
        self.load_data()
        self.create_model()
        if self.data_loader.personas:
            self.create_agent(0)
            print(f"Agent: {self.agent.persona.name}")
            response = self.chat("Hello!")
            print(f"Response: {response}")

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Humanized Conversation with Personality')
    parser.add_argument('--demo', choices=['all', 'basic', 'agent', 'memory', 'eval', 'api'],
                        default='all', help='Select demo to run')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Humanized Conversation with Personality")
    print("=" * 60)
    
    if args.demo == 'all':
        demo_data_loading()
        demo_create_agent()
        demo_memory_selection()
        demo_evaluation()
        print("\n✓ Demos complete! Run --demo api to start the server.")
    elif args.demo == 'basic':
        demo_data_loading()
    elif args.demo == 'agent':
        demo_create_agent()
    elif args.demo == 'memory':
        demo_memory_selection()
    elif args.demo == 'eval':
        demo_evaluation()
    elif args.demo == 'api':
        demo_api_server()

if __name__ == '__main__':
    main()
