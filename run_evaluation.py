"""
Evaluation Runner for Humanized Conversation with Personality

This script manages the end-to-end evaluation pipeline, including:
1. Loading persona datasets and BFI question sets.
2. Orchestrating the interview process (BFI + Random Questions).
3. Integrating with Inference Engines (Llama-3, BERT) and Scoring Engines (GPT-4).
4. Generating detailed reports and comparing results against baselines.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict
import random

from tqdm import tqdm

from config import (
    BIG5_PERSONA_PATH,
    BFI_PATH,
    OUTPUT_DIR,
    evaluation_config
)
from data_processing import DataLoader, Persona
from evaluation import (
    BFIQuestionLoader,
    StructuredInterview,
    PersonalityEvaluator,
    AggregatedEvaluator,
    RandomQuestionEvaluator,
    InterviewResponse,
    EvaluationResult,
    PersonalityScore
)
from inference import InferenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from openai import OpenAI

# =============================================================================
# Evaluation Pipeline
# =============================================================================

class EvaluationPipeline:
    """
    Comprehensive pipeline for assessing AI personality alignment.
    
    Manages the flow of:
    - Interviewing the Persona (Llama-3)
    - Scoring the responses (GPT-4)
    - Aggregating metrics across trials
    """
    
    def __init__(
        self,
        personas_path: str = BIG5_PERSONA_PATH,
        bfi_path: str = BFI_PATH,
        inference_engine: Optional[InferenceEngine] = None,
        evaluator_client = None  # LLM for scoring (GPT-4)
    ):
        """
        Initialize the evaluation pipeline components.
        
        Args:
            personas_path: Path to the JSON file containing persona definitions.
            bfi_path: Path to the BFI (Big Five Inventory) questions file.
            inference_engine: The AI model wrapper used to generate responses.
            evaluator_client: The LLM client (e.g., GPT-4) used to score responses.
        """
        # Load Personas
        self.data_loader = DataLoader()
        self.personas = {}
        
        if os.path.exists(personas_path):
            personas_list = self.data_loader.load_big5_personas(personas_path)
            self.personas = {p.index: p for p in personas_list}
        
        # Load BFI Questions
        self.bfi_loader = BFIQuestionLoader(bfi_path)
        
        # Initialize Evaluators
        self.interview = StructuredInterview(
            self.bfi_loader,
            evaluator_client=evaluator_client
        )
        self.personality_evaluator = PersonalityEvaluator(self.bfi_loader)
        self.random_evaluator = RandomQuestionEvaluator(evaluator_client)
        self.aggregator = AggregatedEvaluator()
        
        # Inference Engine
        self.engine = inference_engine
    
    def evaluate_persona(
        self,
        persona: Persona,
        questions_per_dimension: int = 5,
        use_random_questions: bool = True,
        num_trials: int = 3 
    ) -> EvaluationResult:
        """
        Evaluate a single persona's personality alignment with repetitions.
        
        Protocol:
        - Runs the interview 3 times (trials).
        - Aggregates all responses to calculate stable scores.
        """
        # Get target Big-5
        target_big5 = persona.get_big5_dict()
        
        # Container for all responses across all trials
        all_responses = []

        print("\n" + "="*60)
        print(f"Starting Evaluation for Persona {persona.index}: {persona.name}")
        print(f"Target Big-5: {persona.big5}")
        print(f"Protocol: {num_trials} Independent Trials")
        print("="*60)

        # Loop for Repetitions (Trials)
        for trial in range(num_trials):
            print(f"\n>>> Running Trial {trial + 1} / {num_trials}")
            
            # 1. Conduct Structured Interview (BFI)
            # Note: _conduct_interview uses random sampling internally, so questions vary per trial
            bfi_responses = self._conduct_interview(
                persona,
                questions_per_dimension
            )
            
            # 2. Add random question evaluation if enabled
            if use_random_questions:
                random_responses = self._conduct_random_interview(persona)
                bfi_responses.extend(random_responses)
            
            # 3. Collect responses from this trial
            all_responses.extend(bfi_responses)

        # Compute Final Metric
        # The evaluator calculates the mean score of all collected responses across 3 trials.
        result = self.personality_evaluator.evaluate(all_responses, target_big5)

        print("\n" + "-"*40)
        print(f"Session Complete for {persona.name}")
        print(f"Consistency Score: {result.consistency_score:.2f}")
        print("-"*40 + "\n")
        
        return result
    
    def _conduct_interview(
        self,
        persona: Persona,
        questions_per_dimension: int
    ) -> List[InterviewResponse]:
        """
        Conduct the structured BFI interview.
        Randomly samples 'questions_per_dimension' questions for each Big-5 trait.
        """
        responses = []
        
        for dimension in self.bfi_loader.get_all_dimensions():
            all_questions = self.bfi_loader.get_questions_by_dimension(dimension)
            
            # Randomly sample questions instead of slicing
            num_to_select = min(len(all_questions), questions_per_dimension)
            selected = random.sample(all_questions, num_to_select)
            
            for question in selected:
                print(f"\nAsking the Subject BFI Q{question.id}...", end="", flush=True)
                
                # Generate Response
                if self.engine:
                    response_text, _ = self.engine.generate_response(
                        user_message=question.question,
                        persona=persona,
                        conversation_history=[],
                        memories=[]
                    )
                else:
                    # Simulation mode for testing without GPU
                    response_text = self._simulate_response(persona, question.question)

                # Create Response Object
                response_obj = InterviewResponse(
                    question_id=question.id,
                    question=question.question,
                    response=response_text,
                    dimension=dimension
                )

                # Score Response (using GPT-4 or Keyword)
                score = self.interview._score_single_response(response_obj)
                response_obj.score = score

                # Debug Output
                # print(f"\n  [Q ({dimension})]: {question.question}")
                # print(f"  [A]: {response_text}")   # Print for human annotators to evaluate
                # print(f"  [LLM Evaluator Score]: {score} / 5 ({dimension})")
                # print("-" * 20)

                
                responses.append(response_obj)
        
        return responses
    
    def _conduct_random_interview(
        self,
        persona: Persona,
        num_questions: int = 5
    ) -> List[InterviewResponse]:
        """
        Conduct the unstructured random question interview.
        Randomly samples 'num_questions' from the golden list.
        """
        responses = []
        
        # Randomly sample 5 distinct questions from the pool
        all_random_qs = self.random_evaluator.RANDOM_QUESTIONS
        selected_questions = random.sample(all_random_qs, min(num_questions, len(all_random_qs)))
        
        for i, question_text in enumerate(selected_questions):
            print(f"\nAsking the Subject Random Q{i+1}...", end="", flush=True)
            
            # Generate Response
            if self.engine:
                response_text, _ = self.engine.generate_response(
                    user_message=question_text,
                    persona=persona
                )
            else:
                response_text = self._simulate_response(persona, question_text)

            # Evaluate Consistency (Keyword/Heuristic based for now, or GPT-4 if configured)
            adjustments = self.random_evaluator.evaluate_response(response_text, question_text)

            # Debug Output
            # print(f"\n  [Random Q]: {question.question}")
            # print(f"  [A]: {response_text}")   # Print for human annotators to evaluate
            # print(f"  [LLM Evaluator Result]: {adjustments}") # Review LLM's evaluation after human rating
            # print("-" * 20)
            
            # Store Response (Dimension set to 'Random' to avoid skewing BFI scores)
            responses.append(InterviewResponse(
                question_id=1000 + i,  # Dummy ID for random questions
                question=question_text,
                response=response_text,
                dimension="Random"
            ))
        
        return responses
    
    def _simulate_response(self, persona: Persona, question: str) -> str:
        """
        Generate a dummy response based on personality traits (Simulation Mode).
        Used when no inference engine is loaded.
        """
        big5 = persona.get_big5_dict()
        response_parts = []
        
        # Simple rule-based generation for testing pipeline logic
        if big5.get('Extraversion') == 'high':
            response_parts.append("I'd love to tell you! I enjoy sharing my thoughts.")
        else:
            response_parts.append("I prefer to keep things to myself, but I'll answer.")
        
        if big5.get('Conscientiousness') == 'high':
            response_parts.append("I always plan ahead.")
        
        return " ".join(response_parts)
    
    def evaluate_all_personas(
        self,
        num_samples: Optional[int] = None,
        questions_per_dimension: int = 5,
        num_trials: int = 3
    ) -> Dict:
        """
        Execute evaluation on the full dataset or a subset.
        """
        personas_to_eval = list(self.personas.values())
        
        # Randomly sample personas if num_samples is specified
        if num_samples and num_samples < len(personas_to_eval):
            personas_to_eval = random.sample(personas_to_eval, num_samples)
        
        logger.info(f"Queued {len(personas_to_eval)} personas for evaluation.")
        logger.info(f"Each persona will be evaluated with {num_trials} trials.")
        
        for persona in tqdm(personas_to_eval, desc="Evaluating Personas"):
            try:
                result = self.evaluate_persona(
                    persona,
                    questions_per_dimension=questions_per_dimension,
                    num_trials=num_trials
                )
                self.aggregator.add_result(result)
            except Exception as e:
                logger.error(f"Failed to evaluate persona {persona.index}: {e}")
        
        # Compute aggregate metrics across all personas
        hit_distribution = self.aggregator.compute_hit_rate_distribution()
        dim_accuracy = self.aggregator.compute_dimension_accuracy()
        overall_consistency = self.aggregator.compute_overall_consistency()
        
        return {
            'num_evaluated': len(self.aggregator.results),
            'hit_rate_distribution': hit_distribution,
            'dimension_accuracy': dim_accuracy,
            'overall_consistency': overall_consistency
        }
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate and optionally save a readable evaluation report."""
        report = self.aggregator.generate_report()
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Detailed report saved to {output_path}")
        
        return report

# =============================================================================
# Main Evaluation Runner
# =============================================================================

def run_full_evaluation(args):
    """
    Entry point for running the full evaluation experiment.
    Initializes the engine, connects to GPT-4 evaluator, and runs the loop.
    """
    
    # Internal class for GPT-4 scoring (Evaluator Client)
    class OpenAIClient:
        def __init__(self, key):
            self.client = OpenAI(api_key=key)
        
        def generate(self, prompt):
            """Generate score (1-5) using GPT-4-turbo"""
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, # Deterministic for scoring
                    max_tokens=5
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Scoring API Error: {e}")
                return "3" # Fallback to Neutral

    logger.info("=" * 60)
    logger.info("Humanized Conversation - Personality Alignment Evaluation")
    logger.info("=" * 60)
    
    # Setup Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)

    # 1. Initialize Inference Engine (Llama-3 / BERT)
    engine = None
    if args.llama_path or args.bert_path:
        logger.info(f"Loading Inference Engine from {args.llama_path}...")
        engine = InferenceEngine(
            llama_path=args.llama_path,
            bert_path=args.bert_path
        )
    
    # 2. Initialize Evaluator Client (GPT-4)
    my_api_key = os.getenv("OPENAI_API_KEY", "sk-xxxxxxxx") # Replace with actual OpenAI API Key
    gpt_evaluator = None
    
    if not my_api_key or "sk-xxxx" in my_api_key:
        logger.warning("No valid OpenAI Key found. Evaluation will fallback to keyword matching.")
    else:
        logger.info("Connecting to GPT-4-turbo for automated scoring...")
        gpt_evaluator = OpenAIClient(my_api_key)

    # 3. Initialize Pipeline
    pipeline = EvaluationPipeline(
        personas_path=args.personas_path,
        bfi_path=args.bfi_path,
        inference_engine=engine, 
        evaluator_client=gpt_evaluator
    )
    
    logger.info(f"Loaded {len(pipeline.personas)} personas from dataset.")
    logger.info(f"Loaded {len(pipeline.bfi_loader.questions)} BFI questions.")
    
    # 4. Run Evaluation
    logger.info(f"Starting evaluation of {args.num_samples} samples with {args.n_trials} trials each...")
    results = pipeline.evaluate_all_personas(
        num_samples=args.num_samples,
        questions_per_dimension=args.questions_per_dim,
        num_trials=args.n_trials
    )
    
    # 5. Save and Display Results
    report_path = os.path.join(eval_dir, "evaluation_report.txt")
    report = pipeline.generate_report(report_path)
    
    results_path = os.path.join(eval_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)
    print(report)
    print(f"\nFull results saved to: {eval_dir}")
    
    return results

# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run Personality Evaluation Experiment'
    )
    
    parser.add_argument(
        '--personas-path',
        default=BIG5_PERSONA_PATH,
        help='Path to the persona JSON dataset'
    )
    parser.add_argument(
        '--bfi-path',
        default=BFI_PATH,
        help='Path to the BFI questions JSON'
    )
    parser.add_argument(
        '--output-dir',
        default=OUTPUT_DIR,
        help='Directory to save evaluation artifacts'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of random personas to evaluate'
    )
    parser.add_argument(
        '--questions-per-dim',
        type=int,
        default=5,
        help='Number of BFI questions per dimension (Default: 5)'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=3,
        help='Number of independent trials per persona (Default: 3, per paper Section 9.1.2)'
    )
    parser.add_argument(
        '--llama-path',
        help='Path to the fine-tuned Llama-3 model'
    )
    parser.add_argument(
        '--bert-path',
        help='Path to the fine-tuned BERT model (if used)'
    )
    
    args = parser.parse_args()
    
    run_full_evaluation(args)

if __name__ == '__main__':
    main()
