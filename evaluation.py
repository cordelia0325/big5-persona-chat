"""
Evaluation Module for Humanized Conversation with Personality

This module implements a structured interview methodology for evaluating
AI personality alignment. It orchestrates the administration of the Big Five 
Inventory (BFI-44) and open-ended behavioral questions, handles response 
scoring via LLM-as-a-Judge, and computes psychometric metrics including 
accuracy and consistency scores.
"""

import json
import random
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from config import evaluation_config, BFI_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class InterviewQuestion:
    """Represents a single item from the personality inventory."""
    id: int
    original_statement: str
    question: str
    dimension: str
    category: str  # "positive" or "negative" keying
    is_reverse: bool = False

@dataclass
class InterviewResponse:
    """Captures the AI's generated response to an interview question."""
    question_id: int
    question: str
    response: str
    score: Optional[int] = None  # Likert scale (1-5)
    dimension: str = ""

@dataclass
class PersonalityScore:
    """Aggregated personality scores across the Big Five dimensions."""
    openness: float = 0.0
    conscientiousness: float = 0.0
    extraversion: float = 0.0
    agreeableness: float = 0.0
    neuroticism: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert scores to a dictionary format."""
        return {
            'Openness': self.openness,
            'Conscientiousness': self.conscientiousness,
            'Extraversion': self.extraversion,
            'Agreeableness': self.agreeableness,
            'Neuroticism': self.neuroticism
        }
    
    def get_level(self, dimension: str) -> str:
        """
        Determine the qualitative level (high/low) for a dimension.
        Threshold is set at 3.0 (neutral midpoint of 1-5 scale).
        """
        score = getattr(self, dimension.lower(), 3.0)
        return "high" if score >= 3.0 else "low"

@dataclass
class EvaluationResult:
    """Encapsulates the complete results of an evaluation session."""
    target_big5: Dict[str, str]  # Ground truth personality
    predicted_scores: PersonalityScore
    predicted_big5: Dict[str, str]  # Predicted levels (high/low)
    consistency_score: float
    dimension_accuracy: Dict[str, float]
    hit_at_k: Dict[int, bool]
    responses: List[InterviewResponse] = field(default_factory=list)

# =============================================================================
# BFI Question Loader
# =============================================================================

class BFIQuestionLoader:
    """
    Handles loading and parsing of the Big Five Inventory (BFI) dataset.
    Responsible for identifying reverse-scored items directly from the source configuration.
    """
    
    def __init__(self, bfi_path: str = BFI_PATH):
        self.bfi_data = self._load_bfi(bfi_path)
        self.questions = self._parse_questions()

        # Load reverse-scored item IDs directly from JSON for robust handling
        self.reverse_ids = set(self.bfi_data.get('reverse', []))
        if self.reverse_ids:
            logging.info(f"Loaded {len(self.reverse_ids)} reverse scoring items from BFI configuration.")
    
    def _load_bfi(self, filepath: str) -> Dict:
        """Load the BFI JSON file from disk."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _parse_questions(self) -> List[InterviewQuestion]:
        """Parse raw JSON entries into structured InterviewQuestion objects."""
        questions = []
        reverse_list = set(self.bfi_data.get('reverse', []))
        
        for q_id, q_data in self.bfi_data.get('questions', {}).items():
            q_id_int = int(q_id)
            
            question = InterviewQuestion(
                id=q_id_int,
                original_statement=q_data.get('origin_en', ''),
                question=q_data.get('rewritten_en', ''),
                dimension=q_data.get('dimension', ''),
                category=q_data.get('category', 'positive'),
                # Mark question as reverse-keyed if its ID appears in the reverse list
                is_reverse=(q_id_int in reverse_list)
            )
            questions.append(question)
        
        return questions
    
    def get_questions_by_dimension(self, dimension: str) -> List[InterviewQuestion]:
        """Retrieve all questions corresponding to a specific personality dimension."""
        return [q for q in self.questions if q.dimension == dimension]
    
    def get_all_dimensions(self) -> List[str]:
        """Retrieve a list of all unique personality dimensions present in the inventory."""
        return list(set(q.dimension for q in self.questions))

# =============================================================================
# Structured Interview
# =============================================================================

class StructuredInterview:
    """
    Conducts structured interviews by presenting BFI questions to the AI Agent.
    Manages the flow of questioning and the subsequent scoring of responses.
    """
    
    def __init__(
        self, 
        question_loader: BFIQuestionLoader,
        evaluator_client = None  # LLM client (e.g., GPT-4) used as the judge/scorer
    ):
        self.questions = question_loader.questions
        self.evaluator = evaluator_client
        self.dimensions = question_loader.get_all_dimensions()
    
    def conduct_interview(
        self, 
        ai_client,
        persona: str,
        questions_per_dimension: int = 5
    ) -> List[InterviewResponse]:
        """
        Execute a BFI interview session.
        Randomly samples a subset of questions for each dimension to ensure robust evaluation.
        """
        responses = []
        
        for dimension in self.dimensions:
            dim_questions = [q for q in self.questions if q.dimension == dimension]
            
            # Randomly select a subset of questions to avoid order effects and overfitting
            num_to_select = min(len(dim_questions), questions_per_dimension)
            selected = random.sample(dim_questions, num_to_select)
            
            for question in selected:
                # Elicit response from the AI Agent
                response_text = self._ask_question(ai_client, persona, question)
                
                response = InterviewResponse(
                    question_id=question.id,
                    question=question.question,
                    response=response_text,
                    dimension=dimension
                )
                
                responses.append(response)
        
        return responses
    
    def _ask_question(
        self, 
        ai_client, 
        persona: str, 
        question: InterviewQuestion
    ) -> str:
        """Present a single interview question to the AI Agent."""
        
        prompt = f"""You are being interviewed about your personality. 
Please answer the following question honestly and in character:

{question.question}

Provide a natural response that reflects your true feelings and behaviors."""
        
        # Invoke AI client generation method
        if hasattr(ai_client, 'generate_response'):
            return ai_client.generate_response(
                context=[{"role": "user", "content": prompt}],
                persona=persona
            )
        elif hasattr(ai_client, 'generate'):
            return ai_client.generate(prompt)
        else:
            return "[Error: AI client not properly configured for response generation]"
    
    def score_responses(
        self, 
        responses: List[InterviewResponse]
    ) -> List[InterviewResponse]:
        """Batch score a list of interview responses using the evaluator."""

        scored_responses = []
        
        for response in responses:
            score = self._score_single_response(response)
            response.score = score
            scored_responses.append(response)
        
        return scored_responses
    
    def _score_single_response(self, response: InterviewResponse) -> int:
        """
        Score a single response on a 1-5 Likert scale.
        Uses an LLM judge if available; otherwise falls back to keyword matching.
        """
        
        if self.evaluator is None:
            return self._keyword_score(response)
        
        # LLM Judge Prompt: Designed to emulate a psychological rater
        scoring_prompt = f"""You are an expert psychologist evaluating personality test responses.

Evaluate the following response to a personality assessment question.

Question: "{response.question}" 
Subject's Response: "{response.response}"

Based strictly on the response, to what extent does the subject demonstrate the trait implied by the question?
Rate on a Likert scale of 1 to 5:
1 = Strongly disagree / Trait is absent
2 = Somewhat disagree
3 = Neither agree nor disagree / Ambiguous
4 = Somewhat agree
5 = Strongly agree / Trait is clearly present

Respond with ONLY a single number from 1 to 5."""
        
        try:
            score_text = self.evaluator.generate(scoring_prompt)
            # Extract the first digit found in the response
            match = re.search(r'\d', score_text)
            if match:
                score = int(match.group())
                return max(1, min(5, score)) # Clamp score to valid range
            return 3
        except Exception:
            return 3  # Default to neutral on error
    
    def _keyword_score(self, response: InterviewResponse) -> int:
        """Heuristic-based scoring fallback using simple keyword analysis."""
        text = response.response.lower()
        
        positive_keywords = ['yes', 'definitely', 'always', 'love', 'enjoy', 'often', 'very']
        negative_keywords = ['no', 'never', 'rarely', 'hate', 'dislike', 'don\'t']
        
        pos_count = sum(1 for kw in positive_keywords if kw in text)
        neg_count = sum(1 for kw in negative_keywords if kw in text)
        
        if pos_count > neg_count + 1:
            return 5
        elif pos_count > neg_count:
            return 4
        elif neg_count > pos_count + 1:
            return 1
        elif neg_count > pos_count:
            return 2
        else:
            return 3

# =============================================================================
# Personality Evaluator
# =============================================================================

class PersonalityEvaluator:
    """
    Computes psychometric metrics based on scored interview responses.
    Handles reverse-scoring logic and aggregation of results.
    """
    
    def __init__(self, bfi_loader: BFIQuestionLoader):
        self.bfi = bfi_loader
        self.reverse_items = self.bfi.reverse_ids
    
    def compute_scores(
        self, 
        responses: List[InterviewResponse]
    ) -> PersonalityScore:
        """
        Aggregate individual response scores into dimension-level personality scores.
        Applies reverse-scoring correction where necessary.
        """
        dimension_scores = defaultdict(list)
        
        for response in responses:
            if response.score is not None:
                raw_score = response.score
                
                # Apply reverse scoring logic:
                # If item is reverse-keyed, flip the scale (1->5, 5->1) using: 6 - score
                if response.question_id in self.reverse_items:
                    final_score = 6 - raw_score
                else:
                    final_score = raw_score
                
                dimension_scores[response.dimension].append(final_score)
        
        # Calculate mean score per dimension
        scores = PersonalityScore(
            openness=np.mean(dimension_scores.get('Openness', [3])),
            conscientiousness=np.mean(dimension_scores.get('Conscientiousness', [3])),
            extraversion=np.mean(dimension_scores.get('Extraversion', [3])),
            agreeableness=np.mean(dimension_scores.get('Agreeableness', [3])),
            neuroticism=np.mean(dimension_scores.get('Neuroticism', [3]))
        )
        
        return scores
    
    def compute_accuracy(
        self,
        predicted_scores: PersonalityScore,
        target_big5: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate binary accuracy for each dimension.
        A dimension is accurate (1.0) if the predicted level (High/Low) matches the target.
        """
        accuracy = {}
        
        for dimension in ['Openness', 'Conscientiousness', 'Extraversion', 
                          'Agreeableness', 'Neuroticism']:
            target = target_big5.get(dimension, 'high')
            predicted = predicted_scores.get_level(dimension)
            
            accuracy[dimension] = 1.0 if target == predicted else 0.0
        
        return accuracy
    
    def compute_hit_at_k(
        self,
        predicted_scores: PersonalityScore,
        target_big5: Dict[str, str]
    ) -> Dict[int, bool]:
        """
        Compute Hit@k metrics.
        Hit@k is True if the model correctly matched at least k dimensions.
        """
        dimensions = ['Openness', 'Conscientiousness', 'Extraversion', 
                      'Agreeableness', 'Neuroticism']
        
        matches = 0
        for dim in dimensions:
            target = target_big5.get(dim, 'high')
            predicted = predicted_scores.get_level(dim)
            if target == predicted:
                matches += 1
        
        # Determine if threshold k is met for all k in [0..5]
        hit_at_k = {k: matches >= k for k in range(6)}
        
        return hit_at_k
    
    def compute_consistency_score(
        self,
        hit_at_k: Dict[int, bool],
        k: int = 3
    ) -> float:
        """
        Calculate the consistency score.
        Defined as passing a specific Hit@k threshold (default k=3).
        """
        return 1.0 if hit_at_k.get(k, False) else 0.0
    
    def evaluate(
        self,
        responses: List[InterviewResponse],
        target_big5: Dict[str, str]
    ) -> EvaluationResult:
        """
        Perform a complete evaluation cycle: scoring, aggregation, and metric computation.
        """
        # 1. Compute aggregate scores
        predicted_scores = self.compute_scores(responses)
        
        # 2. Derive predicted levels (High/Low)
        predicted_big5 = {
            dim: predicted_scores.get_level(dim)
            for dim in ['Openness', 'Conscientiousness', 'Extraversion',
                        'Agreeableness', 'Neuroticism']
        }
        
        # 3. Compute evaluation metrics
        accuracy = self.compute_accuracy(predicted_scores, target_big5)
        hit_at_k = self.compute_hit_at_k(predicted_scores, target_big5)
        consistency = self.compute_consistency_score(hit_at_k)
        
        return EvaluationResult(
            target_big5=target_big5,
            predicted_scores=predicted_scores,
            predicted_big5=predicted_big5,
            consistency_score=consistency,
            dimension_accuracy=accuracy,
            hit_at_k=hit_at_k,
            responses=responses
        )

# =============================================================================
# Random Questions Evaluation
# =============================================================================

class RandomQuestionEvaluator:
    """Evaluates personality consistency through unstructured, random dialogues."""
    
    # The "Golden List" of 15 open-ended questions designed to elicit personality traits
    RANDOM_QUESTIONS = [
        "Tell me about your ideal weekend.",
        "What is something you are truly passionate about?",
        "How do you handle stress or difficult situations?",
        "Describe a recent accomplishment you are proud of.",
        "What kind of work environment do you thrive in?",
        "Tell me about your closest friendships.",
        "What are your thoughts on taking risks?",
        "What is your approach to learning new skills?",
        "Describe a time when you had to speak in public.",
        "How do you make important decisions?",
        "What role does creativity play in your life?",
        "How do you respond to criticism?",
        "Tell me about a time when you had to adapt to significant change.",
        "Describe a time when you disagreed with a colleague or a peer. How did you handle it?",
        "If you could change one thing about the world, what would it be?"
    ]
    
    # Keyword mappings for heuristic-based consistency checking
    DIMENSION_KEYWORDS = {
        'Openness': {
            'high': ['creative', 'curious', 'explore', 'new', 'ideas', 'imagine', 'art'],
            'low': ['practical', 'routine', 'traditional', 'familiar', 'stable']
        },
        'Conscientiousness': {
            'high': ['organized', 'plan', 'careful', 'responsible', 'thorough', 'goal'],
            'low': ['flexible', 'spontaneous', 'relaxed', 'easy-going', 'casual']
        },
        'Extraversion': {
            'high': ['social', 'friends', 'party', 'energy', 'outgoing', 'talk', 'people'],
            'low': ['quiet', 'alone', 'peace', 'calm', 'solitude', 'private']
        },
        'Agreeableness': {
            'high': ['help', 'kind', 'understand', 'cooperate', 'harmony', 'care'],
            'low': ['independent', 'challenge', 'debate', 'direct', 'honest']
        },
        'Neuroticism': {
            'high': ['worry', 'stress', 'anxious', 'nervous', 'emotional', 'sensitive'],
            'low': ['calm', 'relaxed', 'stable', 'confident', 'composed']
        }
    }
    
    def __init__(self, evaluator_client=None):
        self.evaluator = evaluator_client

    def get_random_questions(self, n: int = 5) -> List[str]:
        """Select a random subset of n questions from the pool."""
        return random.sample(self.RANDOM_QUESTIONS, min(n, len(self.RANDOM_QUESTIONS)))
    
    def evaluate_response(
        self,
        response: str,
        question: str
    ) -> Dict[str, int]:
        """
        Heuristic evaluation of a response against personality dimensions.
        Returns an adjustment score (-1, 0, +1) for each dimension based on keyword presence.
        """
        adjustments = {}
        response_lower = response.lower()
        
        for dimension, keywords in self.DIMENSION_KEYWORDS.items():
            high_count = sum(1 for kw in keywords['high'] if kw in response_lower)
            low_count = sum(1 for kw in keywords['low'] if kw in response_lower)
            
            if high_count > low_count:
                adjustments[dimension] = 1
            elif low_count > high_count:
                adjustments[dimension] = -1
            else:
                adjustments[dimension] = 0
        
        return adjustments

# =============================================================================
# Aggregated Evaluation
# =============================================================================

class AggregatedEvaluator:
    """Accumulates and summarizes evaluation results across multiple personas."""
    
    def __init__(self):
        self.results: List[EvaluationResult] = []
    
    def add_result(self, result: EvaluationResult):
        """Add a single evaluation result to the aggregate."""
        self.results.append(result)
    
    def compute_hit_rate_distribution(self) -> Dict[int, int]:
        """Compute the frequency distribution of Hit@k scores across all samples."""
        distribution = {k: 0 for k in range(6)}
        
        for result in self.results:
            # Determine the highest k for which hit@k is True
            max_k = 0
            for k in range(6):
                if result.hit_at_k.get(k, False):
                    max_k = k
            distribution[max_k] += 1
        
        return distribution
    
    def compute_dimension_accuracy(self) -> Dict[str, float]:
        """Calculate the average accuracy percentage for each personality dimension."""
        dimension_accuracies = defaultdict(list)
        
        for result in self.results:
            for dim, acc in result.dimension_accuracy.items():
                dimension_accuracies[dim].append(acc)
        
        return {
            dim: np.mean(accs) * 100  # Return as percentage
            for dim, accs in dimension_accuracies.items()
        }
    
    def compute_overall_consistency(self) -> float:
        """Compute the mean consistency score across all evaluations."""
        if not self.results:
            return 0.0
        return np.mean([r.consistency_score for r in self.results])
    
    def generate_report(self) -> str:
        """Compile a formatted summary report of the evaluation metrics."""
        hit_dist = self.compute_hit_rate_distribution()
        dim_acc = self.compute_dimension_accuracy()
        consistency = self.compute_overall_consistency()
        
        report = f"""
Personality Evaluation Report
{'=' * 50}

Number of evaluations: {len(self.results)}

Hit Rate Distribution:
{'-' * 30}
"""
        for k, count in hit_dist.items():
            report += f"  hit@{k}: {count}\n"
        
        report += f"""
Dimension Accuracy:
{'-' * 30}
"""
        for dim, acc in dim_acc.items():
            report += f"  {dim}: {acc:.2f}%\n"
        
        report += f"""
Overall Consistency Score: {consistency:.2f}
{'=' * 50}
"""
        return report

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Example usage for testing module functionality
    print("Evaluation Module Demo")
    print("=" * 50)
    
    # Load BFI questions
    try:
        bfi_loader = BFIQuestionLoader()
        print(f"Loaded {len(bfi_loader.questions)} BFI questions")
        
        # Show sample questions
        print("\nSample questions by dimension:")
        for dim in bfi_loader.get_all_dimensions():
            questions = bfi_loader.get_questions_by_dimension(dim)
            print(f"\n{dim}: {len(questions)} questions")
            if questions:
                print(f"  Example: {questions[0].question}")
    except FileNotFoundError:
        print("BFI.json not found - using demo mode")
    
    # Demo evaluation result
    demo_scores = PersonalityScore(
        openness=4.2,
        conscientiousness=3.8,
        extraversion=4.5,
        agreeableness=3.2,
        neuroticism=2.8
    )
    
    print(f"\nDemo scores: {demo_scores.to_dict()}")
    print(f"Predicted levels: O={demo_scores.get_level('Openness')}, "
          f"C={demo_scores.get_level('Conscientiousness')}, "
          f"E={demo_scores.get_level('Extraversion')}, "
          f"A={demo_scores.get_level('Agreeableness')}, "
          f"N={demo_scores.get_level('Neuroticism')}")
