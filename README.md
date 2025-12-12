# Humanized Conversation with Big5 Personality

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of the paper "Humanized Conversation with Personality: Integrating Big Five Personality Traits into Conversational AI" by Chuqiao Huang, Columbia University EECS E6895, Spring 2024.

## 📖 Overview

This project develops a virtual agent capable of simulating nuanced human conversations infused with distinct Big Five personality traits. By integrating established psychological models into AI systems, we create interactions that are both technically adept and emotionally resonant.

### Key Contributions

- **Dual-Component Framework**: Combines Llama3-8B-Instruct for personality-aware response generation with BERT for dynamic memory selection
- **BIG-5 Dataset**: Proprietary dataset comprising 320 personas (32 Big Five combinations × 10 characters each)
- **Rigorous Evaluation**: Structured interview methodology achieving 46% hit@5 and 77% hit@4 personality consistency

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Conversational AI System                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐        ┌─────────────────────┐     │
│  │   Memory Selector   │        │ Response Generator  │     │
│  │    (BERT-based)     │───────▶│   (Llama3-8B +      │     │
│  │                     │        │      QLoRA)         │     │
│  │  P_η(m_k|C,P,M)     │        │  P_θ(y_k|C,P,m_k)   │     │
│  └─────────────────────┘        └─────────────────────┘     │
│           │                              │                  │
│           │                              │                  │
│           ▼                              ▼                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Big-5 Persona Bank                      │   │
│  │  320 Personas × {O, C, E, A, N} ∈ {high, low}^5      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```


## 🚀 Quick Start

### Prerequisites

```bash
# System requirements
- Python 3.8+
- CUDA-capable GPU (NVIDIA A100 40GB recommended)
- 85GB RAM
- 500GB SSD storage

# Key dependencies
- PyTorch 2.0+
- Transformers 4.30+
- PEFT (LoRA)
- sentence-transformers
```

### Installation

```bash
# Clone the repository
git clone https://github.com/cordelia0325/big5-persona-chat.git
cd big5-persona-chat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional packages for training
pip install bitsandbytes accelerate
```

### Data Preparation

```bash
# 1. Prepare your persona dataset (JSON format)
# Place in data/big5-persona.json

# 2. Create stratified train/val/test splits
python make_stratified_splits.py

# This generates:
#   - data/train_dataset.json (256 personas, 80%)
#   - data/validation_dataset.json (32 personas, 10%)
#   - data/test_dataset.json (32 personas, 10%)
# With balanced representation across all 32 Big-5 combinations

# 3. (Optional) Download PERSONA-CHAT dataset
# Place in data/personality.csv
```

## 📂 Project Structure

```
.
├── config.py                    # Configuration (hyperparameters, paths)
├── data_processing.py           # Data loading and preprocessing
├── memory_selector.py           # BERT-based memory selection
├── response_generator.py        # Llama3 response generation with QLoRA
├── evaluation.py                # Personality evaluation metrics
├── train.py                     # Training pipeline
├── inference.py                 # Inference utilities
├── run_evaluation.py            # Evaluation script
├── make_stratified_splits.py    # Dataset splitting (CRITICAL)
│
├── data/
│   ├── big5-1024-persona.json   # Full persona dataset
│   ├── train_dataset.json       # Training split (generated)
│   ├── validation_dataset.json  # Validation split (generated)
│   ├── test_dataset.json        # Test split (generated)
│   ├── BFI.json                 # Big Five Inventory questions
│   └── personality.csv          # PERSONA-CHAT dataset
│
├── models/                      # Saved model checkpoints
├── outputs/                     # Training outputs and logs
└── README.md                    # This file
```

## 🎯 Usage

### Demo: Basic Functionality

```bash
# Run all demos
python main.py

# Or run specific demos
python main.py --demo basic      # Data loading demo
python main.py --demo agent      # Persona agent creation
python main.py --demo memory     # Memory selection demo
python main.py --demo eval       # Evaluation metrics demo
python main.py --demo api        # Start API server
```

### Training

#### Option 1: Train All Models

```bash
python train.py \
  --train-dataset-path data/train_dataset.json \
  --val-dataset-path data/validation_dataset.json \
  --train-all \
  --seed 42
```

#### Option 2: Train Separately

```bash
# Train BERT memory selector only
python train.py \
  --train-dataset-path data/train_dataset.json \
  --val-dataset-path data/validation_dataset.json \
  --train-bert \
  --bert-epochs 3 \
  --bert-batch-size 16 \
  --bert-lr 2e-5

# Train Llama3 response generator only
python train.py \
  --train-dataset-path data/train_dataset.json \
  --val-dataset-path data/validation_dataset.json \
  --train-llama \
  --llama-epochs 3 \
  --llama-batch-size 4 \
  --llama-lr 2e-4
```

#### Training Configuration

The implementation uses **Parameter-Efficient Fine-Tuning (PEFT)** with QLoRA:

| Model | Parameter | Value | Paper Reference |
|-------|-----------|-------|-----------------|
| **BERT Memory Selector** | | | |
| Learning Rate | `bert_lr` | 2e-5 | Table 10 |
| Batch Size | `bert_batch_size` | 16 | Table 10 |
| Warmup Steps | `warmup_steps` | 500 | Table 10 |
| Training Time | - | ~2 hours | Section E.2 |
| **Llama3 Response Generator** | | | |
| Quantization | `load_in_4bit` | NF4 | Table 11 |
| LoRA Rank | `lora_r` | 16 | Table 13 |
| LoRA Alpha | `lora_alpha` | 32 | Table 13 |
| LoRA Dropout | `lora_dropout` | 0.05 | Table 13 |
| Learning Rate | `llama_lr` | 2e-4 | Table 14 |
| Batch Size | `llama_batch_size` | 4 | Table 14 |
| Grad Accumulation | `gradient_accumulation_steps` | 4 | Table 14 |
| Training Time | - | ~8 hours | Section E.2 |

### Evaluation (requires OpenAI API Key)

```bash
export OPENAI_API_KEY="your-api-key"

# Run evaluation on test set
python run_evaluation.py \
  --test-dataset-path data/test_dataset.json \
  --model-path models/run_YYYYMMDD_HHMMSS/response_generator \
  --bert-path models/run_YYYYMMDD_HHMMSS/memory_selector/bert_memory_selector.pt \
  --n-trials 3 \
  --questions-per-dim 5

# Generate evaluation report
python run_evaluation.py \
  --test-dataset-path data/test_dataset.json \
  --model-path models/run_YYYYMMDD_HHMMSS/response_generator \
  --output-report ouputs/evaluation_report.json
```

### Inference

```python
from personality_model import PersonalityModelFactory
from data_processing import DataLoader, Persona

# Load a persona
loader = DataLoader()
personas = loader.load_big5_personas('data/test_dataset.json')
persona = personas[0]

# Create model and agent
model = PersonalityModelFactory.create_model(
    load_llama=True,
    load_bert=True,
    llama_path='models/run_YYYYMMDD_HHMMSS/response_generator',
    bert_path='models/run_YYYYMMDD_HHMMSS/memory_selector/bert_memory_selector.pt'
)
agent = PersonalityModelFactory.create_agent(persona, model)

# Generate response
response = agent.chat("Hi! Tell me about your hobbies.")
print(f"{persona.name}: {response}")
```

## 📊 Dataset Information

### BIG-5 Persona Bank

Our proprietary dataset contains **320 personas** systematically designed across all 32 possible Big Five combinations:

```python
# Binary Big Five encoding (2^5 = 32 combinations)
Dimensions = {
    'O': Openness       → {high: Inventive/Curious, low: Consistent/Cautious}
    'C': Conscientiousness → {high: Efficient/Organized, low: Extravagant/Careless}
    'E': Extraversion   → {high: Outgoing/Energetic, low: Solitary/Reserved}
    'A': Agreeableness  → {high: Friendly/Compassionate, low: Critical/Judgmental}
    'N': Neuroticism    → {high: Resilient/Confident, low: Sensitive/Nervous}
}

# Each persona includes:
- Name, age, gender, region
- Occupation and communication tone
- Detailed personality description
- Growth experience and family relationships
- Working/social/living conditions
- Current concerns and hobbies
# Total: ~471 words per persona
```

### PERSONA-CHAT Dataset

We augment training with the public PERSONA-CHAT dataset:
- **Total dialogues**: 8,939
- **Average turns per dialogue**: 12-15
- **Average utterance length**: 15-20 words
- **Training split**: 7,151 (80%)
- **Validation split**: 894 (10%)
- **Test split**: 894 (10%)

### Data Format

```json
{
  "uid": persona-0000,
  "big-5": "{high, high, high, high, high}",
  "profile": {
    "name": "Luna Everhart",
    "gender": "Female",
    "age": "35",
    "region": "Brooklyn, New York, USA",
    "job": "Freelance Artist",
    "tone": "Luna's speaking style is animated and expressive...",
    "personality": "Luna is a free-spirited and imaginative individual...",
    "hobby": "Luna's hobbies include painting, sketching...",
    "growth_experience": "Growing up in a bohemian household...",
    "family_relationship": "Luna has a close and supportive relationship...",
    "working_conditions": "Luna's art studio is a vibrant space...",
    "social_relationship": "Luna's social circle consists of...",
    "emotional_state": "Luna's typical emotional baseline is...",
    "living_conditions": "Luna's home is a cozy apartment...",
    "recent_worry_or_anxiety": "Luna's recent worry revolves around...",
    "additional_information": "Luna is known for her quirky style..."
  }
}
```

## 🔬 Methodology

### Objective Function

The system implements the dual-objective function from Section 6.1:

```
P(y_k | C, P, M) ≈ P_θ(y_k | C, P, m_k) · P_η(m_k | C, P, M)

Where:
  C = Dialogue context
  P = Persona (Big Five profile)
  M = Memory bank
  m_k = Selected memory
  y_k = Generated response
  
  P_θ = Response generation model (Llama3-8B-Instruct)
  P_η = Memory selection model (BERT)
```

### Memory Selection

Memories are selected based on semantic similarity:

```python
# Positive samples: embedding similarity > 0.7
# Negative samples: random memories with similarity ≤ 0.7
# Loss function: Binary Cross-Entropy

label = 1 if cosine_similarity(response, memory) > 0.7 else 0
```

### Evaluation Protocol

Following the structured interview methodology (Section 7):

1. **BFI-44 Assessment**: 5 randomly selected questions per dimension
2. **Random Questions**: 5 questions from a pool of 15 behavioral queries
3. **Repetition**: Each interview conducted 3 times (T=0.7)
4. **Scoring**: Responses rated 1-5 using GPT-4-turbo
5. **Metrics**:
   - **hit@k**: Number of dimensions correctly predicted
   - **Dimension Accuracy**: Percentage of correct high/low classifications
   - **Consistency Score**: Overall personality alignment

## 📈 Reproducing Paper Results

### Step 1: Prepare Environment

```bash
# Ensure CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi
```

### Step 2: Data Preparation (CRITICAL)

```bash
# IMPORTANT: Use stratified splitting
python make_stratified_splits.py

# Verify balanced representation
python -c "
import json
from collections import Counter

for split in ['train', 'validation', 'test']:
    with open(f'data/{split}_dataset.json') as f:
        data = json.load(f)
    combos = Counter(p['big-5'] for p in data)
    print(f'{split}: {len(combos)}/32 combinations')
    assert len(combos) == 32, f'Missing combinations in {split}!'
print('All splits have balanced representation')
"
```

### Step 3: Training

```bash
# Train both models (requires ~10 hours on A100)
python train.py \
  --train-dataset-path data/train_dataset.json \
  --val-dataset-path data/validation_dataset.json \
  --train-all \
  --seed 42 \
  2>&1 | tee training.log
```

### Step 4: Evaluation

```bash
# Run evaluation with 3 trials per persona
python run_evaluation.py \
  --test-dataset-path data/test_dataset.json \
  --model-path outputs/run_YYYYMMDD_HHMMSS/response_generator \
  --bert-path outputs/run_YYYYMMDD_HHMMSS/memory_selector/bert_memory_selector.pt \
  --n-trials 3 \
  --evaluator gpt-4-turbo \
  --output-report results/evaluation_report.json

# Analyze results
python analyze_results.py results/evaluation_report.json
```

## ⚙️ Configuration

All hyperparameters are centralized in `config.py`:

```python
# Key configurations
BIG5_PERSONA_PATH = "data/big5-persona.json"
PERSONA_CHAT_PATH = "data/personality.csv"

# Model paths
LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
BERT_MODEL = "bert-base-uncased"

# Training
TRAIN_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
TEMPERATURE = 0.7

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train.py --llama-batch-size 2 --bert-batch-size 8
   
   # Or use gradient checkpointing (slower but memory-efficient)
   # Set in config.py: gradient_checkpointing = True
   ```

2. **ImportError: No module named 'bitsandbytes'**
   ```bash
   pip install bitsandbytes
   # On Windows, use: pip install bitsandbytes-windows
   ```

3. **Unbalanced Test Set**
   ```bash
   # DO NOT use make_test_set.py (simple random sampling)
   # ALWAYS use make_stratified_splits.py (stratified sampling)
   python make_stratified_splits.py
   ```

4. **Model Loading Issues**
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface/
   python train.py --train-bert  # Start with BERT only
   ```

## 📝 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{huang2024humanized,
  title={Humanized Conversation with Personality: Integrating Big Five Personality Traits into Conversational AI},
  author={Huang, Chuqiao},
  journal={Columbia University EECS E6895 Final Project Report},
  year={2024},
  month={May},
  institution={Columbia University},
  address={New York, NY, USA}
}
```

## 🙏 Acknowledgments

- **Advisor**: Professor Ching-Yung Lin, Columbia University
- **Course**: EECS E6895 (Advanced Big Data Analytics and AI)
- **Institution**: Columbia University, Department of Applied Physics and Applied Mathematics
- **Current Affiliation**: Mohamed bin Zayed University of AI (MBZUAI)

Special thanks to:
- Meta AI for Llama3-8B-Instruct
- Hugging Face for Transformers library
- The creators of the PERSONA-CHAT dataset

## 📧 Contact

**Chuqiao Huang**  
- Email: ch3801@columbia.edu
- GitHub: [@cordelia0325](https://github.com/cordelia0325)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Work

- [Llama 3 Model Card](https://ai.meta.com/blog/meta-llama-3/)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PERSONA-CHAT Dataset](https://arxiv.org/abs/1801.07243)
- [Big Five Personality Inventory](https://doi.org/10.1037/t07550-000)

---

<div align="center">
  
**⭐ If you find this project useful, please consider giving it a star! ⭐**

[Report Issues](https://github.com/cordelia0325/humanized-conversation-with-big5-personality/issues) · [Request Features](https://github.com/cordelia0325/humanized-conversation-with-big5-personality/issues)

Made with ❤️ at Columbia University & MBZUAI

</div>
