"""
Phase 2: Batch Inference
Contract: Generate model responses for all question pairs from Phase 1.
Output: JSONL file with exactly 100 responses (50 pairs × 2 variants)
"""
import json
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple


# Phase 2 constants (from spec)
PHASE1_INPUT = "data/raw/question_pairs.json"
PHASE2_OUTPUT = "data/responses/model_1.5B_responses.jsonl"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_TEMPERATURE = 0.6
MAX_NEW_TOKENS = 2048
TOP_P = 0.95


class Phase2Error(Exception):
    """Raised when Phase 2 contracts are violated."""
    pass


def load_model(model_name: str = DEFAULT_MODEL):
    """
    Load model and tokenizer.
    
    Phase 2 Contract:
    - Returns (model, tokenizer)
    - Model loaded with float16 and auto device mapping
    """
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("✓ Model loaded")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    temperature: float = DEFAULT_TEMPERATURE
) -> str:
    """
    Generate response for a single question.
    
    Phase 2 Contract:
    - Returns non-empty string
    - Uses system prompt with <think> tag instructions
    - Generates max 2048 tokens
    - Uses top_p=0.95
    - Strips original prompt from output
    
    Args:
        model: Loaded HuggingFace model
        tokenizer: Corresponding tokenizer
        question: Question string
        temperature: Sampling temperature
    
    Returns:
        Full model response (string, prompt removed)
    
    Raises:
        Phase2Error: If response is empty
    """
    # Phase 2 system prompt (updated for Yes/No format)
    prompt = (
        "You are a helpful AI assistant. Think through the problem step by step "
        "before providing your final answer. Put your reasoning in <think></think> "
        "tags, then provide your final answer as either 'Yes' or 'No'.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate (Phase 2 parameters from spec)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the model's response (remove prompt)
    response = response.replace(prompt, "").strip()
    
    if not response:
        raise Phase2Error(f"Model returned empty response for question: {question}")
    
    return response


def extract_think_and_answer(response: str) -> Tuple[str, str]:
    """
    Split response into think section and final answer.
    
    Phase 2 Contract:
    - Returns (think_section, final_answer)
    - If no <think> tags, uses 70/30 heuristic split
    - At least one of think_section or final_answer is non-empty
    
    Args:
        response: Full model response
    
    Returns:
        (think_section, final_answer)
    """
    # Look for <think> tags
    if "<think>" in response and "</think>" in response:
        start = response.index("<think>") + len("<think>")
        end = response.index("</think>")
        think_section = response[start:end].strip()
        final_answer = response[end + len("</think>"):].strip()
    else:
        # Fallback: first 70% is think, rest is answer (from spec)
        split_point = int(len(response) * 0.7)
        think_section = response[:split_point].strip()
        final_answer = response[split_point:].strip()
    
    return think_section, final_answer


def run_inference(
    questions_path: str = PHASE1_INPUT,
    output_path: str = PHASE2_OUTPUT,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE
) -> None:
    """
    Run inference on all question pairs from Phase 1.
    
    Phase 2 Contract:
    - Reads exactly 50 pairs from Phase 1
    - Generates exactly 100 responses (50 × 2 variants)
    - Each response has required fields per Phase 2 data contract
    - Writes to JSONL format
    
    Args:
        questions_path: Path to Phase 1 question pairs
        output_path: Path to write Phase 2 responses
        model_name: HuggingFace model identifier
        temperature: Sampling temperature
    
    Raises:
        Phase2Error: If Phase 1 input invalid or Phase 2 contracts violated
    """
    # Validate Phase 1 input exists
    if not Path(questions_path).exists():
        raise Phase2Error(
            f"Phase 1 input not found: {questions_path}\n"
            f"Phase 2 depends on Phase 1 completion. Run Phase 1 first."
        )
    
    # Load questions
    with open(questions_path) as f:
        data = json.load(f)
        pairs = data.get('pairs', [])
    
    # Phase 2 contract: exactly 50 pairs
    if len(pairs) != 50:
        raise Phase2Error(
            f"Phase 2 expects exactly 50 pairs from Phase 1, got {len(pairs)}\n"
            f"This violates Phase 1 → Phase 2 contract."
        )
    
    print(f"Loaded {len(pairs)} question pairs from Phase 1")
    print(f"Total prompts to generate: {len(pairs) * 2}")
    
    # Load model
    model, tokenizer = load_model(model_name)
    
    # Generate responses
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    response_count = 0
    with jsonlines.open(output_path, mode='w') as writer:
        for pair in tqdm(pairs, desc="Generating responses"):
            # Validate Phase 1 pair has required fields
            required_fields = ['id', 'q1', 'q2']
            for field in required_fields:
                if field not in pair:
                    raise Phase2Error(
                        f"Phase 1 pair {pair.get('id', 'unknown')} missing required field: {field}"
                    )
            
            # Generate for q1
            response_q1 = generate_response(model, tokenizer, pair['q1'], temperature)
            think_q1, answer_q1 = extract_think_and_answer(response_q1)
            
            writer.write({
                "pair_id": pair['id'],
                "variant": "q1",
                "question": pair['q1'],
                "response": response_q1,
                "think_section": think_q1,
                "final_answer": answer_q1,
                "timestamp": datetime.now().isoformat(),
                "generation_config": {
                    "temperature": temperature,
                    "model": model_name
                }
            })
            response_count += 1
            
            # Generate for q2
            response_q2 = generate_response(model, tokenizer, pair['q2'], temperature)
            think_q2, answer_q2 = extract_think_and_answer(response_q2)
            
            writer.write({
                "pair_id": pair['id'],
                "variant": "q2",
                "question": pair['q2'],
                "response": response_q2,
                "think_section": think_q2,
                "final_answer": answer_q2,
                "timestamp": datetime.now().isoformat(),
                "generation_config": {
                    "temperature": temperature,
                    "model": model_name
                }
            })
            response_count += 1
    
    # Phase 2 contract: exactly 100 responses
    if response_count != 100:
        raise Phase2Error(
            f"Phase 2 contract violated: generated {response_count} responses, expected 100"
        )
    
    print(f"✓ Generated {response_count} responses")
    print(f"✓ Saved to {output_path}")


if __name__ == "__main__":
    run_inference()

