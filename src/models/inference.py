"""
Model inference utilities.

This module provides functions for loading models and generating responses.
"""

import re
import torch
from typing import Dict, Tuple, Optional


def load_model(model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
    """
    Load model and tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"✓ Model loaded on {model.device}")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    temperature: float = 0.6,
    max_tokens: int = 512
) -> str:
    """
    Generate response for a single question.
    
    Args:
        model: Loaded model
        tokenizer: Corresponding tokenizer
        question: Question string
        temperature: Sampling temperature (default: 0.6)
        max_tokens: Maximum tokens to generate (default: 512)
    
    Returns:
        Full model response (string)
    """
    # Construct prompt
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that answers numerical comparison questions."
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                "You will be given a question that requires comparison or calculation.\n\n"
                "First, think through the problem step by step.\n"
                "Then, give a final Yes or No answer.\n\n"
                "Format your response EXACTLY as follows:\n\n"
                "REASONING:\n"
                "<your step-by-step reasoning>\n\n"
                "FINAL_ANSWER:\n"
                "<Yes or No>\n\n"
                "Do not include anything else."
            )
        }
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove prompt from response
    response_text = response_text.replace(prompt, "").strip()
    
    # Handle chat template markers
    assistant_markers = ["<｜Assistant｜>", "<|assistant|>", "Assistant:", "<|im_start|>assistant"]
    for marker in assistant_markers:
        if marker in response_text:
            response_text = response_text.split(marker, 1)[1].strip()
            break
    
    return response_text


def parse_response(response_text: str, question: str) -> Dict[str, any]:
    """
    Parse response into reasoning and answer components.
    
    Args:
        response_text: Raw model response
        question: Original question (for fallback extraction)
    
    Returns:
        dict with:
            - reasoning (str): Extracted reasoning
            - answer (str): Extracted answer (Yes/No/Unknown)
            - raw_response (str): Original response
            - is_valid_format (bool): Whether format was followed
    """
    # Strip <think> tags if present
    if "<think>" in response_text and "</think>" in response_text:
        think_start = response_text.index("<think>")
        think_end = response_text.index("</think>") + len("</think>")
        response_text = response_text[think_end:].strip()
    
    # Try to find answer with flexible matching
    answer_match = re.search(
        r'(?:final[\s_]answer|answer):\s*(yes|no)', 
        response_text, 
        re.IGNORECASE
    )
    
    if answer_match:
        answer = answer_match.group(1).capitalize()
        is_valid = True
        
        # Extract reasoning as everything before the answer marker
        answer_start = answer_match.start()
        reasoning = response_text[:answer_start].strip()
        
        # Remove "REASONING:" label if present
        if reasoning.startswith("REASONING:"):
            reasoning = reasoning[len("REASONING:"):].strip()
    
    # Fallback: Look for explicit REASONING: and FINAL_ANSWER: markers
    elif "REASONING:" in response_text or "FINAL_ANSWER:" in response_text:
        if "REASONING:" in response_text:
            reasoning_start = response_text.index("REASONING:") + len("REASONING:")
            
            if "FINAL_ANSWER:" in response_text:
                reasoning_end = response_text.index("FINAL_ANSWER:")
                reasoning = response_text[reasoning_start:reasoning_end].strip()
                
                answer_start = response_text.index("FINAL_ANSWER:") + len("FINAL_ANSWER:")
                answer_text = response_text[answer_start:].strip()
            else:
                reasoning = response_text[reasoning_start:].strip()
                answer_text = response_text[:reasoning_start]
            
            answer_lower = answer_text.lower()
            if "yes" in answer_lower and "no" not in answer_lower:
                answer = "Yes"
                is_valid = True
            elif "no" in answer_lower and "yes" not in answer_lower:
                answer = "No"
                is_valid = True
            else:
                answer = _extract_yes_no_fallback(response_text)
                is_valid = False
        
        elif "FINAL_ANSWER:" in response_text:
            answer_start = response_text.index("FINAL_ANSWER:") + len("FINAL_ANSWER:")
            answer_text = response_text[answer_start:].strip()
            reasoning = response_text[:answer_start].strip()
            
            if answer_text.lower().startswith("yes"):
                answer = "Yes"
                is_valid = True
            elif answer_text.lower().startswith("no"):
                answer = "No"
                is_valid = True
            else:
                answer = _extract_yes_no_fallback(response_text)
                is_valid = False
    
    # Last resort: fallback extraction
    else:
        answer = _extract_yes_no_fallback(response_text)
        reasoning = response_text[:500]  # Use first 500 chars
        is_valid = False
    
    # If we have answer but no reasoning, use full response
    if is_valid and not reasoning:
        reasoning = response_text[:500]
    
    return {
        'reasoning': reasoning,
        'answer': answer,
        'raw_response': response_text,
        'is_valid_format': is_valid
    }


def _extract_yes_no_fallback(text: str) -> str:
    """Fallback Yes/No extraction."""
    text_lower = text.lower()
    
    # Simple yes/no search in last 150 chars
    tail = text_lower[-150:]
    yes_pos = tail.rfind('yes')
    no_pos = tail.rfind('no')
    
    if yes_pos != -1 and no_pos != -1:
        return "Yes" if yes_pos > no_pos else "No"
    elif yes_pos != -1:
        return "Yes"
    elif no_pos != -1:
        return "No"
    
    return "Unknown"

