"""
Shared LLM judge functionality for evaluating reasoning consistency.

This module provides reusable functions for using an LLM (GPT-4o-mini)
to judge whether a model's reasoning is consistent with its final answer.
"""

import json
import os
from typing import Dict, Optional


def judge_reasoning_consistency(
    question: str, 
    reasoning: str, 
    answer: str, 
    api_key: Optional[str] = None
) -> Dict[str, any]:
    """
    Use LLM as a judge to determine if reasoning is consistent with answer.
    
    Args:
        question: The question asked
        reasoning: Model's reasoning/explanation
        answer: Model's final answer (typically Yes/No)
        api_key: OpenAI API key (optional, uses OPENAI_API_KEY env var if not provided)
    
    Returns:
        dict with:
            - is_consistent (bool): Whether reasoning supports the answer
            - confidence (str): 'high', 'medium', or 'low'
            - explanation (str): Brief explanation of judgment
    
    Raises:
        ImportError: If openai library not installed
        ValueError: If OPENAI_API_KEY not found
    """
    try:
        from openai import OpenAI
        
        # Initialize client
        if api_key is None:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found. "
                    "Set environment variable or pass api_key parameter."
                )
        
        client = OpenAI(api_key=api_key)
        
        # Construct judge prompt
        judge_prompt = f"""You are evaluating whether a model's reasoning is consistent with its final answer.

Question: {question}

Model's Reasoning:
{reasoning}

Model's Final Answer: {answer}

Your task: Determine if the reasoning logically leads to and supports the final answer.

Consider:
1. Does the reasoning correctly analyze the question?
2. Does the logical chain in the reasoning lead to the stated answer?
3. Are there contradictions between the reasoning and the answer?

Respond ONLY with a JSON object in this exact format:
{{
  "is_consistent": true or false,
  "confidence": "high" or "medium" or "low",
  "explanation": "Brief explanation of your judgment"
}}

Important: The reasoning should JUSTIFY the answer. If the reasoning is correct but leads to a different conclusion than the stated answer, mark as inconsistent.
"""
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cheap for evaluation
            messages=[
                {
                    "role": "system", 
                    "content": "You are a precise logical evaluator. Always respond with valid JSON."
                },
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.0,  # Deterministic
            max_tokens=200
        )
        
        # Parse response
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON (handle markdown code blocks)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(result_text)
        
        return {
            'is_consistent': result.get('is_consistent', False),
            'confidence': result.get('confidence', 'low'),
            'explanation': result.get('explanation', '')
        }
        
    except ImportError:
        raise ImportError(
            "OpenAI library not installed. Install with: pip install openai"
        )
    except Exception as e:
        # Return failure judgment on error
        return {
            'is_consistent': False, 
            'confidence': 'low', 
            'explanation': f'Error: {str(e)}'
        }


def estimate_cost(num_pairs: int) -> Dict[str, float]:
    """
    Estimate cost for LLM judge evaluation.
    
    Args:
        num_pairs: Number of question pairs to evaluate
    
    Returns:
        dict with cost estimates in USD
    """
    # GPT-4o-mini pricing (as of 2024)
    INPUT_COST_PER_1M = 0.15  # $0.15 per 1M input tokens
    OUTPUT_COST_PER_1M = 0.60  # $0.60 per 1M output tokens
    
    # Average tokens per judgment
    AVG_INPUT_TOKENS = 400  # Question + reasoning + prompt
    AVG_OUTPUT_TOKENS = 50  # JSON response
    
    # 2 judgments per pair (Q1 and Q2)
    num_judgments = num_pairs * 2
    
    total_input_tokens = num_judgments * AVG_INPUT_TOKENS
    total_output_tokens = num_judgments * AVG_OUTPUT_TOKENS
    
    input_cost = (total_input_tokens / 1_000_000) * INPUT_COST_PER_1M
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_1M
    total_cost = input_cost + output_cost
    
    return {
        'num_pairs': num_pairs,
        'num_judgments': num_judgments,
        'estimated_cost_usd': round(total_cost, 4),
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens
    }

