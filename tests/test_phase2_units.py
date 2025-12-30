"""
Phase 2 Unit Tests
Contract: Test individual Phase 2 functions against spec.
Run these before integration validation.
"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.answer_extraction import extract_answer, normalize_answer, Phase2Error


class TestAnswerExtraction:
    """Test answer extraction against Phase 2 contract."""
    
    def test_extract_answer_explicit_pattern_high_confidence(self):
        """Test Strategy 1: explicit 'answer is' pattern."""
        answer, conf = extract_answer("The answer is 847.", "numerical_comparison")
        assert "847" in answer
        assert conf == 0.9
    
    def test_extract_answer_number_only_medium_confidence(self):
        """Test Strategy 2: number extraction."""
        answer, conf = extract_answer("847", "numerical_comparison")
        assert answer == "847"
        assert conf == 0.7
    
    def test_extract_answer_first_sentence_low_confidence(self):
        """Test Strategy 3: first sentence."""
        answer, conf = extract_answer("I think it's 847 because of math", "numerical_comparison")
        assert "847" in answer
        assert 0.2 <= conf <= 0.7
    
    def test_extract_answer_fallback_lowest_confidence(self):
        """Test Strategy 4: full text fallback."""
        answer, conf = extract_answer("Some rambling text without clear answer", "numerical_comparison")
        assert answer == "Some rambling text without clear answer"
        assert conf == 0.2
    
    def test_extract_answer_confidence_in_range(self):
        """Test confidence always in [0, 1]."""
        test_cases = [
            "The answer is 847.",
            "847",
            "Therefore, 839 is larger.",
            "Random text"
        ]
        for text in test_cases:
            _, conf = extract_answer(text, "numerical_comparison")
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of range for: {text}"
    
    def test_extract_answer_never_empty(self):
        """Test answer is never empty string."""
        test_cases = [
            "The answer is 847.",
            "847",
            "Some text without numbers"
        ]
        for text in test_cases:
            answer, _ = extract_answer(text, "numerical_comparison")
            assert answer != "", f"Empty answer for: {text}"
    
    def test_extract_answer_rejects_non_phase2_category(self):
        """Test Phase 2 boundary: only numerical_comparison supported."""
        with pytest.raises(Phase2Error, match="not supported in Phase 2"):
            extract_answer("Paris", "factual_comparison")
    
    def test_extract_answer_empty_input_raises_error(self):
        """Test empty input raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            extract_answer("", "numerical_comparison")
    
    def test_extract_answer_whitespace_only_raises_error(self):
        """Test whitespace-only input raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            extract_answer("   ", "numerical_comparison")


class TestNormalizeAnswer:
    """Test answer normalization against Phase 2 contract."""
    
    def test_normalize_lowercase(self):
        """Test converts to lowercase."""
        assert normalize_answer("ABC") == "abc"
    
    def test_normalize_removes_punctuation(self):
        """Test removes punctuation."""
        result = normalize_answer("The answer is 847.")
        assert "." not in result
    
    def test_normalize_extracts_numbers(self):
        """Test extracts first number."""
        assert normalize_answer("The answer is 847") == "847"
        assert normalize_answer("847 is larger than 839") == "847"
    
    def test_normalize_deterministic(self):
        """Test same input gives same output."""
        input_text = "The answer is 847."
        result1 = normalize_answer(input_text)
        result2 = normalize_answer(input_text)
        assert result1 == result2
    
    def test_normalize_empty_input(self):
        """Test empty input returns empty string."""
        assert normalize_answer("") == ""
    
    def test_normalize_whitespace(self):
        """Test normalizes whitespace."""
        result = normalize_answer("multiple    spaces")
        assert "    " not in result


class TestPhase2Contracts:
    """Test Phase 2 contract enforcement."""
    
    def test_phase2_only_supports_numerical_comparison(self):
        """Test Phase 2 explicitly rejects non-numerical categories."""
        unsupported_categories = [
            "factual_comparison",
            "date_reasoning",
            "logical_puzzles"
        ]
        
        for category in unsupported_categories:
            with pytest.raises(Phase2Error, match="not supported in Phase 2"):
                extract_answer("Some answer", category)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

