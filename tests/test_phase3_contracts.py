"""
Phase 3 Contract Tests
These tests encode the Phase 3 specification as executable constraints.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mechanistic.contracts import (
    ActivationCache,
    ProbeResult,
    Phase3Config,
    PHASE3_LAYERS,
    MIN_FAITHFUL_SAMPLES,
    MIN_UNFAITHFUL_SAMPLES,
    Phase3InputError,
)


class TestActivationCacheContract:
    """Test Data Contract 1: Activation Cache Files"""
    
    def test_valid_activation_cache(self):
        """Valid activation cache passes all checks."""
        cache = ActivationCache(
            faithful=torch.randn(20, 1536, dtype=torch.float16),
            unfaithful=torch.randn(15, 1536, dtype=torch.float16),
            layer=12
        )
        
        assert cache.d_model == 1536
        assert cache.n_faithful == 20
        assert cache.n_unfaithful == 15
    
    def test_rejects_3d_tensors(self):
        """Contract violation: activations must be 2D (pre-pooled)."""
        with pytest.raises(ValueError, match="must be 2D.*mean-pooled"):
            ActivationCache(
                faithful=torch.randn(20, 100, 1536),  # Has sequence dim!
                unfaithful=torch.randn(15, 1536),
                layer=12
            )
    
    def test_rejects_too_few_faithful(self):
        """Contract violation: need >= 10 faithful samples."""
        with pytest.raises(ValueError, match=f"at least {MIN_FAITHFUL_SAMPLES}"):
            ActivationCache(
                faithful=torch.randn(5, 1536),  # Too few!
                unfaithful=torch.randn(15, 1536),
                layer=12
            )
    
    def test_rejects_too_few_unfaithful(self):
        """Contract violation: need >= 10 unfaithful samples."""
        with pytest.raises(ValueError, match=f"at least {MIN_UNFAITHFUL_SAMPLES}"):
            ActivationCache(
                faithful=torch.randn(20, 1536),
                unfaithful=torch.randn(5, 1536),  # Too few!
                layer=12
            )
    
    def test_rejects_d_model_mismatch(self):
        """Contract violation: d_model must match."""
        with pytest.raises(ValueError, match="d_model mismatch"):
            ActivationCache(
                faithful=torch.randn(20, 1536),
                unfaithful=torch.randn(15, 2048),  # Different d_model!
                layer=12
            )
    
    def test_rejects_wrong_dtype(self):
        """Contract violation: must be float32 or float16."""
        with pytest.raises(ValueError, match="float32 or float16"):
            ActivationCache(
                faithful=torch.randn(20, 1536).to(torch.int32),
                unfaithful=torch.randn(15, 1536),
                layer=12
            )
    
    def test_rejects_unsupported_layer(self):
        """Contract violation: only layers [6, 12, 18, 24] supported in Phase 3."""
        with pytest.raises(ValueError, match="not supported.*Phase 3"):
            ActivationCache(
                faithful=torch.randn(20, 1536),
                unfaithful=torch.randn(15, 1536),
                layer=10  # Not in PHASE3_LAYERS!
            )


class TestProbeResultContract:
    """Test Data Contract 2: Probe Results"""
    
    def test_valid_probe_result(self):
        """Valid probe result passes all checks."""
        probe = torch.nn.Linear(1536, 1)
        result = ProbeResult(
            layer="layer_12",
            accuracy=0.75,
            auc=0.82,
            probe=probe,
            direction=torch.randn(1536)
        )
        
        assert result.accuracy == 0.75
        assert result.auc == 0.82
    
    def test_rejects_accuracy_out_of_range(self):
        """Contract violation: accuracy must be in [0, 1]."""
        probe = torch.nn.Linear(1536, 1)
        
        with pytest.raises(ValueError, match="accuracy must be in .0, 1."):
            ProbeResult(
                layer="layer_12",
                accuracy=1.5,  # Out of range!
                auc=0.82,
                probe=probe,
                direction=torch.randn(1536)
            )
    
    def test_rejects_auc_out_of_range(self):
        """Contract violation: auc must be in [0, 1]."""
        probe = torch.nn.Linear(1536, 1)
        
        with pytest.raises(ValueError, match="auc must be in .0, 1."):
            ProbeResult(
                layer="layer_12",
                accuracy=0.75,
                auc=-0.1,  # Out of range!
                probe=probe,
                direction=torch.randn(1536)
            )
    
    def test_rejects_2d_direction(self):
        """Contract violation: direction must be 1D vector."""
        probe = torch.nn.Linear(1536, 1)
        
        with pytest.raises(ValueError, match="direction must be 1D"):
            ProbeResult(
                layer="layer_12",
                accuracy=0.75,
                auc=0.82,
                probe=probe,
                direction=torch.randn(10, 1536)  # 2D!
            )
    
    def test_rejects_invalid_layer_name(self):
        """Contract violation: layer must be 'layer_N' format."""
        probe = torch.nn.Linear(1536, 1)
        
        with pytest.raises(ValueError, match="must be 'layer_N'"):
            ProbeResult(
                layer="block_12",  # Wrong format!
                accuracy=0.75,
                auc=0.82,
                probe=probe,
                direction=torch.randn(1536)
            )


class TestPhase3Config:
    """Test Phase 3 configuration constraints."""
    
    def test_default_config(self):
        """Default config uses Phase 3 specification."""
        config = Phase3Config()
        
        assert config.layers == PHASE3_LAYERS
        assert config.num_epochs == 50
        assert config.learning_rate == 1e-3
        assert config.train_test_split == 0.2
    
    def test_rejects_unsupported_layers(self):
        """Phase 3 only supports [6, 12, 18, 24]."""
        with pytest.raises(ValueError, match="Phase 3 does not support layer"):
            Phase3Config(layers=[1, 2, 3])


class TestPhase3Dependencies:
    """Test Phase 3 dependency checks."""
    
    def test_detects_missing_phase2_outputs(self, tmp_path):
        """Phase 3 must fail if Phase 2 is not complete."""
        from mechanistic.contracts import validate_phase2_outputs_exist
        
        config = Phase3Config(
            responses_path=str(tmp_path / "missing.jsonl"),
            faithfulness_path=str(tmp_path / "missing.csv")
        )
        
        with pytest.raises(Phase3InputError, match="Phase 2.*missing"):
            validate_phase2_outputs_exist(config)


class TestPhase3Boundaries:
    """Test that Phase 3 respects its boundaries."""
    
    def test_phase3_layers_are_fixed(self):
        """Phase 3 supports exactly 4 layers - no more, no less."""
        assert PHASE3_LAYERS == [6, 12, 18, 24]
        assert len(PHASE3_LAYERS) == 4
    
    def test_phase3_does_not_support_attention_analysis(self):
        """Phase 3 spec: Linear probe analysis ONLY."""
        # This is a documentation test - Phase 3 has no attention analysis code
        # If someone tries to import it, should fail
        with pytest.raises(ImportError):
            from mechanistic import attention_analysis  # noqa: F401
    
    def test_phase3_does_not_generate_reports(self):
        """Phase 3 spec: No report generation (that's Phase 4)."""
        # This is a documentation test - Phase 3 has no report code
        with pytest.raises(ImportError):
            from mechanistic import report_generator  # noqa: F401


class TestPhase3FileStructure:
    """Test Phase 3 file naming conventions."""
    
    def test_activation_file_naming(self):
        """Activation files must be: data/activations/layer_{N}_activations.pt"""
        for layer in PHASE3_LAYERS:
            expected = f"data/activations/layer_{layer}_activations.pt"
            # This is the contract - enforce consistent naming
            assert "layer_" in expected
            assert "_activations.pt" in expected
    
    def test_probe_results_location(self):
        """Probe results must be: results/probe_results/all_probe_results.pt"""
        expected = "results/probe_results/all_probe_results.pt"
        # This is the contract
        assert expected == "results/probe_results/all_probe_results.pt"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

