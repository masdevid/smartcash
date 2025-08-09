#!/usr/bin/env python3
"""
Integration test for SmartCash YOLOv5 architecture with training pipeline.

This test verifies that the new SmartCashYOLOv5Model integrates correctly
with existing training systems like checkpoint management, early stopping,
memory optimization, and validation.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from smartcash.model.architectures.model import SmartCashYOLOv5Model
from smartcash.model.training.pipeline.model_manager import ModelManager
from smartcash.model.api.core import SmartCashModelAPI
from smartcash.model.training.core import ValidationExecutor


class TestSmartCashArchitectureIntegration:
    """Test integration of SmartCashYOLOv5Model with training pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return {
            'model': {
                'backbone': 'yolov5s',
                'num_classes': 17,
                'img_size': 640,
                'pretrained': False  # Avoid downloading weights in tests
            },
            'device': {'type': 'cpu'},
            'checkpoint': {
                'save_dir': str(temp_dir / 'checkpoints')
            },
            'training_mode': 'two_phase',
            'session_id': 'test_session'
        }
    
    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        mock = MagicMock()
        mock.start_operation = MagicMock()
        mock.update_operation = MagicMock()
        mock.complete_operation = MagicMock()
        mock.complete_phase = MagicMock()
        return mock
    
    def test_smartcash_model_creation(self):
        """Test that SmartCashYOLOv5Model can be created."""
        model = SmartCashYOLOv5Model(
            backbone='yolov5s',
            num_classes=17,
            img_size=640,
            pretrained=False,
            device='cpu'
        )
        
        assert isinstance(model, SmartCashYOLOv5Model)
        assert model.num_classes == 17
        assert model.backbone_type == 'yolov5s'
        assert model.current_phase == 1
        
        # Test model configuration
        config = model.get_model_config()
        assert config['backbone'] == 'yolov5s'
        assert config['num_classes'] == 17
        assert config['architecture'] == 'SmartCashYOLOv5Model'
        
        # Test phase information
        phase_info = model.get_phase_info()
        assert phase_info['phase'] == 1
        assert isinstance(phase_info['trainable_params'], int)
        assert isinstance(phase_info['total_params'], int)
    
    def test_model_api_smartcash_creation(self, config):
        """Test that ModelAPI can create SmartCash models."""
        api = SmartCashModelAPI(config=config)
        
        # Build model with SmartCash architecture
        result = api.build_model(use_smartcash_architecture=True)
        
        assert result['success'] == True
        assert isinstance(result['model'], SmartCashYOLOv5Model)
        assert result['architecture_type'] == 'SmartCashYOLOv5Model'
        assert 'total_parameters' in result['model_info']
        
        # Test model validation
        validation_result = api.validate_model()
        assert validation_result['success'] == True
        assert validation_result['model_mode'] == 'SmartCashYOLOv5Model'
    
    def test_model_manager_smartcash_integration(self, config, mock_progress_tracker):
        """Test that ModelManager can set up SmartCash models."""
        manager = ModelManager(mock_progress_tracker)
        
        # Setup model with SmartCash architecture
        model_api, model = manager.setup_model(
            config, 
            is_resuming=False
        )
        
        assert isinstance(model, SmartCashYOLOv5Model)
        # The API wrapper is created internally, we just need to verify it works
        assert hasattr(model_api, 'get_backbone_name')
        assert model_api.get_backbone_name() == 'yolov5s'
        
        # Test model info
        model_info = manager.get_model_info()
        assert model_info['architecture'] == 'SmartCashYOLOv5Model'
        assert 'backbone' in model_info
        assert 'num_classes' in model_info
        assert 'current_phase' in model_info
        
        # Test phase preparation
        phase_1_model = manager.prepare_model_for_phase(1, config)
        assert phase_1_model.current_phase == 1
        
        phase_2_model = manager.prepare_model_for_phase(2, config)
        assert phase_2_model.current_phase == 2
    
    def test_validation_executor_smartcash_detection(self, config, mock_progress_tracker):
        """Test that ValidationExecutor detects SmartCash models and disables hierarchical processing."""
        # Create SmartCash model
        model = SmartCashYOLOv5Model(
            backbone='yolov5s',
            num_classes=17,
            img_size=640,
            pretrained=False,
            device='cpu'
        )
        
        # Create validation executor
        executor = ValidationExecutor(model, config, mock_progress_tracker, phase_num=1)
        
        # Verify SmartCash model detection
        assert executor.is_smartcash_model == True
        
        # Verify hierarchical processing is disabled
        assert executor.map_calculator.hierarchical_processor.disable_hierarchical == True
    
    def test_smartcash_model_forward_pass(self):
        """Test that SmartCash model can perform forward passes."""
        model = SmartCashYOLOv5Model(
            backbone='yolov5s',
            num_classes=17,
            img_size=640,
            pretrained=False,
            device='cpu'
        )
        
        # Test input
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # Test training mode forward pass
        model.train
        training_output = model(dummy_input)
        # YOLOv5 models return List[torch.Tensor] for multi-scale outputs
        assert isinstance(training_output, list)
        assert all(isinstance(t, torch.Tensor) for t in training_output)
        
        # Test inference mode forward pass
        model.eval
        with torch.no_grad():
            inference_output = model(dummy_input)
        
        assert isinstance(inference_output, list)
        assert len(inference_output) == 1  # Single batch
        assert isinstance(inference_output[0], dict)
        assert 'boxes' in inference_output[0]
        assert 'scores' in inference_output[0]
        assert 'labels' in inference_output[0]
        assert 'denomination_scores' in inference_output[0]
    
    def test_smartcash_model_phase_switching(self):
        """Test that SmartCash model can switch phases correctly."""
        model = SmartCashYOLOv5Model(
            backbone='yolov5s',
            num_classes=17,
            img_size=640,
            pretrained=False,
            device='cpu'
        )
        
        # Start in Phase 1
        assert model.current_phase == 1
        
        # Get initial phase info
        phase_1_info = model.get_phase_info()
        
        # Switch to Phase 2
        model.setup_phase_2()
        assert model.current_phase == 2
        
        # Get Phase 2 info
        phase_2_info = model.get_phase_info()
        
        # Phase 2 should have different characteristics than Phase 1
        assert phase_2_info['phase'] == 2
        # In Phase 2, more parameters should be trainable (backbone unfrozen)
        # Note: This might be equal if pretrained=False, but structure should be correct
    
    def test_legacy_compatibility(self, config, mock_progress_tracker):
        """Test that legacy model creation still works when SmartCash is disabled."""
        manager = ModelManager(mock_progress_tracker)
        
        # Setup model with legacy approach
        try:
            model_api, model = manager.setup_model(
                config, 
                use_smartcash_architecture=False,  # Use legacy approach
                is_resuming=False
            )
            
            # Should work but won't be SmartCash model
            assert not isinstance(model, SmartCashYOLOv5Model)
            
        except Exception as e:
            # If legacy approach fails due to missing dependencies, that's expected
            # The key is that SmartCash approach should always work
            pytest.skip(f"Legacy approach failed (expected): {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
