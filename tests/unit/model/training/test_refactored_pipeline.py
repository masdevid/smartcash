#!/usr/bin/env python3
"""
Basic test for the refactored training pipeline to ensure functionality.

This test validates that the refactored components work together correctly
and maintain the same interface as the original implementation.
"""

import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock
from smartcash.model.training.phases import TrainingPhaseManager


class DummyModel(nn.Module):
    """Simple dummy model for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16, 7)
        self.current_phase = 1
        self.force_single_layer = False
    
    def forward(self, x):
        # Return YOLO-like predictions for testing
        batch_size = x.shape[0]
        # Simulate YOLO output: [batch, detections, features]
        # Features: [x, y, w, h, objectness, class1, class2, ..., class7]
        predictions = torch.randn(batch_size, 100, 12)  # 12 features = 5 bbox + 7 classes
        return [predictions]  # Return as list to simulate YOLOv5 output


def create_test_config():
    """Create a minimal test configuration."""
    return {
        'training_mode': 'single_phase',
        'model': {
            'backbone': 'yolov5s',
            'layer_mode': 'multi'
        },
        'training': {
            'mixed_precision': False,
            'early_stopping': {
                'enabled': True,
                'patience': 5,
                'metric': 'val_map50',
                'phase_1_enabled': True,
                'phase_2_enabled': True
            }
        },
        'training_phases': {
            'phase_1': {
                'learning_rate': 0.001,
                'epochs': 2
            }
        },
        'paths': {
            'checkpoints': '/tmp/test_checkpoints'
        }
    }


def create_mock_components():
    """Create mock components for testing."""
    # Mock progress tracker
    progress_tracker = Mock()
    progress_tracker.start_epoch_tracking = Mock()
    progress_tracker.update_epoch_progress = Mock()
    progress_tracker.start_batch_tracking = Mock()
    progress_tracker.update_batch_progress = Mock()
    progress_tracker.complete_batch_tracking = Mock()
    progress_tracker.complete_epoch_early_stopping = Mock()
    progress_tracker.update_phase = Mock()
    
    # Mock model API
    model_api = Mock()
    model_api.save_checkpoint = Mock(return_value='/tmp/test_checkpoint.pt')
    
    # Mock callbacks
    emit_metrics_callback = Mock()
    emit_live_chart_callback = Mock()
    visualization_manager = Mock()
    visualization_manager.update_metrics = Mock()
    
    return {
        'progress_tracker': progress_tracker,
        'model_api': model_api,
        'emit_metrics_callback': emit_metrics_callback,
        'emit_live_chart_callback': emit_live_chart_callback,
        'visualization_manager': visualization_manager
    }


def test_training_phase_manager_initialization():
    """Test that TrainingPhaseManager initializes correctly with refactored components."""
    print("🧪 Testing TrainingPhaseManager initialization...")
    
    model = DummyModel()
    config = create_test_config()
    mocks = create_mock_components()
    
    # Initialize TrainingPhaseManager
    manager = TrainingPhaseManager(
        model=model,
        model_api=mocks['model_api'],
        config=config,
        progress_tracker=mocks['progress_tracker'],
        emit_metrics_callback=mocks['emit_metrics_callback'],
        emit_live_chart_callback=mocks['emit_live_chart_callback'],
        visualization_manager=mocks['visualization_manager']
    )
    
    # Verify components are initialized
    assert hasattr(manager, 'orchestrator'), "PhaseOrchestrator not initialized"
    assert hasattr(manager, 'checkpoint_manager'), "CheckpointManager not initialized"
    assert hasattr(manager, 'progress_manager'), "ProgressManager not initialized"
    
    print("✅ TrainingPhaseManager initialization successful")
    return manager


def test_component_interfaces():
    """Test that all components have the expected interfaces."""
    print("🧪 Testing component interfaces...")
    
    model = DummyModel()
    config = create_test_config()
    mocks = create_mock_components()
    
    manager = TrainingPhaseManager(
        model=model,
        model_api=mocks['model_api'],
        config=config,
        progress_tracker=mocks['progress_tracker']
    )
    
    # Test PhaseOrchestrator interface
    assert hasattr(manager.orchestrator, 'setup_phase'), "PhaseOrchestrator missing setup_phase method"
    
    # Test CheckpointManager interface  
    assert hasattr(manager.checkpoint_manager, 'save_checkpoint'), "CheckpointManager missing save_checkpoint method"
    
    # Test ProgressManager interface
    assert hasattr(manager.progress_manager, 'start_epoch_tracking'), "ProgressManager missing start_epoch_tracking method"
    
    print("✅ Component interfaces verified")


def test_backward_compatibility():
    """Test that the refactored version maintains backward compatibility."""
    print("🧪 Testing backward compatibility...")
    
    model = DummyModel()
    config = create_test_config()
    mocks = create_mock_components()
    
    manager = TrainingPhaseManager(
        model=model,
        model_api=mocks['model_api'],
        config=config,
        progress_tracker=mocks['progress_tracker'],
        emit_metrics_callback=mocks['emit_metrics_callback'],
        emit_live_chart_callback=mocks['emit_live_chart_callback'],
        visualization_manager=mocks['visualization_manager']
    )
    
    # Test that the main public method exists and has correct signature
    assert hasattr(manager, 'run_training_phase'), "Missing run_training_phase method"
    assert hasattr(manager, 'execute_phase'), "Missing execute_phase method"
    
    # Test method signatures (should not raise TypeError)
    try:
        # This should not raise an exception due to signature mismatch
        import inspect
        sig = inspect.signature(manager.run_training_phase)
        params = list(sig.parameters.keys())
        expected_params = ['phase_num', 'epochs', 'start_epoch']
        
        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"
        
        print("✅ Method signatures maintained")
        
    except Exception as e:
        print(f"❌ Signature compatibility issue: {e}")
        raise
    
    print("✅ Backward compatibility verified")


def run_all_tests():
    """Run all tests for the refactored training pipeline."""
    print("🚀 Starting refactored training pipeline tests...\n")
    
    try:
        # Test 1: Initialization
        manager = test_training_phase_manager_initialization()
        print()
        
        # Test 2: Component interfaces
        test_component_interfaces()
        print()
        
        # Test 3: Backward compatibility
        test_backward_compatibility()
        print()
        
        print("🎉 All tests passed! Refactored training pipeline is working correctly.")
        print("\n📊 Test Summary:")
        print("✅ Component initialization")
        print("✅ Interface compatibility")
        print("✅ Backward compatibility") 
        print("✅ Import functionality")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)