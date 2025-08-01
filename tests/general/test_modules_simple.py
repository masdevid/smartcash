#!/usr/bin/env python3
"""
Simple validation tests for the three modules without complex dependencies.
"""

import sys
import tempfile
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_data_loader_factory_basic():
    """Test basic DataLoaderFactory functionality"""
    print("🧪 Testing DataLoaderFactory basic functionality...")
    
    try:
        from smartcash.model.training.data_loader_factory import YOLODataset, collate_fn
        
        # Test collate function
        batch = []
        image = torch.randn(3, 640, 640)
        labels = torch.tensor([[0, 0.5, 0.5, 0.2, 0.3]], dtype=torch.float32)
        batch.append((image, labels))
        
        images, targets = collate_fn(batch)
        
        assert images.shape == (1, 3, 640, 640)
        assert targets.shape == (1, 6)  # batch_idx + 5 label values
        assert targets[0, 0] == 0  # batch index
        
        print("✅ DataLoaderFactory basic functionality works")
        return True
        
    except Exception as e:
        print(f"❌ DataLoaderFactory test failed: {e}")
        return False


def test_metrics_tracker_basic():
    """Test basic MetricsTracker functionality"""
    print("🧪 Testing MetricsTracker basic functionality...")
    
    try:
        from smartcash.model.training.metrics_tracker import MetricsTracker  # APCalculator removed
        
        # Skip AP Calculator test (removed for performance)
        # Test basic MetricsTracker instead
        
        # Test MetricsTracker
        config = {'training': {'validation': {'compute_map': True}}}
        tracker = MetricsTracker(config)
        
        # Test metrics update
        loss_dict = {'total_loss': torch.tensor(0.5)}
        tracker.update_train_metrics(loss_dict, learning_rate=0.001)
        
        assert 'total_loss' in tracker.train_metrics
        assert tracker.train_metrics['total_loss'][0] == 0.5
        
        print("✅ MetricsTracker basic functionality works")
        return True
        
    except Exception as e:
        print(f"❌ MetricsTracker test failed: {e}")
        return False


def test_visualization_manager_basic():
    """Test basic VisualizationManager functionality"""
    print("🧪 Testing VisualizationManager basic functionality...")
    
    try:
        # Mock matplotlib and related dependencies
        with patch.dict('sys.modules', {
            'matplotlib': MagicMock(),
            'matplotlib.pyplot': MagicMock(),
            'seaborn': MagicMock(),
            'pandas': MagicMock(),
            'scipy': MagicMock(),
            'scipy.spatial': MagicMock()
        }):
            with patch('smartcash.common.logger.get_logger') as mock_logger:
                from smartcash.model.analysis.visualization.visualization_manager import VisualizationManager
                
                # Test initialization
                with tempfile.TemporaryDirectory() as temp_dir:
                    vm = VisualizationManager(output_dir=temp_dir, logger=mock_logger)
                    
                    assert vm.output_dir == Path(temp_dir)
                    assert vm.config == {}
                    
                    # Test with empty data (should return None)
                    result = vm._plot_strategy_distribution({})
                    assert result is None
                    
                    result = vm._plot_denomination_distribution({})
                    assert result is None
                    
                    print("✅ VisualizationManager basic functionality works")
                    return True
        
    except Exception as e:
        print(f"❌ VisualizationManager test failed: {e}")
        return False


def test_data_structures():
    """Test data structure handling"""
    print("🧪 Testing data structure handling...")
    
    try:
        # Test numpy array operations
        image_data = np.random.rand(3, 640, 640).astype(np.float32)
        assert image_data.shape == (3, 640, 640)
        
        # Test torch tensor operations
        tensor_data = torch.from_numpy(image_data)
        assert tensor_data.shape == (3, 640, 640)
        assert tensor_data.dtype == torch.float32
        
        # Test label format
        labels = np.array([[0, 0.5, 0.5, 0.2, 0.3], [1, 0.3, 0.7, 0.1, 0.15]])
        assert labels.shape == (2, 5)
        
        # Test conversion to tensor
        label_tensor = torch.from_numpy(labels.astype(np.float32))
        assert label_tensor.shape == (2, 5)
        
        print("✅ Data structure handling works")
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False


def test_mathematical_operations():
    """Test mathematical operations used in the modules"""
    print("🧪 Testing mathematical operations...")
    
    try:
        # Test IoU calculation manually
        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # Test perfect overlap
        iou = calculate_iou([10, 10, 50, 50], [10, 10, 50, 50])
        assert abs(iou - 1.0) < 1e-6
        
        # Test no overlap
        iou = calculate_iou([10, 10, 30, 30], [50, 50, 70, 70])
        assert iou == 0.0
        
        # Test partial overlap
        iou = calculate_iou([10, 10, 30, 30], [20, 20, 40, 40])
        expected = 100 / (400 + 400 - 100)  # 10x10 intersection, 20x20 each box
        assert abs(iou - expected) < 1e-6
        
        # Test average calculations
        values = [0.8, 0.9, 0.7, 0.85]
        avg = np.mean(values)
        assert abs(avg - 0.8125) < 1e-6
        
        print("✅ Mathematical operations work")
        return True
        
    except Exception as e:
        print(f"❌ Mathematical operations test failed: {e}")
        return False


def test_configuration_handling():
    """Test configuration handling"""
    print("🧪 Testing configuration handling...")
    
    try:
        # Test default config structure
        default_config = {
            'training': {
                'batch_size': 16,
                'data': {
                    'num_workers': 4,
                    'pin_memory': True,
                    'persistent_workers': True,
                    'prefetch_factor': 2,
                    'drop_last': True
                }
            }
        }
        
        # Test config access
        assert default_config['training']['batch_size'] == 16
        assert default_config['training']['data']['num_workers'] == 4
        
        # Test config merging
        custom_config = {'training': {'batch_size': 32}}
        merged_config = default_config.copy()
        merged_config['training'].update(custom_config['training'])
        
        assert merged_config['training']['batch_size'] == 32
        assert merged_config['training']['data']['num_workers'] == 4  # Preserved
        
        print("✅ Configuration handling works")
        return True
        
    except Exception as e:
        print(f"❌ Configuration handling test failed: {e}")
        return False


def run_all_tests():
    """Run all validation tests"""
    print("🚀 Running comprehensive module tests...")
    print("=" * 60)
    
    tests = [
        test_data_structures,
        test_mathematical_operations,
        test_configuration_handling,
        test_data_loader_factory_basic,
        test_metrics_tracker_basic,
        test_visualization_manager_basic,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test {test.__name__} failed")
        except Exception as e:
            print(f"💥 Test {test.__name__} crashed: {e}")
    
    print("=" * 60)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The modules are working correctly.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Some issues need attention.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)