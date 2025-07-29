#!/usr/bin/env python3
"""
Test DataLoader compatibility fixes for PyTorch 2.7.1.

This script tests the DataLoader factory with the _workers_status AttributeError fix.
"""

# Fix OpenMP duplicate library issue
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_dataloader_creation():
    """Test DataLoader creation with PyTorch 2.7+ compatibility"""
    print("🧪 Testing DataLoader creation with PyTorch compatibility fixes...")
    
    try:
        from smartcash.model.training.data_loader_factory import DataLoaderFactory
        
        # Create a minimal test config
        test_config = {
            'training': {
                'batch_size': 2,
                'data': {
                    'num_workers': 2,  # Use fewer workers for testing
                    'pin_memory': False,  # Disable for testing
                    'persistent_workers': False,  # Disable for PyTorch 2.7+
                    'prefetch_factor': 2,
                    'drop_last': True,
                    'timeout': 10
                }
            }
        }
        
        # Create temporary data directories with minimal test data
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create directory structure
            for split in ['train', 'valid']:
                (temp_path / split / 'images').mkdir(parents=True)
                (temp_path / split / 'labels').mkdir(parents=True)
                
                # Create minimal test data
                for i in range(2):  # Create 2 test files
                    # Create dummy .npy image file
                    dummy_image = np.random.rand(3, 640, 640).astype(np.float32)
                    image_path = temp_path / split / 'images' / f'pre_test_{i}.npy'
                    np.save(image_path, dummy_image)
                    
                    # Create dummy label file
                    label_path = temp_path / split / 'labels' / f'pre_test_{i}.txt'
                    with open(label_path, 'w') as f:
                        f.write("0 0.5 0.5 0.1 0.1\n")  # class x y w h
            
            # Test DataLoaderFactory creation
            factory = DataLoaderFactory(config=test_config, data_dir=str(temp_path))
            print(f"   ✅ DataLoaderFactory created successfully")
            
            # Test train loader creation
            train_loader = factory.create_train_loader(img_size=640)
            print(f"   ✅ Training DataLoader created (num_workers={train_loader.num_workers})")
            
            # Test validation loader creation
            val_loader = factory.create_val_loader(img_size=640)
            print(f"   ✅ Validation DataLoader created (num_workers={val_loader.num_workers})")
            
            # Test basic iteration (this is where the _workers_status error would occur)
            print("   🔍 Testing DataLoader iteration...")
            
            # Test train loader iteration
            train_iter = iter(train_loader)
            batch = next(train_iter)
            print(f"   ✅ Training batch loaded: images shape {batch[0].shape}, targets shape {batch[1].shape}")
            
            # Test validation loader iteration
            val_iter = iter(val_loader)
            batch = next(val_iter)
            print(f"   ✅ Validation batch loaded: images shape {batch[0].shape}, targets shape {batch[1].shape}")
            
            # Test cleanup functionality
            factory.cleanup()
            print(f"   ✅ DataLoader cleanup completed successfully")
            
            return True
            
    except Exception as e:
        print(f"   ❌ DataLoader test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_pytorch_version_detection():
    """Test PyTorch version detection and configuration adjustment"""
    print("\n🧪 Testing PyTorch version detection...")
    
    try:
        from smartcash.model.training.data_loader_factory import DataLoaderFactory
        
        # Test with mock PyTorch versions
        with patch('torch.__version__', '2.7.1'):
            factory = DataLoaderFactory()
            config = factory._get_fallback_config()
            
            # Should use reduced workers and disabled persistent_workers for 2.7+
            data_config = config['training']['data']
            assert data_config['num_workers'] == 2, f"Expected 2 workers, got {data_config['num_workers']}"
            assert data_config['persistent_workers'] == False, f"Expected persistent_workers=False, got {data_config['persistent_workers']}"
            print(f"   ✅ PyTorch 2.7+ configuration: workers={data_config['num_workers']}, persistent={data_config['persistent_workers']}")
        
        with patch('torch.__version__', '2.6.0'):
            factory = DataLoaderFactory()
            config = factory._get_fallback_config()
            
            # Should use original settings for older versions
            data_config = config['training']['data']
            assert data_config['num_workers'] == 4, f"Expected 4 workers, got {data_config['num_workers']}"
            assert data_config['persistent_workers'] == True, f"Expected persistent_workers=True, got {data_config['persistent_workers']}"
            print(f"   ✅ PyTorch 2.6 configuration: workers={data_config['num_workers']}, persistent={data_config['persistent_workers']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Version detection test failed: {str(e)}")
        return False

def test_worker_cleanup_compatibility():
    """Test worker cleanup compatibility across PyTorch versions"""
    print("\n🧪 Testing worker cleanup compatibility...")
    
    try:
        from smartcash.model.training.data_loader_factory import DataLoaderFactory
        
        # Create mock DataLoader with different worker attributes
        mock_dataloader = MagicMock()
        
        # Test case 1: DataLoader with _iterator that has _shutdown_workers
        mock_iterator = MagicMock()
        mock_iterator._shutdown_workers = MagicMock()
        mock_dataloader._iterator = mock_iterator
        
        factory = DataLoaderFactory()
        factory._dataloaders = [mock_dataloader]
        
        # This should not raise an exception
        factory.cleanup()
        mock_iterator._shutdown_workers.assert_called_once()
        print("   ✅ Iterator _shutdown_workers cleanup works")
        
        # Test case 2: DataLoader with _iterator that has shutdown method
        mock_iterator2 = MagicMock()
        mock_iterator2.shutdown = MagicMock()
        delattr(mock_iterator2, '_shutdown_workers')  # Remove _shutdown_workers
        mock_dataloader2 = MagicMock()
        mock_dataloader2._iterator = mock_iterator2
        
        factory2 = DataLoaderFactory()
        factory2._dataloaders = [mock_dataloader2]
        
        factory2.cleanup()
        mock_iterator2.shutdown.assert_called_once()
        print("   ✅ Iterator shutdown method cleanup works")
        
        # Test case 3: DataLoader with _shutdown_workers directly
        mock_dataloader3 = MagicMock()
        mock_dataloader3._iterator = None
        mock_dataloader3._shutdown_workers = MagicMock()
        
        factory3 = DataLoaderFactory()
        factory3._dataloaders = [mock_dataloader3]
        
        factory3.cleanup()
        mock_dataloader3._shutdown_workers.assert_called_once()
        print("   ✅ Direct _shutdown_workers cleanup works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Worker cleanup test failed: {str(e)}")
        return False

def main():
    """Run comprehensive DataLoader compatibility tests"""
    print("🚀 DATALOADER PYTORCH 2.7+ COMPATIBILITY TEST SUITE")
    print("=" * 80)
    print(f"Testing with PyTorch version: {torch.__version__}")
    print("=" * 80)
    
    test_functions = [
        ("DataLoader Creation", test_dataloader_creation),
        ("PyTorch Version Detection", test_pytorch_version_detection),
        ("Worker Cleanup Compatibility", test_worker_cleanup_compatibility)
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_name, test_func in test_functions:
        try:
            result = test_func()
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"💥 {test_name} crashed: {str(e)}")
    
    print("\n" + "=" * 80)
    print("📊 DATALOADER COMPATIBILITY TEST RESULTS")
    print("=" * 80)
    print(f"🎯 Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 ALL DATALOADER COMPATIBILITY TESTS PASSED!")
        print("✅ _workers_status AttributeError has been resolved")
        print("✅ PyTorch 2.7+ multiprocessing compatibility implemented")
        print("✅ Worker cleanup handles all PyTorch versions")
        return True
    else:
        print("⚠️  Some DataLoader compatibility tests failed")
        print("🔍 Review the output above for details")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)