#!/usr/bin/env python3
"""
Test script for training batch processing optimizations.

This script tests the performance improvements from the applied optimizations
by comparing optimized vs standard training configurations.
"""

import torch
import time
import logging
from pathlib import Path

# Add smartcash to path for imports
import sys
sys.path.append('/Users/masdevid/Projects/smartcash')

from smartcash.model.training.core.training_executor import TrainingExecutor
from smartcash.model.training.data_loader_factory import DataLoaderFactory
from smartcash.ui.model.training.configs.training_defaults import get_default_training_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockModel(torch.nn.Module):
    """Mock model for testing training performance."""
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 255, 1)  # YOLOv5-like output
        self.to(device)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        # Return YOLOv5-like output format
        return [x, x.clone(), x.clone()]  # Simulate 3 detection layers


class MockProgressTracker:
    """Mock progress tracker for testing."""
    
    def start_batch_tracking(self, num_batches):
        pass
        
    def update_batch_progress(self, current, total, message, **kwargs):
        pass
        
    def complete_batch_tracking(self):
        pass


class MockLossManager:
    """Mock loss manager for testing."""
    
    def compute_loss(self, predictions, targets, image_size):
        # Simple mock loss calculation
        loss = torch.tensor(0.5, requires_grad=True)
        return loss, {'total_loss': 0.5}


def create_mock_dataloader(batch_size: int, num_batches: int, device: str = 'cpu'):
    """Create mock dataloader for testing."""
    class MockDataset:
        def __init__(self, size):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Mock image (batch will be created by DataLoader)
            image = torch.randn(3, 640, 640)
            # Mock targets in YOLO format: [batch_idx, class, x, y, w, h]
            targets = torch.tensor([[0, 1, 0.5, 0.5, 0.3, 0.3]])  # Single object
            return image, targets
    
    from torch.utils.data import DataLoader
    dataset = MockDataset(num_batches * batch_size)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing for simple test
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.cat([torch.cat([torch.tensor([[i]]), item[1]], dim=1) for i, item in enumerate(batch)])
        )
    )


def test_training_performance():
    """Test training performance with and without optimizations."""
    logger.info("🚀 Testing Training Performance Optimizations")
    logger.info("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Standard Training',
            'config': {
                'mixed_precision': False,
                'gradient_accumulation': False,
                'compile_model': False,
            }
        },
        {
            'name': 'Optimized Training',
            'config': {
                'mixed_precision': True and torch.cuda.is_available(),
                'gradient_accumulation': True,
                'accumulation_steps': 2,
                'compile_model': False,  # Disabled for compatibility
            }
        }
    ]
    
    # Test parameters
    batch_size = 4
    num_batches = 10
    
    results = []
    
    for test_config in test_configs:
        logger.info(f"\n📊 Testing: {test_config['name']}")
        logger.info(f"   Config: {test_config['config']}")
        
        try:
            # Create mock components
            model = MockModel(device)
            progress_tracker = MockProgressTracker()
            loss_manager = MockLossManager()
            
            # Create training executor with config
            training_executor = TrainingExecutor(
                model=model,
                config=test_config['config'],
                progress_tracker=progress_tracker
            )
            
            # Create mock dataloader
            train_loader = create_mock_dataloader(batch_size, num_batches, device)
            
            # Create optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            # Warm up run (don't count in timing)
            logger.info("   Warming up...")
            try:
                # Create smaller warmup loader
                warmup_loader = create_mock_dataloader(batch_size, 2, device)
                training_executor.train_epoch(
                    warmup_loader, optimizer, loss_manager, None, 0, 1, 1, 1
                )
            except Exception as e:
                logger.warning(f"   Warmup failed: {e}")
            
            # Performance test
            logger.info("   Running performance test...")
            start_time = time.time()
            
            result = training_executor.train_epoch(
                train_loader, optimizer, loss_manager, None, 0, 1, 1, 1
            )
            
            end_time = time.time()
            
            training_time = end_time - start_time
            time_per_batch = training_time / num_batches
            
            logger.info(f"   ✅ Completed in {training_time:.3f}s ({time_per_batch:.3f}s per batch)")
            logger.info(f"   Loss: {result.get('train_loss', 0):.4f}")
            
            test_result = {
                'config_name': test_config['name'],
                'training_time': training_time,
                'time_per_batch': time_per_batch,
                'loss': result.get('train_loss', 0),
                'effective_batch_size': result.get('effective_batch_size', batch_size),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"   ❌ Test failed: {e}")
            test_result = {
                'config_name': test_config['name'],
                'success': False,
                'error': str(e)
            }
        
        results.append(test_result)
    
    # Performance comparison
    logger.info("\n📋 PERFORMANCE COMPARISON")
    logger.info("=" * 60)
    
    successful_results = [r for r in results if r.get('success')]
    
    if len(successful_results) >= 2:
        standard = next((r for r in successful_results if 'Standard' in r['config_name']), None)
        optimized = next((r for r in successful_results if 'Optimized' in r['config_name']), None)
        
        if standard and optimized:
            speedup = standard['training_time'] / optimized['training_time']
            time_saved = standard['training_time'] - optimized['training_time']
            
            logger.info(f"Standard Training:    {standard['training_time']:.3f}s ({standard['time_per_batch']:.3f}s/batch)")
            logger.info(f"Optimized Training:   {optimized['training_time']:.3f}s ({optimized['time_per_batch']:.3f}s/batch)")
            logger.info(f"Speedup:              {speedup:.2f}x")
            logger.info(f"Time saved:           {time_saved:.3f}s ({time_saved/standard['training_time']*100:.1f}%)")
            logger.info(f"Effective batch size: {optimized.get('effective_batch_size', batch_size)}")
        
        # DataLoader configuration test
        logger.info(f"\n🔄 DATALOADER OPTIMIZATION TEST")
        logger.info("=" * 40)
        
        try:
            # Test optimized DataLoader configuration
            factory = DataLoaderFactory()
            config = factory._get_optimal_dataloader_config()
            
            logger.info("✅ DataLoader optimization applied:")
            logger.info(f"   Workers: {config['num_workers']}")
            logger.info(f"   Prefetch factor: {config['prefetch_factor']}")
            logger.info(f"   Pin memory: {config['pin_memory']}")
            logger.info(f"   Persistent workers: {config['persistent_workers']}")
            
        except Exception as e:
            logger.error(f"❌ DataLoader optimization test failed: {e}")
    
    else:
        logger.warning("Not enough successful results for comparison")
    
    logger.info("\n✅ Performance testing completed!")
    return results


def test_system_optimizations():
    """Test system-level optimizations."""
    logger.info("\n🔧 SYSTEM OPTIMIZATION TEST")
    logger.info("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test CUDA optimizations
    if torch.cuda.is_available():
        logger.info("✅ CUDA optimizations:")
        logger.info(f"   cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        logger.info(f"   cuDNN deterministic: {torch.backends.cudnn.deterministic}")
    
    # Test CPU thread optimization
    import os
    cpu_threads = torch.get_num_threads()
    optimal_threads = min(os.cpu_count() or 4, 8)
    
    logger.info(f"✅ CPU optimization:")
    logger.info(f"   CPU cores: {os.cpu_count()}")
    logger.info(f"   PyTorch threads: {cpu_threads}")
    logger.info(f"   Optimal threads: {optimal_threads}")
    
    # Test mixed precision availability
    mixed_precision_available = torch.cuda.is_available()
    logger.info(f"✅ Mixed precision: {'Available' if mixed_precision_available else 'Not available (CPU only)'}")
    
    # Test model compilation availability
    model_compile_available = hasattr(torch, 'compile')
    logger.info(f"✅ Model compilation: {'Available (PyTorch 2.0+)' if model_compile_available else 'Not available'}")


if __name__ == '__main__':
    try:
        # Run performance tests
        results = test_training_performance()
        
        # Run system optimization tests
        test_system_optimizations()
        
        logger.info("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        logger.error(traceback.format_exc())