#!/usr/bin/env python3
"""
Test script to verify MPS memory management fixes.

This script tests that the training optimizations work properly on MPS
without causing out-of-memory errors.
"""

import torch
import time
import logging
from pathlib import Path
import psutil
import os

# Add smartcash to path for imports
import sys
sys.path.append('/Users/masdevid/Projects/smartcash')

from smartcash.model.training.core.training_executor import TrainingExecutor
from smartcash.model.training.data_loader_factory import DataLoaderFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_memory_info():
    """Get current memory usage information."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    info = {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
    }
    
    if torch.backends.mps.is_available():
        try:
            # MPS doesn't have memory_allocated equivalent, so we'll track process memory
            info['device'] = 'mps'
        except:
            pass
    elif torch.cuda.is_available():
        try:
            info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
            info['device'] = 'cuda'
        except:
            pass
    else:
        info['device'] = 'cpu'
    
    return info


class MockModel(torch.nn.Module):
    """Mock model for testing MPS memory management.""" 
    
    def __init__(self, device='cpu'):
        super().__init__()
        # Much smaller model for MPS memory testing
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.output = torch.nn.Conv2d(128, 255, 1)  # YOLOv5-like output
        
        self.to(device)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.output(x)
        # Return YOLOv5-like output format
        return [x, x.clone(), x.clone()]


class MockProgressTracker:
    def start_batch_tracking(self, num_batches): pass
    def update_batch_progress(self, *args, **kwargs): pass
    def complete_batch_tracking(self): pass


class MockLossManager:
    def compute_loss(self, predictions, targets, image_size):
        # Create realistic loss that requires gradients
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        for pred_list in predictions.values():
            for pred in pred_list:
                # Simple loss calculation
                loss_component = pred.mean() * 0.1
                total_loss = total_loss + loss_component
        
        return total_loss, {'total_loss': total_loss.item()}


def create_test_dataloader(batch_size: int, num_batches: int, device: str = 'cpu'):
    """Create test dataloader with larger images to test memory pressure."""
    class MockDataset:
        def __init__(self, size):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Smaller images for MPS memory testing
            image = torch.randn(3, 320, 320)
            # Mock targets in YOLO format
            targets = torch.tensor([[0, 1, 0.5, 0.5, 0.3, 0.3]])
            return image, targets
    
    from torch.utils.data import DataLoader
    dataset = MockDataset(num_batches * batch_size)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing for memory testing
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.cat([torch.cat([torch.tensor([[i]]), item[1]], dim=1) for i, item in enumerate(batch)])
        )
    )


def test_mps_memory_management():
    """Test MPS memory management during training."""
    logger.info("🧪 Testing MPS Memory Management")
    logger.info("=" * 50)
    
    # Detect available device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("✅ Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("✅ Using CUDA device")
    else:
        device = torch.device('cpu')
        logger.info("✅ Using CPU device")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Conservative MPS Settings',
            'config': {
                'mixed_precision': False,  # Should be auto-disabled on MPS
                'gradient_accumulation': True,
                'accumulation_steps': 2,  # Should be auto-reduced on MPS
                'compile_model': False,  # Should be auto-disabled on MPS
            },
            'batch_size': 2,  # Very small batch for MPS memory testing
            'num_batches': 10
        }
    ]
    
    for test_config in test_configs:
        logger.info(f"\n📊 Testing: {test_config['name']}")
        
        # Log initial memory
        initial_memory = get_memory_info()
        logger.info(f"   Initial memory: {initial_memory}")
        
        try:
            # Create components
            model = MockModel(device)
            progress_tracker = MockProgressTracker()
            loss_manager = MockLossManager()
            
            # Create training executor
            executor = TrainingExecutor(
                model=model,
                config=test_config['config'],
                progress_tracker=progress_tracker
            )
            
            # Create test dataloader
            train_loader = create_test_dataloader(
                test_config['batch_size'], 
                test_config['num_batches'], 
                device
            )
            
            # Create optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            # Test training loop with memory monitoring
            logger.info("   Running training with memory monitoring...")
            
            memory_snapshots = []
            start_time = time.time()
            
            result = executor.train_epoch(
                train_loader, optimizer, loss_manager, None, 0, 1, 1, 1
            )
            
            end_time = time.time()
            
            # Log final memory
            final_memory = get_memory_info()
            training_time = end_time - start_time
            
            logger.info(f"   ✅ Training completed successfully!")
            logger.info(f"   Time: {training_time:.2f}s")
            logger.info(f"   Loss: {result.get('train_loss', 0):.4f}")
            logger.info(f"   Effective batch size: {result.get('effective_batch_size', 'N/A')}")
            logger.info(f"   Final memory: {final_memory}")
            
            # Calculate memory delta
            memory_delta = final_memory['rss_mb'] - initial_memory['rss_mb']
            logger.info(f"   Memory delta: {memory_delta:.1f}MB")
            
            # Check if memory usage seems reasonable
            if memory_delta > 2000:  # More than 2GB increase
                logger.warning(f"   ⚠️ High memory usage increase: {memory_delta:.1f}MB")
            else:
                logger.info(f"   ✅ Memory usage looks reasonable: +{memory_delta:.1f}MB")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"   ❌ Out of memory error: {e}")
                logger.error("   MPS memory optimization may need adjustment")
            else:
                logger.error(f"   ❌ Runtime error: {e}")
        except Exception as e:
            logger.error(f"   ❌ Unexpected error: {e}")


def test_dataloader_optimizations():
    """Test DataLoader optimizations for MPS."""
    logger.info(f"\n🔄 Testing DataLoader MPS Optimizations")
    logger.info("=" * 40)
    
    try:
        # Test DataLoader factory
        factory = DataLoaderFactory()
        config = factory._get_optimal_dataloader_config()
        
        logger.info("✅ DataLoader optimization results:")
        logger.info(f"   Workers: {config['num_workers']}")
        logger.info(f"   Pin memory: {config['pin_memory']}")
        logger.info(f"   Prefetch factor: {config['prefetch_factor']}")
        logger.info(f"   Persistent workers: {config['persistent_workers']}")
        
        # Check MPS-specific optimizations
        has_mps = torch.backends.mps.is_available()
        if has_mps:
            logger.info("✅ MPS-specific optimizations:")
            logger.info(f"   Reduced workers: {config['num_workers'] <= 2}")
            logger.info(f"   No memory pinning: {not config['pin_memory']}")
            logger.info(f"   Conservative prefetch: {config['prefetch_factor'] <= 2}")
        
    except Exception as e:
        logger.error(f"❌ DataLoader optimization test failed: {e}")


def test_system_optimizations():
    """Test system-level MPS optimizations."""
    logger.info(f"\n🔧 Testing System MPS Optimizations")
    logger.info("=" * 40)
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        
        # Create a simple model to test system optimizations
        model = torch.nn.Linear(10, 5).to(device)
        config = {'mixed_precision': True, 'gradient_accumulation': False}
        progress_tracker = MockProgressTracker()
        
        try:
            executor = TrainingExecutor(model, config, progress_tracker)
            
            logger.info("✅ System optimizations applied:")
            logger.info(f"   Mixed precision disabled: {not executor.use_mixed_precision}")
            logger.info(f"   Accumulation steps reduced: {executor.accumulation_steps <= 2}")
            
            # Test MPS cache clearing
            try:
                torch.mps.empty_cache()
                logger.info("✅ MPS cache clearing works")
            except Exception as e:
                logger.warning(f"⚠️ MPS cache clearing failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ System optimization test failed: {e}")
    else:
        logger.info("⚠️ MPS not available, skipping MPS-specific tests")


if __name__ == '__main__':
    try:
        # Run MPS memory management tests
        test_mps_memory_management()
        
        # Run DataLoader optimization tests
        test_dataloader_optimizations()
        
        # Run system optimization tests
        test_system_optimizations()
        
        logger.info("\n🎉 All MPS memory management tests completed!")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        logger.error(traceback.format_exc())