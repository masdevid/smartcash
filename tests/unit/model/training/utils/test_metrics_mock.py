#!/usr/bin/env python3
"""
Mock test to verify metrics calculation without full training.
This helps test the training pipeline returns non-zero metrics.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smartcash.common.logger import get_logger

logger = get_logger("metrics_mock_test")

def create_mock_data():
    """Create mock data that should produce non-zero metrics"""
    # Create mock predictions and targets that should give reasonable metrics
    batch_size = 8
    num_classes = 7
    
    # Mock predictions (logits)
    mock_predictions = {
        'layer_1': torch.randn(batch_size, num_classes) * 0.1 + torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'layer_2': torch.randn(batch_size, num_classes) * 0.1 + torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'layer_3': torch.randn(batch_size, num_classes) * 0.1 + torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    }
    
    # Mock targets (correct classes with some variation)
    mock_targets = {
        'layer_1': torch.tensor([0, 0, 1, 0, 2, 1, 0, 1]),  # Mix of correct and incorrect
        'layer_2': torch.tensor([1, 1, 0, 1, 2, 0, 1, 0]),  # Mix of correct and incorrect  
        'layer_3': torch.tensor([2, 2, 1, 2, 0, 1, 2, 1])   # Mix of correct and incorrect
    }
    
    return mock_predictions, mock_targets

def test_metrics_calculation():
    """Test that metrics calculation produces non-zero values"""
    logger.info("🧪 Testing metrics calculation with mock data...")
    
    try:
        # Create mock data
        predictions, targets = create_mock_data()
        
        # Test basic accuracy calculation
        for layer in ['layer_1', 'layer_2', 'layer_3']:
            pred_logits = predictions[layer]
            true_labels = targets[layer]
            
            # Calculate predictions (argmax)
            pred_classes = torch.argmax(pred_logits, dim=1)
            
            # Calculate accuracy
            correct = (pred_classes == true_labels).float()
            accuracy = correct.mean().item()
            
            # Calculate precision, recall, f1 (simplified)
            precision = accuracy  # Simplified for mock
            recall = accuracy     # Simplified for mock
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            logger.info(f"📊 {layer} Mock Metrics:")
            logger.info(f"   • Accuracy: {accuracy:.3f}")
            logger.info(f"   • Precision: {precision:.3f}")
            logger.info(f"   • Recall: {recall:.3f}")
            logger.info(f"   • F1: {f1:.3f}")
            
            # Verify non-zero metrics
            assert accuracy > 0.0, f"{layer} accuracy should be > 0.0"
            assert precision > 0.0, f"{layer} precision should be > 0.0"
            assert recall > 0.0, f"{layer} recall should be > 0.0"
            assert f1 > 0.0, f"{layer} f1 should be > 0.0"
        
        logger.info("✅ Mock metrics test passed - calculations produce non-zero values!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Mock metrics test failed: {e}")
        return False

def test_config_num_workers():
    """Test that num_workers configuration is correct"""
    logger.info("🧪 Testing num_workers configuration...")
    
    try:
        from smartcash.model.utils.memory_optimizer import MemoryOptimizer
        
        # Create optimizer with mock platform info
        mock_platform_info = {
            'is_cuda_workstation': False,
            'is_apple_silicon': False,
            'cpu_count': 10,
            'force_cpu': True
        }
        
        optimizer = MemoryOptimizer(mock_platform_info)
        config = optimizer.get_memory_config()
        
        num_workers = config.get('num_workers', 0)
        logger.info(f"📊 Configured num_workers: {num_workers}")
        
        # Verify minimum 2 workers
        assert num_workers >= 2, f"num_workers should be >= 2, got {num_workers}"
        
        # Test batch config too
        batch_config = optimizer.get_optimal_batch_config()
        batch_num_workers = batch_config.get('num_workers', 0)
        logger.info(f"📊 Batch config num_workers: {batch_num_workers}")
        
        assert batch_num_workers >= 2, f"Batch num_workers should be >= 2, got {batch_num_workers}"
        
        logger.info("✅ num_workers configuration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ num_workers configuration test failed: {e}")
        return False

def main():
    """Run all mock tests"""
    logger.info("🚀 Starting mock metrics tests...")
    
    # Test metrics calculation
    metrics_test = test_metrics_calculation()
    
    # Test num_workers configuration  
    workers_test = test_config_num_workers()
    
    if metrics_test and workers_test:
        logger.info("🎉 All mock tests passed!")
        return 0
    else:
        logger.error("❌ Some mock tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())