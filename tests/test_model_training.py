"""
Test script for verifying model building and training with the updated API.

This script tests the end-to-end model building and training process
using the new API without any legacy YOLOv5 integration.
"""

import os
import sys
import unittest
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from smartcash.model.api import create_api
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class TestModelTraining(unittest.TestCase):
    """Test cases for model building and training."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test output directory
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)
        
        # Test configuration
        self.config = {
            "backbone": "efficientnet_b4",  # Using efficientnet as a test backbone
            "num_classes": 17,  # Fixed number of classes for SmartCash
            "batch_size": 2,    # Small batch size for testing
            "epochs": 1,        # Just one epoch for testing
            "learning_rate": 0.001,
            "checkpoint_dir": str(self.test_dir / "checkpoints"),
            "log_dir": str(self.test_dir / "logs"),
            "data": {
                "train": "data/train",  # Update with your test data path
                "val": "data/val"       # Update with your test data path
            }
        }
        
        # Create API instance
        self.api = create_api()
    
    def test_model_building(self):
        """Test that a model can be built with the new API."""
        logger.info("Testing model building...")
        
        # Build model
        result = self.api.build_model(
            backbone=self.config["backbone"],
            num_classes=self.config["num_classes"]
        )
        
        # Verify build was successful
        self.assertTrue(result["success"], f"Model building failed: {result.get('error', 'Unknown error')}")
        self.assertIsNotNone(self.api.model, "Model was not assigned to API instance")
        
        logger.info("✅ Model building test passed")
    
    def test_training_pipeline(self):
        """Test the training pipeline with the new API."""
        logger.info("Testing training pipeline...")
        
        # First build the model
        build_result = self.api.build_model(
            backbone=self.config["backbone"],
            num_classes=self.config["num_classes"]
        )
        self.assertTrue(build_result["success"], "Model building failed")
        
        # Set up a simple progress callback
        def progress_callback(progress: float, status: str, **kwargs):
            logger.info(f"Training progress: {progress:.1f}% - {status}")
            if "metrics" in kwargs:
                logger.info(f"Metrics: {kwargs['metrics']}")
        
        # Run training
        training_result = self.api.train_model(
            train_data=self.config["data"]["train"],
            val_data=self.config["data"]["val"],
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            learning_rate=self.config["learning_rate"],
            checkpoint_dir=self.config["checkpoint_dir"],
            log_dir=self.config["log_dir"],
            progress_callback=progress_callback
        )
        
        # Verify training was successful
        self.assertTrue(training_result["success"], 
                       f"Training failed: {training_result.get('error', 'Unknown error')}")
        
        logger.info("✅ Training pipeline test passed")
    
    def tearDown(self):
        """Clean up test environment."""
        # Clean up test directory
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

if __name__ == "__main__":
    unittest.main()
