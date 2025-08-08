#!/usr/bin/env python3
"""
Test suite for the new visualization package.

Tests functionality of the visualization package with different configurations
and verifies chart generation capabilities.
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from smartcash.model.training.visualization import VisualizationManager


class TestVisualizationPackage:
    """Test suite for the visualization package functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_classes_per_layer = {
            'banknote': 2,  # Binary classification for banknote detection
            'denomination': 7,  # 7 denominations
            'features': 3  # 3 feature types
        }
        self.test_class_names = {
            'banknote': ['No Banknote', 'Banknote'],
            'denomination': [f'Class {i}' for i in range(7)],
            'features': ['Feature A', 'Feature B', 'Feature C']
        }
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_visualization_manager_creation(self):
        """Test creating a visualization manager instance."""
        # Test with minimal parameters
        viz_manager = VisualizationManager(
            num_classes_per_layer=self.test_classes_per_layer,
            save_dir=self.temp_dir
        )
        
        assert viz_manager is not None
        assert viz_manager.save_dir == Path(self.temp_dir)
        assert not viz_manager.verbose
        
        # Test with all parameters
        viz_manager = VisualizationManager(
            num_classes_per_layer=self.test_classes_per_layer,
            class_names=self.test_class_names,
            save_dir=self.temp_dir,
            verbose=True
        )
        
        assert viz_manager is not None
        assert viz_manager.verbose
    
    def test_update_metrics(self):
        """Test updating metrics in the visualization manager."""
        viz_manager = VisualizationManager(
            num_classes_per_layer=self.test_classes_per_layer,
            class_names=self.test_class_names,
            save_dir=self.temp_dir,
            verbose=True
        )
        
        # Test updating metrics for training phase
        metrics = {
            'loss': 0.5,
            'accuracy': 0.9,
            'precision': 0.88,
            'recall': 0.92,
            'f1': 0.90
        }
        
        viz_manager.update_metrics(
            epoch=1,
            metrics=metrics,
            phase='train',
            learning_rate=0.001
        )
        
        # Test updating metrics for validation phase
        viz_manager.update_metrics(
            epoch=1,
            metrics=metrics,
            phase='val',
            learning_rate=0.001
        )
        
        # Verify metrics were recorded
        assert len(viz_manager.metrics_history) == 2
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_generate_charts(self, mock_savefig, mock_show):
        """Test generating charts."""
        viz_manager = VisualizationManager(
            num_classes_per_layer=self.test_classes_per_layer,
            class_names=self.test_class_names,
            save_dir=self.temp_dir,
            verbose=True
        )
        
        # Add some test data
        for epoch in range(1, 6):
            metrics = {
                'loss': 1.0 / epoch,
                'accuracy': 0.8 + (epoch * 0.03),
                'precision': 0.78 + (epoch * 0.02),
                'recall': 0.82 + (epoch * 0.02),
                'f1': 0.80 + (epoch * 0.025)
            }
            viz_manager.update_metrics(
                epoch=epoch,
                metrics=metrics,
                phase='train',
                learning_rate=0.001 / (epoch ** 0.5)
            )
            
            # Add validation metrics
            val_metrics = {k: v * 0.95 for k, v in metrics.items()}
            viz_manager.update_metrics(
                epoch=epoch,
                metrics=val_metrics,
                phase='val',
                learning_rate=0.001 / (epoch ** 0.5)
            )
        
        # Generate all charts
        results = viz_manager.generate_all_charts()
        
        # Verify charts were generated
        assert results is not None
        assert 'training_curves' in results
        assert 'learning_rate' in results
        
        # Test saving metrics summary
        summary_path = viz_manager.save_metrics_summary()
        assert summary_path.exists()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_confusion_matrices(self, mock_savefig, mock_show):
        """Test generating confusion matrices."""
        viz_manager = VisualizationManager(
            num_classes_per_layer=self.test_classes_per_layer,
            class_names=self.test_class_names,
            save_dir=self.temp_dir,
            verbose=True
        )
        
        # Create a confusion matrix for each layer
        confusion_matrices = {}
        for layer, num_classes in self.test_classes_per_layer.items():
            # Create a random confusion matrix
            cm = np.random.randint(0, 100, size=(num_classes, num_classes))
            np.fill_diagonal(cm, np.random.randint(100, 200, size=num_classes))
            confusion_matrices[layer] = cm
        
        # Update metrics with confusion matrices
        viz_manager.update_metrics(
            epoch=1,
            metrics={'accuracy': 0.9, 'loss': 0.3},
            phase='val',
            learning_rate=0.001,
            confusion_matrices=confusion_matrices
        )
        
        # Generate confusion matrices
        results = viz_manager.generate_confusion_matrices()
        
        # Verify confusion matrices were generated
        assert results is not None
        for layer in self.test_classes_per_layer.keys():
            assert layer in results
    
    def test_cleanup(self):
        """Test cleaning up resources."""
        viz_manager = VisualizationManager(
            num_classes_per_layer=self.test_classes_per_layer,
            save_dir=self.temp_dir
        )
        
        # No exceptions should be raised
        viz_manager.cleanup()


if __name__ == "__main__":
    # Run tests with increased verbosity
    pytest.main(["-v", __file__])
