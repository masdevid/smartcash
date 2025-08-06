"""
Test configuration and fixtures for visualization module tests.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    with patch('smartcash.common.logger.get_logger') as mock_get_logger:
        logger = MagicMock()
        mock_get_logger.return_value = logger
        yield logger


@pytest.fixture
def sample_metrics_tracker(mock_logger):
    """Create a sample metrics tracker for testing."""
    from smartcash.model.training.visualization.base_metrics_tracker import BaseMetricsTracker
    
    num_classes_per_layer = {
        'layer_1': 7,  # Banknote detection
        'layer_2': 7,  # Denomination features
        'layer_3': 3   # Common features
    }
    
    class_names = {
        'layer_1': ['001', '002', '005', '010', '020', '050', '100'],
        'layer_2': [f'l2_{i:03d}' for i in [1, 2, 5, 10, 20, 50, 100]],
        'layer_3': ['sign', 'text', 'thread']
    }
    
    return BaseMetricsTracker(
        num_classes_per_layer=num_classes_per_layer,
        class_names=class_names,
        save_dir=str(Path('/tmp/test_visualization')),
        verbose=True
    )


@pytest.fixture
def sample_metrics():
    """Sample metrics dictionary for testing."""
    return {
        'train_loss': 0.5,
        'val_loss': 0.6,
        'learning_rate': 0.001,
        'layer_1_accuracy': 0.9,
        'layer_2_accuracy': 0.85,
        'layer_3_accuracy': 0.95,
        'layer_1_precision': 0.88,
        'layer_2_precision': 0.83,
        'layer_3_precision': 0.93,
        'layer_1_recall': 0.89,
        'layer_2_recall': 0.84,
        'layer_3_recall': 0.94,
        'layer_1_f1': 0.885,
        'layer_2_f1': 0.835,
        'layer_3_f1': 0.935
    }


@pytest.fixture
def sample_predictions():
    """Sample predictions for testing."""
    np.random.seed(42)
    return {
        'layer_1': np.random.rand(10, 7),  # 10 samples, 7 classes
        'layer_2': np.random.rand(10, 7),
        'layer_3': np.random.rand(10, 3)
    }


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth for testing."""
    np.random.seed(42)
    return {
        'layer_1': np.random.randint(0, 7, size=(10,)),
        'layer_2': np.random.randint(0, 7, size=(10,)),
        'layer_3': np.random.randint(0, 3, size=(10,))
    }
