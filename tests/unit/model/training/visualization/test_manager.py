"""
Unit tests for the visualization manager.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, ANY, mock_open
import numpy as np
from pathlib import Path

# Mock all visualization-related imports
class MockFigure:
    def __init__(self, *args, **kwargs):
        pass
    
    def savefig(self, *args, **kwargs):
        pass

class MockAxes:
    def set_title(self, *args, **kwargs):
        pass
    
    def set_xlabel(self, *args, **kwargs):
        pass
    
    def set_ylabel(self, *args, **kwargs):
        pass
    
    def legend(self, *args, **kwargs):
        pass
    
    def grid(self, *args, **kwargs):
        pass

class MockMatplotlib:
    def __init__(self):
        self.figure = MockFigure
        self.Figure = MockFigure
        self.Axes = MockAxes
        self.pyplot = MagicMock()
        self.pyplot.figure.return_value = MockFigure()
        self.pyplot.subplots.return_value = (MockFigure(), [MockAxes()])

# Set up mock modules
sys.modules['matplotlib'] = MockMatplotlib()
sys.modules['matplotlib.figure'] = MockFigure
sys.modules['matplotlib.axes'] = MagicMock()
sys.modules['matplotlib.axes._axes'] = MagicMock()
sys.modules['matplotlib.axes._axes.Axes'] = MockAxes
sys.modules['matplotlib.pyplot'] = MockMatplotlib().pyplot
sys.modules['seaborn'] = MagicMock()
sys.modules['seaborn.heatmap'] = MagicMock()

# Mock our own modules
sys.modules['smartcash.common.logger'] = MagicMock()
sys.modules['smartcash.model.analysis'] = MagicMock()

# Now import the module to test
with patch('matplotlib.pyplot', MockMatplotlib().pyplot), \
     patch('matplotlib.figure', MockFigure), \
     patch('seaborn.heatmap', MagicMock()):
    from smartcash.model.training.visualization.manager import VisualizationManager

class TestVisualizationManager(unittest.TestCase):
    """Test cases for VisualizationManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_classes = {'layer1': 2, 'layer2': 5}
        self.class_names = {
            'layer1': ['class0', 'class1'],
            'layer2': [f'class{i}' for i in range(5)]
        }
        self.save_dir = "/tmp/test_visualization"
        
        # Create the visualization manager
        self.viz = VisualizationManager(
            num_classes_per_layer=self.num_classes,
            class_names=self.class_names,
            save_dir=self.save_dir,
            verbose=True
        )
    
    def test_initialization(self):
        """Test that the visualization manager initializes correctly."""
        self.assertEqual(self.viz.num_classes_per_layer, self.num_classes)
        self.assertEqual(self.viz.class_names, self.class_names)
        self.assertEqual(str(self.viz.save_dir), self.save_dir)
        self.assertTrue(self.viz.verbose)
        self.assertEqual(len(self.viz.metrics_history), 0)
        self.assertEqual(len(self.viz.confusion_matrices), 0)
        self.assertEqual(len(self.viz.learning_rates), 0)
    
    def test_update_metrics(self):
        """Test updating metrics."""
        # Test with minimal metrics
        self.viz.update_metrics(
            epoch=1,
            metrics={'loss': 0.5, 'accuracy': 0.8},
            phase='train',
            learning_rate=0.001
        )
        
        self.assertEqual(len(self.viz.metrics_history), 1)
        self.assertEqual(self.viz.metrics_history[0]['epoch'], 1)
        self.assertEqual(self.viz.metrics_history[0]['phase'], 'train')
        self.assertEqual(self.viz.metrics_history[0]['loss'], 0.5)
        self.assertEqual(self.viz.learning_rates, [0.001])
        
        # Test with confusion matrices
        cm = {
            'layer1': np.array([[10, 2], [1, 12]]),
            'layer2': np.eye(5) * 5
        }
        self.viz.update_metrics(
            epoch=1,
            metrics={'val_loss': 0.4, 'val_accuracy': 0.85},
            phase='val',
            confusion_matrices=cm
        )
        
        self.assertIn('layer1', self.viz.confusion_matrices)
        self.assertIn('layer2', self.viz.confusion_matrices)
        self.assertEqual(len(self.viz.confusion_matrices['layer1']), 1)

if __name__ == '__main__':
    unittest.main()
