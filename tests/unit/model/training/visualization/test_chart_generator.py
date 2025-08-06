"""
Unit tests for chart_generator.py

Tests for the ChartGenerator class.
"""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY
import numpy as np
import pytest

# Import the ChartGenerator class
from smartcash.model.training.visualization.chart_generator import ChartGenerator

class TestChartGenerator:
    """Test cases for ChartGenerator."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock metrics tracker
        self.metrics_tracker = MagicMock()
        
        # Set up default attributes
        self.metrics_tracker.train_losses = [0.8, 0.6, 0.5, 0.4]
        self.metrics_tracker.val_losses = [0.9, 0.7, 0.6, 0.5]
        self.metrics_tracker.learning_rates = [0.001, 0.0008, 0.0006, 0.0004]
        self.metrics_tracker.confusion_matrices = {}
        self.metrics_tracker.phase_transitions = []
        self.metrics_tracker.num_classes_per_layer = {}
        self.metrics_tracker.class_names = {}
        self.metrics_tracker.layer_metrics = {}
        
        # Create the generator instance
        self.generator = ChartGenerator(
            metrics_tracker=self.metrics_tracker,
            save_dir='/tmp/test_visualization',
            verbose=False
        )
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_generate_training_curves(self, mock_tight_layout, mock_savefig, mock_close, mock_subplots):
        """Test generation of training curves."""
        # Setup test data
        self.metrics_tracker.train_losses = [0.8, 0.6, 0.5, 0.4]
        self.metrics_tracker.val_losses = [0.9, 0.7, 0.6, 0.5]
        self.metrics_tracker.learning_rates = [0.001, 0.0008, 0.0006, 0.0004]
        
        # Setup mocks
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])
        
        # Mock the savefig to return the expected path
        session_dir = Path('/tmp/test_session')
        expected_path = session_dir / 'training_curves.png'
        mock_savefig.return_value = expected_path
        
        # Call the method
        result = self.generator.generate_training_curves(session_dir)
        
        # Verify the result (method returns the save path)
        assert str(result) == str(expected_path)
        
        # Verify subplots were created with correct parameters
        mock_subplots.assert_called_once_with(2, 1, figsize=(12, 12), 
                                            gridspec_kw={'height_ratios': [2, 1]})
        
        # Verify plots were created
        assert mock_ax1.plot.call_count == 2  # train and val loss
        assert mock_ax2.plot.call_count == 1  # learning rate
        
        # Verify figure was saved
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert str(args[0]) == str(expected_path)
        assert kwargs == {'dpi': 300, 'bbox_inches': 'tight'}
        
        # Verify figure was properly closed
        mock_tight_layout.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('seaborn.heatmap')
    def test_generate_confusion_matrix(self, mock_heatmap, mock_tight_layout, 
                                     mock_savefig, mock_close, mock_figure):
        """Test generation of confusion matrix."""
        # Setup test data
        layer = 'layer_1'
        self.metrics_tracker.confusion_matrices = {layer: [{
            'epoch': 1,
            'matrix': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # Identity matrix for testing
            'accuracy': 1.0
        }]}
        self.metrics_tracker.num_classes_per_layer = {layer: 3}
        self.metrics_tracker.class_names = {layer: ['Class1', 'Class2', 'Class3']}
        
        # Setup mocks
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        
        # Mock the savefig to return the expected path
        session_dir = Path('/tmp/test_session')
        expected_path = session_dir / f'confusion_matrix_{layer}.png'
        mock_savefig.return_value = expected_path
        
        # Call the method
        result = self.generator.generate_confusion_matrix(layer, session_dir)
        
        # Verify the result (method returns the save path)
        assert str(result) == str(expected_path)
        
        # Verify seaborn heatmap was called with correct parameters
        mock_heatmap.assert_called_once()
        args, kwargs = mock_heatmap.call_args
        
        # Use almost equal for floating point comparison
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_almost_equal(args[0], expected)
        assert kwargs['annot'] is True
        assert kwargs['fmt'] == '.2f'
        assert kwargs['cmap'] == 'Blues'
        
        # Verify figure was saved
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert str(args[0]) == str(expected_path)
        assert kwargs == {'dpi': 300, 'bbox_inches': 'tight'}
        
        # Verify figure was properly closed
        mock_tight_layout.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    @patch.object(ChartGenerator, '_add_phase_transition_markers')
    def test_generate_phase_analysis(self, mock_add_markers, mock_tight_layout, 
                                   mock_savefig, mock_close, mock_subplots):
        """Test generation of phase analysis chart."""
        # Setup test data
        self.metrics_tracker.train_losses = [0.8, 0.6, 0.5, 0.4, 0.35, 0.3]
        self.metrics_tracker.val_losses = [0.9, 0.7, 0.6, 0.5, 0.45, 0.4]
        
        # Add phase_transitions to the metrics_tracker as a list of dictionaries
        phase_transitions = [
            {'epoch': 2, 'from_phase': 'phase_1', 'to_phase': 'phase_2'}
        ]
        # Set the phase_transitions attribute on the metrics_tracker
        setattr(self.metrics_tracker, 'phase_transitions', phase_transitions)
        
        # Setup mocks
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Mock the savefig to return the expected path
        session_dir = Path('/tmp/test_session')
        expected_path = session_dir / 'phase_analysis.png'
        mock_savefig.return_value = expected_path
        
        # Call the method
        phase_num = 2
        result = self.generator.generate_phase_analysis(session_dir, phase_num)
        
        # Verify the result (method returns the save path)
        assert result is not None  # Should not be None when phase_transitions exist
        assert str(result) == str(expected_path)
        
        # Verify subplots were created with correct parameters
        mock_subplots.assert_called_once_with(figsize=(12, 6))
        
        # Verify plots were created - at least 2 calls (train and val loss)
        assert mock_ax.plot.call_count >= 2
        
        # Verify phase transition markers were added with detailed=True
        mock_add_markers.assert_called_once_with(mock_ax, range(1, 7), detailed=True)
        
        # Verify figure was saved
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert str(args[0]) == str(expected_path)
        assert kwargs == {'dpi': 300, 'bbox_inches': 'tight'}
        
        # Verify figure was properly closed
        mock_tight_layout.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('pathlib.Path.mkdir')
    def test_generate_all_charts(self, mock_mkdir):
        """Test generation of all charts."""
        # Setup test data
        self.metrics_tracker.num_classes_per_layer = {
            'layer_1': 3,
            'layer_2': 2
        }
        
        # Define the expected save directory
        save_dir = Path('/tmp/test_visualization/test_session')
        
        # Patch the chart generation methods to return expected paths
        with patch.object(self.generator, 'generate_training_curves') as mock_curves, \
             patch.object(self.generator, 'generate_confusion_matrix') as mock_cm, \
             patch.object(self.generator, 'generate_layer_metrics_comparison') as mock_metrics, \
             patch.object(self.generator, 'generate_phase_analysis') as mock_phase:
            
            # Setup return values for each method
            mock_curves.return_value = save_dir / 'training_curves.png'
            mock_cm.side_effect = [
                save_dir / 'confusion_matrix_layer_1.png',
                save_dir / 'confusion_matrix_layer_2.png'
            ]
            mock_metrics.return_value = save_dir / 'layer_metrics_comparison.png'
            mock_phase.return_value = save_dir / 'phase_analysis.png'
            
            # Call the method
            session_id = 'test_session'
            phase_num = 2
            results = self.generator.generate_all_charts(session_id, phase_num)
            
            # Verify the results - convert all paths to strings for comparison
            expected_results = {
                'training_curves': str(save_dir / 'training_curves.png'),
                'confusion_matrix_layer_1': str(save_dir / 'confusion_matrix_layer_1.png'),
                'confusion_matrix_layer_2': str(save_dir / 'confusion_matrix_layer_2.png'),
                'layer_metrics': str(save_dir / 'layer_metrics_comparison.png'),
                'phase_analysis': str(save_dir / 'phase_analysis.png')
            }
            
            # Convert actual results to strings for comparison
            actual_results = {k: str(v) if v is not None else None for k, v in results.items()}
            assert actual_results == expected_results
            
            # Verify all chart generation methods were called
            mock_curves.assert_called_once_with(save_dir)
            assert mock_cm.call_count == 2  # Called for each layer
            assert mock_cm.call_args_list[0][0] == ('layer_1', save_dir)
            assert mock_cm.call_args_list[1][0] == ('layer_2', save_dir)
            mock_metrics.assert_called_once_with(save_dir)
            mock_phase.assert_called_once_with(save_dir, phase_num)
            
            # Verify directory was created with parents=True
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_save_plot(self, mock_tight_layout, mock_savefig, mock_close, mock_subplots):
        """Test saving a plot to file through generate_training_curves."""
        # Setup test data
        self.metrics_tracker.train_losses = [0.8, 0.6, 0.5, 0.4]
        self.metrics_tracker.val_losses = [0.9, 0.7, 0.6, 0.5]
        self.metrics_tracker.learning_rates = [0.001, 0.0008, 0.0006, 0.0004]
        
        # Setup mocks
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])
        
        # Mock the savefig to return the expected path
        session_dir = Path('/tmp/test_session')
        expected_path = session_dir / 'training_curves.png'
        mock_savefig.return_value = expected_path
        
        # Call the method
        result = self.generator.generate_training_curves(session_dir)
        
        # Verify the result (method returns the save path)
        assert str(result) == str(expected_path)
        
        # Verify figure was saved with the correct parameters
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert str(args[0]) == str(expected_path)
        assert kwargs == {'dpi': 300, 'bbox_inches': 'tight'}
        
        # Verify figure was properly closed
        mock_tight_layout.assert_called_once()
        mock_close.assert_called_once()
