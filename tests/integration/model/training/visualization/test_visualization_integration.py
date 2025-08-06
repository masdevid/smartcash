"""
Integration tests for the visualization module.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from smartcash.model.analysis.visualization.visualization_manager import VisualizationManager


class TestVisualizationIntegration:
    """Integration tests for the visualization module."""
    
    def test_visualization_manager_integration(self, tmp_path):
        """Test the visualization manager as the main entry point."""
        # Configuration
        num_classes_per_layer = {
            'layer_1': 7,
            'layer_2': 7,
            'layer_3': 3
        }
        
        # Create visualization manager
        viz_manager = VisualizationManager(
            config={'visualization': {'charts': {'figure_size': [12, 8], 'dpi': 150, 'style': 'seaborn-v0_8', 'color_palette': 'Set2'}}},
            output_dir=str(tmp_path / 'viz_manager')
        )
        
        # Mock analysis results structure that VisualizationManager expects
        mock_analysis_results = {
            'currency_analysis': {
                'aggregated_metrics': {
                    'strategy_distribution': {'strategy_a': 10, 'strategy_b': 20},
                    'denomination_distribution': {'denom_a': 5, 'denom_b': 15}
                },
                'batch_summary': {
                    'successful_analysis': 30, 'total_images': 30, 'images_with_currency': 25,
                    'success_rate': 0.8, 'detection_rate': 0.7
                }
            },
            'layer_analysis': {
                'aggregated_layer_metrics': {
                    'layer_1': {'avg_precision': 0.8, 'avg_recall': 0.7, 'avg_f1_score': 0.75, 'avg_confidence': 0.9},
                    'layer_2': {'avg_precision': 0.7, 'avg_recall': 0.6, 'avg_f1_score': 0.65, 'avg_confidence': 0.8}
                },
                'batch_insights': {
                    'layer_activity_rates': {'layer_1': 0.9, 'layer_2': 0.7}
                },
                'layer_consistency': {
                    'layer_1': {'consistency_score': 0.95},
                    'layer_2': {'consistency_score': 0.85}
                }
            },
            'class_analysis': {
                'per_class_metrics': {
                    'class_0': {'precision': 0.9, 'recall': 0.8, 'f1_score': 0.85, 'ap': 0.88},
                    'class_1': {'precision': 0.8, 'recall': 0.7, 'f1_score': 0.75, 'ap': 0.78}
                },
                'confusion_matrix': {
                    'matrix': [[10, 1], [2, 15]],
                    'class_names': ['class_0', 'class_1']
                },
                'class_distribution': {
                    'class_0': 100, 'class_1': 50
                }
            },
            'comparative_analysis': {
                'backbone_comparison': {
                    'backbone_a': {'accuracy': 0.9, 'inference_time': 0.01},
                    'backbone_b': {'accuracy': 0.8, 'inference_time': 0.02}
                },
                'scenario_comparison': {
                    'scenario_x': {'map': 0.85},
                    'scenario_y': {'map': 0.75}
                },
                'efficiency_analysis': {
                    'backbone_a': {'accuracy': 0.9, 'inference_time': 0.01},
                    'backbone_b': {'accuracy': 0.8, 'inference_time': 0.02}
                }
            }
        }

        from smartcash.model.training.visualization import (
    BaseMetricsTracker,
    ChartGenerator,
    MetricsAnalyzer,
    create_visualization_manager
)


class TestVisualizationIntegration:
    """Integration tests for the visualization module."""
    
    @pytest.fixture
    def setup_components(self, tmp_path):
        """Set up components for integration testing."""
        # Configuration
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
        
        # Create components
        save_dir = tmp_path / "visualization"
        save_dir.mkdir()
        
        tracker = BaseMetricsTracker(
            num_classes_per_layer=num_classes_per_layer,
            class_names=class_names,
            save_dir=str(save_dir),
            verbose=False
        )
        
        generator = ChartGenerator(
            metrics_tracker=tracker,
            save_dir=str(save_dir),
            verbose=False
        )
        
        analyzer = MetricsAnalyzer(
            metrics_tracker=tracker,
            verbose=False
        )
        
        return {
            'tracker': tracker,
            'generator': generator,
            'analyzer': analyzer,
            'save_dir': save_dir,
            'num_classes_per_layer': num_classes_per_layer,
            'class_names': class_names
        }
    
    def test_end_to_end_training_visualization(self, setup_components):
        """Test the complete training visualization workflow."""
        # Setup
        tracker = setup_components['tracker']
        generator = setup_components['generator']
        analyzer = setup_components['analyzer']
        save_dir = setup_components['save_dir']
        num_classes = setup_components['num_classes_per_layer']
        
        # Simulate a training run with 5 epochs
        np.random.seed(42)
        num_epochs = 5
        
        for epoch in range(num_epochs):
            # Create sample metrics
            metrics = {
                'train_loss': 1.0 - (epoch * 0.15) + (np.random.rand() * 0.05),
                'val_loss': 1.2 - (epoch * 0.12) + (np.random.rand() * 0.05),
                'learning_rate': 0.001 * (0.9 ** epoch),
                'phase': 'training' if epoch % 2 == 0 else 'validation',
                'phase_num': 1 if epoch < 3 else 2  # Phase transition at epoch 3
            }
            
            # Add layer-specific metrics
            for layer in num_classes:
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    metrics[f'{layer}_{metric}'] = 0.7 + (epoch * 0.05) + (np.random.rand() * 0.05)
            
            # Generate sample predictions and ground truth (every other epoch)
            predictions = {}
            ground_truth = {}
            if epoch % 2 == 0:
                for layer, n_classes in num_classes.items():
                    batch_size = 10
                    predictions[layer] = np.random.rand(batch_size, n_classes)
                    ground_truth[layer] = np.random.randint(0, n_classes, size=(batch_size,))
            
            # Update metrics
            tracker.update_metrics(
                epoch=epoch,
                phase=metrics['phase'],
                metrics=metrics,
                predictions=predictions if predictions else None,
                ground_truth=ground_truth if ground_truth else None,
                phase_num=metrics['phase_num']
            )
        
        # Generate visualizations
        with patch('matplotlib.pyplot.figure'):  # Mock matplotlib
            with patch('seaborn.heatmap'):      # Mock seaborn
                charts = generator.generate_all_charts(session_id='test_integration')
        
        # Check that charts were generated
        assert len(charts) >= 6  # At least 6 charts (curves, 3x confusion, metrics, phase)
        
        # Analyze the training
        analysis = analyzer.analyze_training_progress()
        
        # Check analysis results
        assert analysis['final_metrics']['epoch'] == num_epochs - 1  # 0-based
        assert len(analysis['phase_transitions']) > 0
        assert 'layer_1' in analysis['layer_performance']
        
        # Save analysis report
        report_path = analyzer.save_analysis_report(
            session_dir=save_dir / 'test_integration',
            session_id='test_integration'
        )
        
        assert report_path is not None
        assert report_path.exists()
    
    def test_visualization_manager_integration(self, tmp_path):
        """Test the visualization manager as the main entry point."""
        # Configuration
        num_classes_per_layer = {
            'layer_1': 7,
            'layer_2': 7,
            'layer_3': 3
        }
        
        # Create visualization manager
        viz_manager = create_visualization_manager(
            num_classes_per_layer=num_classes_per_layer,
            save_dir=str(tmp_path / 'viz_manager'),
            verbose=False
        )
        
        # Simulate a training run
        num_epochs = 3
        
        for epoch in range(num_epochs):
            # Sample metrics
            metrics = {
                'train_loss': 1.0 - (epoch * 0.2),
                'val_loss': 1.1 - (epoch * 0.15),
                'learning_rate': 0.001 * (0.9 ** epoch),
            }
            
            # Add layer metrics
            for layer in num_classes_per_layer:
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    metrics[f'{layer}_{metric}'] = 0.7 + (epoch * 0.1)
            
            # Update metrics
            viz_manager.update_metrics(
                epoch=epoch,
                phase='training' if epoch % 2 == 0 else 'validation',
                metrics=metrics,
                phase_num=1 if epoch < 2 else 2
            )
        
        # Generate visualizations
        with patch('matplotlib.pyplot.figure'):
            with patch('seaborn.heatmap'):
                charts = viz_manager.generate_visualizations(session_id='test_manager')
        
        # Check results
        assert len(charts) >= 6  # At least 6 charts
        
        # Cleanup
        viz_manager.cleanup()
    
    def test_context_manager_usage(self, tmp_path):
        """Test using VisualizationManager as a context manager."""
        num_classes_per_layer = {'layer_1': 3, 'layer_2': 2}
        
        with create_visualization_manager(
            num_classes_per_layer=num_classes_per_layer,
            save_dir=str(tmp_path / 'context_test')
        ) as viz_manager:
            # Update metrics
            viz_manager.update_metrics(
                epoch=0,
                phase='training',
                metrics={
                    'train_loss': 1.0,
                    'val_loss': 1.1,
                    'learning_rate': 0.001,
                    'layer_1_accuracy': 0.8,
                    'layer_2_accuracy': 0.85
                }
            )
            
            # Generate visualizations
            with patch('matplotlib.pyplot.figure'):
                with patch('seaborn.heatmap'):
                    charts = viz_manager.generate_visualizations(session_id='test_context')
            
            assert charts is not None
        
        # Manager should be cleaned up after context exits
        assert viz_manager.metrics_tracker._shutdown is True
        
        
