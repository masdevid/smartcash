#!/usr/bin/env python3
"""
Comprehensive tests for VisualizationManager module.

Tests plot generation, chart creation, visualization management,
and comprehensive dashboard functionality.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import json

# Import the modules to test
import sys
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Global mock setup for all visualization dependencies
mock_matplotlib = MagicMock()
mock_pyplot = MagicMock()
mock_seaborn = MagicMock()
mock_pandas = MagicMock()
mock_scipy = MagicMock()
mock_scipy_spatial = MagicMock()

# Configure mock return values to prevent AttributeError
mock_pyplot.subplots.return_value = (MagicMock(), MagicMock())
mock_pyplot.figure.return_value = MagicMock()
mock_pyplot.Circle.return_value = MagicMock()
mock_pyplot.style = MagicMock()
mock_pyplot.style.available = ['default', 'seaborn-v0_8']
mock_pyplot.style.use = MagicMock()

mock_seaborn.color_palette.return_value = ['#1f77b4', '#ff7f0e', '#2ca02c']
mock_seaborn.set_palette = MagicMock()
mock_seaborn.heatmap.return_value = MagicMock()
mock_seaborn.barplot.return_value = MagicMock()

mock_pandas.DataFrame.return_value = MagicMock()

# Mock the ConvexHull
mock_scipy_spatial.ConvexHull.return_value = MagicMock()
mock_scipy_spatial.ConvexHull.return_value.vertices = [0, 1, 2]

# Install mocks in sys.modules BEFORE importing
sys.modules['matplotlib'] = mock_matplotlib
sys.modules['matplotlib.pyplot'] = mock_pyplot
sys.modules['seaborn'] = mock_seaborn
sys.modules['pandas'] = mock_pandas
sys.modules['scipy'] = mock_scipy
sys.modules['scipy.spatial'] = mock_scipy_spatial

# Now import after mocking - this should prevent the import dependency chain error
from smartcash.model.analysis.visualization.visualization_manager import VisualizationManager


class TestVisualizationManager:
    """Test cases for VisualizationManager class"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            'visualization': {
                'charts': {
                    'figure_size': [12, 8],
                    'dpi': 150,
                    'style': 'seaborn-v0_8',
                    'color_palette': 'Set2'
                }
            }
        }
    
    @pytest.fixture
    def mock_matplotlib(self):
        """Mock matplotlib components"""
        with patch('matplotlib.pyplot') as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            mock_plt.figure.return_value = mock_fig
            mock_fig.add_subplot.return_value = mock_ax
            mock_fig.add_gridspec.return_value = MagicMock()
            yield mock_plt, mock_fig, mock_ax
    
    def test_initialization_with_config(self, temp_output_dir, sample_config, mock_matplotlib):
        """Test VisualizationManager initialization with config"""
        mock_plt, mock_fig, mock_ax = mock_matplotlib
        
        with patch('smartcash.common.logger.get_logger') as mock_logger:
            vm = VisualizationManager(
                config=sample_config,
                output_dir=temp_output_dir,
                logger=mock_logger
            )
            
            assert vm.config == sample_config
            assert vm.output_dir == Path(temp_output_dir)
            assert vm.figure_size == [12, 8]
            assert vm.dpi == 150
            assert vm.style == 'seaborn-v0_8'
            assert vm.color_palette == 'Set2'
    
    def test_initialization_without_config(self, temp_output_dir, mock_matplotlib):
        """Test VisualizationManager initialization without config"""
        mock_plt, mock_fig, mock_ax = mock_matplotlib
        
        with patch('smartcash.common.logger.get_logger') as mock_logger:
            vm = VisualizationManager(
                output_dir=temp_output_dir,
                logger=mock_logger
            )
            
            assert vm.config == {}
            assert vm.output_dir.exists()
    
    @patch('seaborn.set_palette')
    @patch('matplotlib.pyplot.style')
    def test_matplotlib_style_setup(self, mock_style, mock_set_palette, temp_output_dir, sample_config):
        """Test matplotlib style setup"""
        mock_style.available = ['seaborn-v0_8', 'default']
        mock_style.use = MagicMock()
        
        with patch('smartcash.common.logger.get_logger'):
            vm = VisualizationManager(
                config=sample_config,
                output_dir=temp_output_dir
            )
            
            mock_set_palette.assert_called_with('Set2')


class TestCurrencyAnalysisPlots:
    """Test currency analysis plot generation"""
    
    @pytest.fixture
    def vm_instance(self, temp_output_dir):
        """Create VisualizationManager instance for testing"""
        with patch('smartcash.common.logger.get_logger'):
            return VisualizationManager(output_dir=temp_output_dir)
    
    @pytest.fixture
    def sample_currency_results(self):
        """Sample currency analysis results"""
        return {
            'aggregated_metrics': {
                'strategy_distribution': {
                    'single_detection': 45,
                    'multi_detection': 25,
                    'confidence_based': 30
                },
                'denomination_distribution': {
                    'Rp1K': 10,
                    'Rp2K': 8,
                    'Rp5K': 15,
                    'Rp10K': 12,
                    'Rp20K': 20,
                    'Rp50K': 18,
                    'Rp100K': 17
                }
            },
            'batch_summary': {
                'total_images': 100,
                'successful_analysis': 95,
                'images_with_currency': 80,
                'success_rate': 0.95,
                'detection_rate': 0.84
            }
        }
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    def test_generate_currency_analysis_plots(self, mock_tight_layout, mock_close, mock_savefig, 
                                            vm_instance, sample_currency_results):
        """Test currency analysis plots generation"""
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            with patch('seaborn.color_palette', return_value=['red', 'green', 'blue']):
                with patch.object(vm_instance, '_plot_strategy_distribution', return_value='strategy.png'):
                    with patch.object(vm_instance, '_plot_denomination_distribution', return_value='denom.png'):
                        with patch.object(vm_instance, '_plot_detection_rates', return_value='rates.png'):
                            
                            plot_paths = vm_instance.generate_currency_analysis_plots(
                                sample_currency_results, save_plots=True
                            )
                            
                            assert 'strategy_distribution' in plot_paths
                            assert 'denomination_distribution' in plot_paths
                            assert 'detection_rates' in plot_paths
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    def test_plot_strategy_distribution(self, mock_tight_layout, mock_close, mock_savefig, vm_instance):
        """Test strategy distribution plot"""
        strategy_data = {
            'single_detection': 45,
            'multi_detection': 25,
            'confidence_based': 30
        }
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            mock_ax.pie.return_value = ([], [], [])
            
            with patch('seaborn.color_palette', return_value=['red', 'green', 'blue']):
                result = vm_instance._plot_strategy_distribution(strategy_data)
                
                assert result is not None
                assert 'currency_strategy_distribution.png' in result
                mock_ax.pie.assert_called_once()
                mock_savefig.assert_called_once()
                mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    def test_plot_denomination_distribution(self, mock_tight_layout, mock_close, mock_savefig, vm_instance):
        """Test denomination distribution plot"""
        denom_data = {
            'Rp1K': 10,
            'Rp2K': 8,
            'Rp5K': 15,
            'Rp10K': 12
        }
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
            mock_ax1.bar.return_value = []
            mock_ax2.pie.return_value = ([], [], [])
            
            with patch('seaborn.color_palette', return_value=['red', 'green', 'blue', 'yellow']):
                with patch('matplotlib.pyplot.Circle'):
                    result = vm_instance._plot_denomination_distribution(denom_data)
                    
                    assert result is not None
                    assert 'denomination_distribution.png' in result
                    mock_ax1.bar.assert_called_once()
                    mock_ax2.pie.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    def test_plot_detection_rates(self, mock_tight_layout, mock_close, mock_savefig, vm_instance):
        """Test detection rates plot"""
        batch_summary = {
            'total_images': 100,
            'successful_analysis': 95,
            'images_with_currency': 80,
            'success_rate': 0.95,
            'detection_rate': 0.84
        }
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
            mock_ax1.pie.return_value = ([], [], [])
            mock_ax2.bar.return_value = []
            
            with patch('seaborn.color_palette', return_value=['red', 'green', 'blue', 'yellow']):
                result = vm_instance._plot_detection_rates(batch_summary)
                
                assert result is not None
                assert 'detection_rates.png' in result
                mock_ax1.pie.assert_called_once()
                mock_ax2.bar.assert_called_once()
    
    def test_empty_strategy_distribution(self, vm_instance):
        """Test strategy distribution plot with empty data"""
        result = vm_instance._plot_strategy_distribution({})
        assert result is None
    
    def test_empty_denomination_distribution(self, vm_instance):
        """Test denomination distribution plot with empty data"""
        result = vm_instance._plot_denomination_distribution({})
        assert result is None


class TestLayerAnalysisPlots:
    """Test layer analysis plot generation"""
    
    @pytest.fixture
    def vm_instance(self, temp_output_dir):
        """Create VisualizationManager instance for testing"""
        with patch('smartcash.common.logger.get_logger'):
            return VisualizationManager(output_dir=temp_output_dir)
    
    @pytest.fixture
    def sample_layer_results(self):
        """Sample layer analysis results"""
        return {
            'aggregated_layer_metrics': {
                'layer_1': {
                    'avg_precision': 0.85,
                    'avg_recall': 0.80,
                    'avg_f1_score': 0.82,
                    'avg_confidence': 0.88
                },
                'layer_2': {
                    'avg_precision': 0.78,
                    'avg_recall': 0.75,
                    'avg_f1_score': 0.76,
                    'avg_confidence': 0.82
                },
                'layer_3': {
                    'avg_precision': 0.82,
                    'avg_recall': 0.85,
                    'avg_f1_score': 0.83,
                    'avg_confidence': 0.79
                }
            },
            'batch_insights': {
                'layer_activity_rates': {
                    'layer_1': 0.95,
                    'layer_2': 0.87,
                    'layer_3': 0.92
                }
            },
            'layer_consistency': {
                'layer_1': {'consistency_score': 0.88},
                'layer_2': {'consistency_score': 0.82},
                'layer_3': {'consistency_score': 0.85}
            }
        }
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('pandas.DataFrame')
    def test_generate_layer_analysis_plots(self, mock_dataframe, mock_tight_layout, 
                                         mock_close, mock_savefig, vm_instance, sample_layer_results):
        """Test layer analysis plots generation"""
        # Mock pandas DataFrame
        mock_df = MagicMock()
        mock_df.pivot.return_value = MagicMock()
        mock_dataframe.return_value = mock_df
        
        with patch.object(vm_instance, '_plot_layer_performance', return_value='layer_perf.png'):
            with patch.object(vm_instance, '_plot_layer_utilization', return_value='layer_util.png'):
                with patch.object(vm_instance, '_plot_layer_consistency', return_value='layer_cons.png'):
                    
                    plot_paths = vm_instance.generate_layer_analysis_plots(
                        sample_layer_results, save_plots=True
                    )
                    
                    assert 'layer_performance' in plot_paths
                    assert 'layer_utilization' in plot_paths
                    assert 'layer_consistency' in plot_paths
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('pandas.DataFrame')
    @patch('seaborn.heatmap')
    @patch('seaborn.barplot')
    def test_plot_layer_performance(self, mock_barplot, mock_heatmap, mock_dataframe,
                                  mock_tight_layout, mock_close, mock_savefig, vm_instance):
        """Test layer performance plot"""
        layer_metrics = {
            'layer_1': {
                'avg_precision': 0.85,
                'avg_recall': 0.80,
                'avg_f1_score': 0.82,
                'avg_confidence': 0.88
            },
            'layer_2': {
                'avg_precision': 0.78,
                'avg_recall': 0.75,
                'avg_f1_score': 0.76,
                'avg_confidence': 0.82
            }
        }
        
        # Mock pandas DataFrame and pivot
        mock_df = MagicMock()
        mock_pivot = MagicMock()
        mock_df.pivot.return_value = mock_pivot
        mock_dataframe.return_value = mock_df
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
            
            result = vm_instance._plot_layer_performance(layer_metrics)
            
            assert result is not None
            assert 'layer_performance_comparison.png' in result
            mock_heatmap.assert_called_once()
            mock_barplot.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    def test_plot_layer_utilization(self, mock_tight_layout, mock_close, mock_savefig, vm_instance):
        """Test layer utilization plot"""
        activity_rates = {
            'layer_1': 0.95,
            'layer_2': 0.87,
            'layer_3': 0.92
        }
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            mock_ax.barh.return_value = []
            
            result = vm_instance._plot_layer_utilization(activity_rates)
            
            assert result is not None
            assert 'layer_utilization.png' in result
            mock_ax.barh.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    def test_plot_layer_consistency(self, mock_tight_layout, mock_close, mock_savefig, vm_instance):
        """Test layer consistency plot"""
        consistency_data = {
            'layer_1': {'consistency_score': 0.88},
            'layer_2': {'consistency_score': 0.82},
            'layer_3': {'consistency_score': 0.85}
        }
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            mock_ax.bar.return_value = []
            
            with patch('seaborn.color_palette', return_value=['red', 'green', 'blue']):
                result = vm_instance._plot_layer_consistency(consistency_data)
                
                assert result is not None
                assert 'layer_consistency.png' in result
                mock_ax.bar.assert_called_once()


class TestClassAnalysisPlots:
    """Test class analysis plot generation"""
    
    @pytest.fixture
    def vm_instance(self, temp_output_dir):
        """Create VisualizationManager instance for testing"""
        with patch('smartcash.common.logger.get_logger'):
            return VisualizationManager(output_dir=temp_output_dir)
    
    @pytest.fixture
    def sample_class_results(self):
        """Sample class analysis results"""
        return {
            'per_class_metrics': {
                'Rp1K': {'precision': 0.85, 'recall': 0.80, 'f1_score': 0.82, 'ap': 0.83},
                'Rp2K': {'precision': 0.78, 'recall': 0.75, 'f1_score': 0.76, 'ap': 0.77},
                'Rp5K': {'precision': 0.82, 'recall': 0.85, 'f1_score': 0.83, 'ap': 0.84}
            },
            'confusion_matrix': {
                'matrix': [[50, 2, 1], [3, 45, 2], [1, 2, 48]],
                'class_names': ['Rp1K', 'Rp2K', 'Rp5K']
            },
            'class_distribution': {
                'Rp1K': 53,
                'Rp2K': 50,
                'Rp5K': 51
            }
        }
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    def test_generate_class_analysis_plots(self, mock_tight_layout, mock_close, 
                                         mock_savefig, vm_instance, sample_class_results):
        """Test class analysis plots generation"""
        with patch.object(vm_instance, '_plot_class_metrics_heatmap', return_value='heatmap.png'):
            with patch.object(vm_instance, '_plot_confusion_matrix', return_value='confusion.png'):
                with patch.object(vm_instance, '_plot_class_distribution', return_value='distribution.png'):
                    
                    plot_paths = vm_instance.generate_class_analysis_plots(
                        sample_class_results, save_plots=True
                    )
                    
                    assert 'class_metrics_heatmap' in plot_paths
                    assert 'confusion_matrix' in plot_paths
                    assert 'class_distribution' in plot_paths
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('seaborn.heatmap')
    def test_plot_class_metrics_heatmap(self, mock_heatmap, mock_tight_layout, 
                                      mock_close, mock_savefig, vm_instance):
        """Test class metrics heatmap plot"""
        class_metrics = {
            'Rp1K': {'precision': 0.85, 'recall': 0.80, 'f1_score': 0.82, 'ap': 0.83},
            'Rp2K': {'precision': 0.78, 'recall': 0.75, 'f1_score': 0.76, 'ap': 0.77}
        }
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = vm_instance._plot_class_metrics_heatmap(class_metrics)
            
            assert result is not None
            assert 'class_metrics_heatmap.png' in result
            mock_heatmap.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('seaborn.heatmap')
    def test_plot_confusion_matrix(self, mock_heatmap, mock_tight_layout, 
                                 mock_close, mock_savefig, vm_instance):
        """Test confusion matrix plot"""
        confusion_data = {
            'matrix': [[50, 2, 1], [3, 45, 2], [1, 2, 48]],
            'class_names': ['Rp1K', 'Rp2K', 'Rp5K']
        }
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = vm_instance._plot_confusion_matrix(confusion_data)
            
            assert result is not None
            assert 'confusion_matrix.png' in result
            mock_heatmap.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    def test_plot_class_distribution(self, mock_tight_layout, mock_close, 
                                   mock_savefig, vm_instance):
        """Test class distribution plot"""
        class_distribution = {
            'Rp1K': 53,
            'Rp2K': 50,
            'Rp5K': 51
        }
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            mock_ax.bar.return_value = []
            
            with patch('seaborn.color_palette', return_value=['red', 'green', 'blue']):
                result = vm_instance._plot_class_distribution(class_distribution)
                
                assert result is not None
                assert 'class_distribution.png' in result
                mock_ax.bar.assert_called_once()


class TestComparisonPlots:
    """Test comparison plot generation"""
    
    @pytest.fixture
    def vm_instance(self, temp_output_dir):
        """Create VisualizationManager instance for testing"""
        with patch('smartcash.common.logger.get_logger'):
            return VisualizationManager(output_dir=temp_output_dir)
    
    @pytest.fixture
    def sample_comparison_data(self):
        """Sample comparison data"""
        return {
            'backbone_comparison': {
                'cspdarknet': {
                    'accuracy': 0.85,
                    'precision': 0.83,
                    'recall': 0.87,
                    'f1_score': 0.85,
                    'inference_time': 0.05
                },
                'efficientnet_b4': {
                    'accuracy': 0.82,
                    'precision': 0.80,
                    'recall': 0.84,
                    'f1_score': 0.82,
                    'inference_time': 0.08
                }
            },
            'scenario_comparison': {
                'scenario_1': {
                    'map': 0.85,
                    'accuracy': 0.87,
                    'precision': 0.83,
                    'recall': 0.89,
                    'f1_score': 0.86
                },
                'scenario_2': {
                    'map': 0.82,
                    'accuracy': 0.84,
                    'precision': 0.80,
                    'recall': 0.88,
                    'f1_score': 0.84
                }
            },
            'efficiency_analysis': {
                'model_a': {'accuracy': 0.85, 'inference_time': 0.05},
                'model_b': {'accuracy': 0.82, 'inference_time': 0.03}
            }
        }
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    def test_generate_comparison_plots(self, mock_tight_layout, mock_close, 
                                     mock_savefig, vm_instance, sample_comparison_data):
        """Test comparison plots generation"""
        with patch.object(vm_instance, '_plot_backbone_comparison', return_value='backbone.png'):
            with patch.object(vm_instance, '_plot_scenario_comparison', return_value='scenario.png'):
                with patch.object(vm_instance, '_plot_efficiency_analysis', return_value='efficiency.png'):
                    
                    plot_paths = vm_instance.generate_comparison_plots(
                        sample_comparison_data, save_plots=True
                    )
                    
                    assert 'backbone_comparison' in plot_paths
                    assert 'scenario_comparison' in plot_paths
                    assert 'efficiency_analysis' in plot_paths
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    def test_plot_backbone_comparison(self, mock_tight_layout, mock_close, 
                                    mock_savefig, vm_instance):
        """Test backbone comparison plot"""
        backbone_data = {
            'cspdarknet': {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87,
                'f1_score': 0.85,
                'inference_time': 0.05
            },
            'efficientnet_b4': {
                'accuracy': 0.82,
                'precision': 0.80,
                'recall': 0.84,
                'f1_score': 0.82,
                'inference_time': 0.08
            }
        }
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_axes = [MagicMock() for _ in range(6)]
            mock_fig.suptitle = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            for ax in mock_axes:
                ax.bar.return_value = []
                ax.set_title = MagicMock()
                ax.set_ylabel = MagicMock()
                ax.text = MagicMock()
                ax.set_visible = MagicMock()
            
            with patch('seaborn.color_palette', return_value=['red', 'blue']):
                result = vm_instance._plot_backbone_comparison(backbone_data)
                
                assert result is not None
                assert 'backbone_comparison.png' in result
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    def test_plot_scenario_comparison(self, mock_tight_layout, mock_close, 
                                    mock_savefig, vm_instance):
        """Test scenario comparison radar plot"""
        scenario_data = {
            'scenario_1': {
                'map': 0.85,
                'accuracy': 0.87,
                'precision': 0.83,
                'recall': 0.89,
                'f1_score': 0.86
            },
            'scenario_2': {
                'map': 0.82,
                'accuracy': 0.84,
                'precision': 0.80,
                'recall': 0.88,
                'f1_score': 0.84
            }
        }
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            with patch('numpy.linspace', return_value=[0, 1, 2, 3, 4]):
                with patch('seaborn.color_palette', return_value=['red', 'blue']):
                    result = vm_instance._plot_scenario_comparison(scenario_data)
                    
                    assert result is not None
                    assert 'scenario_comparison_radar.png' in result
                    assert mock_ax.plot.call_count == 2  # Two scenarios
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    def test_plot_efficiency_analysis(self, mock_tight_layout, mock_close, 
                                    mock_savefig, vm_instance):
        """Test efficiency analysis scatter plot"""
        efficiency_data = {
            'model_a': {'accuracy': 0.85, 'inference_time': 0.05},
            'model_b': {'accuracy': 0.82, 'inference_time': 0.03},
            'model_c': {'accuracy': 0.88, 'inference_time': 0.08}
        }
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            mock_ax.scatter.return_value = MagicMock()
            
            with patch('seaborn.color_palette', return_value=['red', 'green', 'blue']):
                # Mock scipy ConvexHull
                with patch('scipy.spatial.ConvexHull') as mock_hull:
                    mock_hull_instance = MagicMock()
                    mock_hull_instance.vertices = [0, 1, 2]
                    mock_hull.return_value = mock_hull_instance
                    
                    with patch('numpy.column_stack'):
                        result = vm_instance._plot_efficiency_analysis(efficiency_data)
                        
                        assert result is not None
                        assert 'efficiency_analysis.png' in result
                        mock_ax.scatter.assert_called_once()


class TestComprehensiveDashboard:
    """Test comprehensive dashboard generation"""
    
    @pytest.fixture
    def vm_instance(self, temp_output_dir):
        """Create VisualizationManager instance for testing"""
        with patch('smartcash.common.logger.get_logger'):
            return VisualizationManager(output_dir=temp_output_dir)
    
    @pytest.fixture
    def sample_all_results(self):
        """Sample comprehensive results data"""
        return {
            'currency_results': {
                'aggregated_metrics': {
                    'strategy_distribution': {
                        'single_detection': 45,
                        'multi_detection': 30,
                        'confidence_based': 25
                    }
                }
            },
            'layer_results': {
                'aggregated_layer_metrics': {
                    'layer_1': {'avg_precision': 0.85},
                    'layer_2': {'avg_precision': 0.78},
                    'layer_3': {'avg_precision': 0.82}
                }
            }
        }
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_comprehensive_dashboard(self, mock_close, mock_savefig, 
                                           vm_instance, sample_all_results):
        """Test comprehensive dashboard generation"""
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = MagicMock()
            mock_figure.return_value = mock_fig
            mock_fig.add_gridspec.return_value = MagicMock()
            mock_fig.add_subplot.return_value = MagicMock()
            mock_fig.suptitle = MagicMock()
            
            result = vm_instance.generate_comprehensive_dashboard(
                sample_all_results, save_dashboard=True
            )
            
            assert result is not None
            assert 'comprehensive_dashboard.png' in result
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    def test_dashboard_generation_error_handling(self, vm_instance):
        """Test dashboard generation error handling"""
        # Test with invalid data that should cause an error
        invalid_results = "invalid_data"
        
        result = vm_instance.generate_comprehensive_dashboard(invalid_results)
        
        assert result is None  # Should return None on error


class TestUtilityFunctions:
    """Test utility functions"""
    
    @pytest.fixture
    def vm_instance(self, temp_output_dir):
        """Create VisualizationManager instance for testing"""
        with patch('smartcash.common.logger.get_logger'):
            return VisualizationManager(output_dir=temp_output_dir)
    
    def test_cleanup_plots(self, vm_instance, temp_output_dir):
        """Test plot cleanup functionality"""
        # Create some test plot files
        for i in range(15):
            plot_file = vm_instance.output_dir / f"test_plot_{i}.png"
            plot_file.touch()
        
        # Test cleanup (keep 10 latest)
        vm_instance.cleanup_plots(keep_latest=10)
        
        remaining_files = list(vm_instance.output_dir.glob('*.png'))
        assert len(remaining_files) <= 10
    
    def test_cleanup_plots_with_no_files(self, vm_instance):
        """Test plot cleanup with no existing files"""
        # Should not raise error
        vm_instance.cleanup_plots(keep_latest=10)
    
    def test_cleanup_plots_error_handling(self, vm_instance):
        """Test plot cleanup error handling"""
        # Mock Path.glob to raise an exception
        with patch.object(vm_instance.output_dir, 'glob', side_effect=Exception("Test error")):
            # Should not raise error, but log warning
            vm_instance.cleanup_plots(keep_latest=10)


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.fixture
    def vm_instance(self, temp_output_dir):
        """Create VisualizationManager instance for testing"""
        with patch('smartcash.common.logger.get_logger'):
            return VisualizationManager(output_dir=temp_output_dir)
    
    def test_generate_currency_plots_with_error(self, vm_instance):
        """Test currency plots generation with error"""
        # Mock _plot_strategy_distribution to raise an exception
        with patch.object(vm_instance, '_plot_strategy_distribution', side_effect=Exception("Test error")):
            currency_results = {
                'aggregated_metrics': {
                    'strategy_distribution': {'test': 1}
                }
            }
            
            plot_paths = vm_instance.generate_currency_analysis_plots(currency_results)
            
            assert plot_paths == {}  # Should return empty dict on error
    
    def test_generate_layer_plots_with_error(self, vm_instance):
        """Test layer plots generation with error"""
        with patch.object(vm_instance, '_plot_layer_performance', side_effect=Exception("Test error")):
            layer_results = {
                'aggregated_layer_metrics': {'layer_1': {'avg_precision': 0.85}}
            }
            
            plot_paths = vm_instance.generate_layer_analysis_plots(layer_results)
            
            assert plot_paths == {}
    
    def test_generate_class_plots_with_error(self, vm_instance):
        """Test class plots generation with error"""
        with patch.object(vm_instance, '_plot_class_metrics_heatmap', side_effect=Exception("Test error")):
            class_results = {
                'per_class_metrics': {'class1': {'precision': 0.85}}
            }
            
            plot_paths = vm_instance.generate_class_analysis_plots(class_results)
            
            assert plot_paths == {}
    
    def test_generate_comparison_plots_with_error(self, vm_instance):
        """Test comparison plots generation with error"""
        with patch.object(vm_instance, '_plot_backbone_comparison', side_effect=Exception("Test error")):
            comparison_data = {
                'backbone_comparison': {'backbone1': {'accuracy': 0.85}}
            }
            
            plot_paths = vm_instance.generate_comparison_plots(comparison_data)
            
            assert plot_paths == {}
    
    def test_plot_functions_with_none_data(self, vm_instance):
        """Test plot functions with None data"""
        assert vm_instance._plot_strategy_distribution(None) is None
        assert vm_instance._plot_denomination_distribution(None) is None
        assert vm_instance._plot_layer_performance(None) is None
        assert vm_instance._plot_layer_utilization(None) is None
        assert vm_instance._plot_layer_consistency(None) is None
        assert vm_instance._plot_backbone_comparison(None) is None
        assert vm_instance._plot_scenario_comparison(None) is None
        assert vm_instance._plot_efficiency_analysis(None) is None
    
    def test_confusion_matrix_without_matrix_key(self, vm_instance):
        """Test confusion matrix plot without matrix key"""
        confusion_data = {'class_names': ['class1', 'class2']}  # Missing 'matrix' key
        
        result = vm_instance._plot_confusion_matrix(confusion_data)
        
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])