"""
file_path: tests/ui/dataset/visualization/conftest.py
Pytest configuration and fixtures for visualization module tests.
"""
import pytest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

@pytest.fixture
def mock_visualization_config():
    """Mock configuration for visualization module."""
    return {
        "title": "Visualization Test",
        "description": "Test visualization module",
        "ui_config": {
            "title": "Test Visualization",
            "subtitle": "Test Subtitle",
            "icon": "📊"
        },
        "button_config": {
            "analyze": {"label": "Analyze", "icon": "🔍"},
            "export": {"label": "Export", "icon": "💾"},
            "refresh": {"label": "Refresh", "icon": "🔄"}
        }
    }

@pytest.fixture
def mock_ui_components():
    """Mock UI components for testing."""
    return {
        'containers': {
            'header_container': widgets.VBox(),
            'form_container': widgets.VBox(),
            'dashboard_container': widgets.VBox(),
            'visualization_container': widgets.VBox(),
            'controls_container': widgets.VBox(),
            'progress_container': widgets.VBox(),
            'footer_container': widgets.VBox()
        },
        'widgets': {
            'visualization_type': widgets.Dropdown(
                options=['bar', 'line', 'scatter'],
                value='bar'
            ),
            'split_selector': widgets.Dropdown(
                options=['train', 'valid', 'test', 'all'],
                value='all'
            )
        }
    }

@pytest.fixture
def mock_visualization_module(mock_visualization_config, mock_ui_components):
    """Create a test instance of VisualizationUIModule with mocks."""
    with patch('smartcash.ui.dataset.visualization.visualization_uimodule.create_visualization_ui', 
              return_value=mock_ui_components):
        from smartcash.ui.dataset.visualization.visualization_uimodule import VisualizationUIModule
        module = VisualizationUIModule(enable_environment=False)
        
        # Mock the config handler
        module._config_handler = MagicMock()
        module._config_handler.config = mock_visualization_config
        
        # Mock the operations
        module._operations = {
            'refresh': MagicMock(),
            'load_preprocessed': MagicMock(),
            'load_augmented': MagicMock()
        }
        
        # Mock the backend APIs
        module._backend_apis = {
            'dataset': MagicMock(),
            'augmentation': MagicMock()
        }
        
        return module

@pytest.fixture
def sample_stats_data():
    """Sample statistics data for testing."""
    return {
        'dataset_stats': {
            'success': True,
            'by_split': {
                'train': {'raw': 100, 'preprocessed': 80, 'augmented': 200},
                'valid': {'raw': 20, 'preprocessed': 15, 'augmented': 0},
                'test': {'raw': 30, 'preprocessed': 25, 'augmented': 0}
            },
            'overview': {
                'total_files': 150,
                'total_size_mb': 45.7,
                'avg_file_size_kb': 311.8
            },
            'last_updated': '2025-07-22 13:00:00'
        },
        'augmentation_stats': {
            'success': True,
            'by_split': {
                'train': {'file_count': 200},
                'valid': {'file_count': 0},
                'test': {'file_count': 0}
            },
            'last_updated': '2025-07-22 13:00:00'
        }
    }
