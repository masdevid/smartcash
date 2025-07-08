"""
File: tests/unit/ui/dataset/preprocess/conftest.py
Description: Pytest configuration and fixtures for preprocessing tests
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from smartcash.ui.dataset.preprocess.configs.preprocess_defaults import get_default_preprocessing_config
from smartcash.ui.dataset.preprocess.configs.preprocess_config_handler import PreprocessConfigHandler


@pytest.fixture
def mock_ui_components():
    """Mock UI components for testing"""
    return {
        'preprocess_btn': Mock(),
        'check_btn': Mock(),
        'cleanup_btn': Mock(),
        'operation_container': Mock(),
        'progress_tracker': Mock(),
        'log_accordion': Mock(),
        'confirmation_dialog': Mock(),
        'summary_container': Mock(),
        'update_status': Mock(),
        'resolution_dropdown': Mock(value='yolov5s'),
        'normalization_dropdown': Mock(value='minmax'),
        'preserve_aspect_checkbox': Mock(value=True),
        'target_splits_select': Mock(value=['train', 'valid']),
        'batch_size_input': Mock(value=32),
        'validation_checkbox': Mock(value=False),
        'move_invalid_checkbox': Mock(value=False),
        'invalid_dir_input': Mock(value='data/invalid'),
        'cleanup_target_dropdown': Mock(value='preprocessed'),
        'backup_checkbox': Mock(value=True)
    }


@pytest.fixture
def default_config():
    """Default preprocessing configuration for testing"""
    return get_default_preprocessing_config()


@pytest.fixture
def custom_config():
    """Custom preprocessing configuration for testing"""
    return {
        'preprocessing': {
            'normalization': {
                'preset': 'yolov5l',
                'method': 'zscore',
                'target_size': [832, 832],
                'preserve_aspect_ratio': False
            },
            'target_splits': ['train', 'test'],
            'batch_size': 64,
            'validation': {
                'enabled': True,
                'structure_check': True,
                'filename_check': True
            },
            'move_invalid': True,
            'invalid_dir': 'data/bad',
            'cleanup_target': 'both',
            'backup_enabled': False
        },
        'data': {
            'dir': 'custom_data',
            'preprocessed_dir': 'custom_output'
        },
        'performance': {
            'num_workers': 8,
            'pin_memory': True
        }
    }


@pytest.fixture
def mock_config_handler():
    """Mock configuration handler for testing"""
    handler = Mock(spec=PreprocessConfigHandler)
    handler.get_default_config.return_value = get_default_preprocessing_config()
    handler.extract_config_from_ui.return_value = get_default_preprocessing_config()
    handler.validate_config.return_value = (True, [])
    handler.module_name = 'preprocess'
    handler.parent_module = 'dataset'
    return handler


@pytest.fixture
def mock_backend_apis():
    """Mock backend API functions for testing"""
    return {
        'preprocess_dataset': Mock(return_value={
            'success': True,
            'message': 'Preprocessing completed',
            'stats': {'total_files': 100, 'processed_files': 100},
            'processed_splits': ['train', 'valid']
        }),
        'get_preprocessing_status': Mock(return_value={
            'success': True,
            'service_ready': True,
            'file_statistics': {
                'train': {'raw_images': 50, 'preprocessed_files': 50},
                'valid': {'raw_images': 25, 'preprocessed_files': 25}
            }
        }),
        'cleanup_preprocessing_files': Mock(return_value={
            'success': True,
            'files_removed': 75,
            'message': 'Cleanup completed'
        }),
        'get_cleanup_preview': Mock(return_value={
            'success': True,
            'total_files': 75,
            'total_size_mb': 150.0
        }),
        'get_dataset_stats': Mock(return_value={
            'success': True,
            'by_split': {
                'train': {
                    'file_counts': {'raw': 50, 'preprocessed': 50},
                    'total_size_mb': 100.0
                },
                'valid': {
                    'file_counts': {'raw': 25, 'preprocessed': 25},
                    'total_size_mb': 50.0
                }
            }
        })
    }


@pytest.fixture
def operation_results():
    """Sample operation results for testing"""
    return {
        'preprocess': {
            'operation': 'preprocess',
            'success': True,
            'message': 'Preprocessing completed successfully',
            'stats': {
                'total_files': 200,
                'processed_files': 195,
                'failed_files': 5
            },
            'configuration': {
                'normalization_preset': 'yolov5l',
                'target_splits': ['train', 'valid', 'test']
            },
            'processing_time': 120.5
        },
        'check': {
            'operation': 'check',
            'success': True,
            'service_ready': True,
            'file_statistics': {
                'train': {'raw_images': 100, 'preprocessed_files': 90},
                'valid': {'raw_images': 50, 'preprocessed_files': 45},
                'test': {'raw_images': 25, 'preprocessed_files': 20}
            }
        },
        'cleanup': {
            'operation': 'cleanup',
            'success': True,
            'files_removed': 150,
            'cleanup_target': 'preprocessed',
            'affected_splits': ['train', 'valid', 'test']
        }
    }


@pytest.fixture
def existing_data_samples():
    """Sample existing data scenarios for testing"""
    return {
        'no_existing': {
            'has_existing': False,
            'by_split': {},
            'total_existing': 0,
            'requires_confirmation': False
        },
        'some_existing': {
            'has_existing': True,
            'by_split': {
                'train': {'existing_files': 30, 'path': 'data/preprocessed/train'},
                'valid': {'existing_files': 20, 'path': 'data/preprocessed/valid'}
            },
            'total_existing': 50,
            'requires_confirmation': True
        },
        'all_existing': {
            'has_existing': True,
            'by_split': {
                'train': {'existing_files': 100, 'path': 'data/preprocessed/train'},
                'valid': {'existing_files': 50, 'path': 'data/preprocessed/valid'},
                'test': {'existing_files': 25, 'path': 'data/preprocessed/test'}
            },
            'total_existing': 175,
            'requires_confirmation': True
        }
    }


@pytest.fixture
def cleanup_preview_samples():
    """Sample cleanup preview scenarios for testing"""
    return {
        'no_files': {
            'success': True,
            'total_files': 0,
            'total_size_mb': 0.0,
            'by_split': {}
        },
        'some_files': {
            'success': True,
            'total_files': 50,
            'total_size_mb': 125.5,
            'by_split': {
                'train': {'files': 30, 'size_mb': 75.0},
                'valid': {'files': 20, 'size_mb': 50.5}
            }
        },
        'many_files': {
            'success': True,
            'total_files': 500,
            'total_size_mb': 1250.0,
            'by_split': {
                'train': {'files': 300, 'size_mb': 750.0},
                'valid': {'files': 150, 'size_mb': 375.0},
                'test': {'files': 50, 'size_mb': 125.0}
            }
        }
    }


@pytest.fixture(autouse=True)
def mock_backend_imports():
    """Automatically mock backend imports for all tests"""
    with patch('smartcash.dataset.preprocessor.preprocess_dataset') as mock_preprocess:
        with patch('smartcash.dataset.preprocessor.get_preprocessing_status') as mock_status:
            with patch('smartcash.dataset.preprocessor.api.cleanup_api.cleanup_preprocessing_files') as mock_cleanup:
                with patch('smartcash.dataset.preprocessor.api.cleanup_api.get_cleanup_preview') as mock_preview:
                    with patch('smartcash.dataset.preprocessor.get_dataset_stats') as mock_stats:
                        yield {
                            'preprocess_dataset': mock_preprocess,
                            'get_preprocessing_status': mock_status,
                            'cleanup_preprocessing_files': mock_cleanup,
                            'get_cleanup_preview': mock_preview,
                            'get_dataset_stats': mock_stats
                        }


@pytest.fixture
def mock_ipython_environment():
    """Mock IPython environment for UI testing"""
    with patch('IPython.get_ipython') as mock_ipython:
        mock_ipython.return_value = Mock()
        yield mock_ipython


# Helper functions for tests

def assert_widget_properties(widget, expected_properties: Dict[str, Any]):
    """Assert that widget has expected properties"""
    for prop, value in expected_properties.items():
        assert hasattr(widget, prop), f"Widget missing property: {prop}"
        if value is not None:
            assert getattr(widget, prop) == value, f"Widget property {prop} = {getattr(widget, prop)}, expected {value}"


def assert_ui_component_structure(ui_components: Dict[str, Any], required_components: list):
    """Assert that UI components contain required components"""
    for component in required_components:
        assert component in ui_components, f"Missing required component: {component}"
        assert ui_components[component] is not None, f"Component {component} is None"


def create_mock_progress_callback():
    """Create a mock progress callback for testing"""
    callback = Mock()
    callback.side_effect = lambda level, current, total, message: None
    return callback


def create_mock_log_callback():
    """Create a mock log callback for testing"""
    callback = Mock()
    callback.side_effect = lambda level, message: None
    return callback