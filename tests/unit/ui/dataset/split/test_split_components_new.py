"""
Tests for the dataset split UI components.

This module tests the UI component creation functions and their
integration for the dataset split configuration interface.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, ANY
from typing import Dict, Any, List, Tuple, Callable
import copy
import ipywidgets as widgets

from smartcash.ui.dataset.split.components.split_ui import create_split_ui_components
from smartcash.ui.dataset.split.configs.split_defaults import DEFAULT_SPLIT_CONFIG
from smartcash.ui.dataset.split.constants import UI_CONFIG, SPLIT_RATIO_DEFAULTS


class TestSplitUIComponents:
    """Test cases for split UI component creation."""
    
    @pytest.fixture
    def mock_config(self) -> Dict[str, Any]:
        """Provide a mock configuration for testing."""
        return copy.deepcopy(DEFAULT_SPLIT_CONFIG)
    
    @pytest.fixture
    def mock_ui_components(self) -> Dict[str, Any]:
        """Create mock UI components for testing."""
        return {
            'header_container': MagicMock(),
            'form_container': MagicMock(),
            'action_container': MagicMock(),
            'footer_container': MagicMock(),
            'main_container': MagicMock()
        }
        
    @pytest.fixture
    def mock_widgets(self) -> Dict[str, Any]:
        """Create mock widgets for testing."""
        return {
            'train_ratio': widgets.FloatSlider(value=0.7, min=0.1, max=0.9, step=0.05),
            'val_ratio': widgets.FloatSlider(value=0.2, min=0.1, max=0.5, step=0.05),
            'test_ratio': widgets.FloatSlider(value=0.1, min=0.0, max=0.5, step=0.05),
            'input_dir': widgets.Text(value='data/input'),
            'output_dir': widgets.Text(value='data/output'),
            'random_seed': widgets.IntText(value=42),
            'stratify': widgets.Checkbox(value=True)
        }

    def test_create_split_ui_components_default_config(
        self, 
        mock_config: Dict[str, Any],
        mock_ui_components: Dict[str, Any],
        mock_widgets: Dict[str, Any]
    ):
        """Test UI component creation with default configuration."""
        with patch('smartcash.ui.dataset.split.components.split_ui.create_header_container') as mock_header, \
             patch('smartcash.ui.dataset.split.components.split_ui.create_form_container') as mock_form, \
             patch('smartcash.ui.dataset.split.components.split_ui.create_action_container') as mock_action, \
             patch('smartcash.ui.dataset.split.components.split_ui.create_footer_container') as mock_footer, \
             patch('smartcash.ui.dataset.split.components.split_ui.create_main_container') as mock_main:
            
            # Setup mock returns
            mock_header.return_value = MagicMock(container=MagicMock())
            mock_form.return_value = MagicMock(container=MagicMock(), get_form_container=lambda: MagicMock(children=[]))
            mock_action.return_value = MagicMock(container=MagicMock(), buttons={}, primary_button=MagicMock())
            mock_footer.return_value = MagicMock(container=MagicMock())
            mock_main.return_value = MagicMock(container=MagicMock())
            
            # Call the function
            components = create_split_ui_components(mock_config)
            
            # Assertions
            assert isinstance(components, dict)
            assert 'header_container' in components
            assert 'form_container' in components
            assert 'action_container' in components
            assert 'footer_container' in components
            assert 'main_container' in components
            
            # Verify UI component creation
            mock_header.assert_called_once_with(
                title=UI_CONFIG['title'],
                subtitle=UI_CONFIG['subtitle'],
                icon=UI_CONFIG['icon']
            )
            mock_form.assert_called_once()
            mock_action.assert_called_once()
            mock_footer.assert_called_once()
            mock_main.assert_called_once()
    
    def test_ratio_section_creation(self, mock_config: Dict[str, Any], mock_widgets: Dict[str, Any]):
        """Test creation of ratio section with default values."""
        with patch('smartcash.ui.dataset.split.components.split_ui.create_ratio_section') as mock_ratio_section:
            mock_ratio_section.return_value = {
                'train_ratio': mock_widgets['train_ratio'],
                'val_ratio': mock_widgets['val_ratio'],
                'test_ratio': mock_widgets['test_ratio']
            }
            
            # Call the function
            result = create_split_ui_components(mock_config)
            
            # Assertions
            assert 'train_ratio' in result
            assert 'val_ratio' in result
            assert 'test_ratio' in result
            assert result['train_ratio'].value == SPLIT_RATIO_DEFAULTS['train']
            assert result['val_ratio'].value == SPLIT_RATIO_DEFAULTS['val']
            assert result['test_ratio'].value == SPLIT_RATIO_DEFAULTS['test']
    
    def test_path_section_creation(self, mock_config: Dict[str, Any], mock_widgets: Dict[str, Any]):
        """Test creation of path section with default values."""
        with patch('smartcash.ui.dataset.split.components.split_ui.create_path_section') as mock_path_section:
            mock_path_section.return_value = {
                'input_dir': mock_widgets['input_dir'],
                'output_dir': mock_widgets['output_dir']
            }
            
            # Call the function
            result = create_split_ui_components(mock_config)
            
            # Assertions
            assert 'input_dir' in result
            assert 'output_dir' in result
            assert result['input_dir'].value == 'data/input'
            assert result['output_dir'].value == 'data/output'
    
    def test_advanced_section_creation(self, mock_config: Dict[str, Any], mock_widgets: Dict[str, Any]):
        """Test creation of advanced options section."""
        with patch('smartcash.ui.dataset.split.components.split_ui.create_advanced_section') as mock_advanced_section:
            mock_advanced_section.return_value = {
                'random_seed': mock_widgets['random_seed'],
                'stratify': mock_widgets['stratify']
            }
            
            # Call the function
            result = create_split_ui_components(mock_config)
            
            # Assertions
            assert 'random_seed' in result
            assert 'stratify' in result
            assert result['random_seed'].value == 42
            assert result['stratify'].value is True
