"""
Tests for the dataset split UI components.

This module tests the UI component creation functions and their
integration for the dataset split configuration interface.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import copy

from smartcash.ui.dataset.split.components.split_ui import create_split_ui_components
from smartcash.ui.dataset.split.configs.split_defaults import DEFAULT_SPLIT_CONFIG


class TestSplitUIComponents:
    """Test cases for split UI component creation."""
    
    @pytest.fixture
    def mock_config(self):
        """Provide a mock configuration for testing."""
        return copy.deepcopy(DEFAULT_SPLIT_CONFIG)
    
    @patch('smartcash.ui.dataset.split.components.split_ui.create_ratio_section')
    @patch('smartcash.ui.dataset.split.components.split_ui.create_path_section')
    @patch('smartcash.ui.dataset.split.components.split_ui.create_advanced_section')
    @patch('smartcash.ui.dataset.split.components.split_ui.create_main_container')
    @patch('smartcash.ui.dataset.split.components.split_ui.create_header_container')
    @patch('smartcash.ui.dataset.split.components.split_ui.create_form_container')
    @patch('smartcash.ui.dataset.split.components.split_ui.create_action_container')
    @patch('smartcash.ui.dataset.split.components.split_ui.create_footer_container')
    def test_create_split_ui_components_default_config(
        self,
        mock_footer,
        mock_action,
        mock_form,
        mock_header,
        mock_main,
        mock_advanced,
        mock_path,
        mock_ratio
    ):
        """Test UI component creation with default configuration."""
        # Setup mock returns
        mock_ratio.return_value = {
            'ratio_section': Mock(),
            'train_ratio': Mock(),
            'val_ratio': Mock(),
            'test_ratio': Mock()
        }
        mock_path.return_value = {
            'path_section': Mock(),
            'train_dir': Mock(),
            'val_dir': Mock(),
            'test_dir': Mock()
        }
        mock_advanced.return_value = {
            'advanced_section': Mock(),
            'seed': Mock(),
            'shuffle': Mock(),
            'stratify': Mock()
        }
        mock_header.return_value = Mock()
        mock_form.return_value = {'container': Mock()}
        mock_action.return_value = {
            'container': Mock(),
            'buttons': {
                'save_button': Mock(),
                'reset_button': Mock()
            }
        }
        mock_footer.return_value = Mock()
        mock_main.return_value = Mock()
        
        # Call function
        result = create_split_ui_components()
        
        # Verify all sections were called
        mock_ratio.assert_called_once_with({})
        mock_path.assert_called_once_with({})
        mock_advanced.assert_called_once_with({})
        
        # Verify containers were created
        mock_header.assert_called_once()
        mock_form.assert_called_once()
        mock_action.assert_called_once()
        mock_footer.assert_called_once()
        mock_main.assert_called_once()
        
        # Verify result structure
        assert 'main_container' in result
        assert 'header_container' in result
        assert 'form_container' in result
        assert 'action_container' in result
        assert 'footer_container' in result
        assert 'form_components' in result
        assert 'log_output' in result
        assert 'log_accordion' in result
        assert 'save_button' in result
        assert 'reset_button' in result
    
    @patch('smartcash.ui.dataset.split.components.split_ui.create_ratio_section')
    @patch('smartcash.ui.dataset.split.components.split_ui.create_path_section')
    @patch('smartcash.ui.dataset.split.components.split_ui.create_advanced_section')
    @patch('smartcash.ui.dataset.split.components.split_ui.create_main_container')
    @patch('smartcash.ui.dataset.split.components.split_ui.create_header_container')
    @patch('smartcash.ui.dataset.split.components.split_ui.create_form_container')
    @patch('smartcash.ui.dataset.split.components.split_ui.create_action_container')
    @patch('smartcash.ui.dataset.split.components.split_ui.create_footer_container')
    def test_create_split_ui_components_with_config(
        self,
        mock_footer,
        mock_action,
        mock_form,
        mock_header,
        mock_main,
        mock_advanced,
        mock_path,
        mock_ratio,
        mock_config
    ):
        """Test UI component creation with custom configuration."""
        # Setup mock returns
        mock_ratio.return_value = {'ratio_section': Mock()}
        mock_path.return_value = {'path_section': Mock()}
        mock_advanced.return_value = {'advanced_section': Mock()}
        mock_header.return_value = Mock()
        mock_form.return_value = {'container': Mock()}
        mock_action.return_value = {
            'container': Mock(),
            'buttons': {
                'save_button': Mock(),
                'reset_button': Mock()
            }
        }
        mock_footer.return_value = Mock()
        mock_main.return_value = Mock()
        
        # Call function with config
        result = create_split_ui_components(config=mock_config, theme='dark')
        
        # Verify sections were called with config
        mock_ratio.assert_called_once_with(mock_config)
        mock_path.assert_called_once_with(mock_config)
        mock_advanced.assert_called_once_with(mock_config)
        
        # Verify kwargs were passed through
        assert mock_form.call_args[1].get('theme') == 'dark'
        assert mock_action.call_args[1].get('theme') == 'dark'
        assert mock_main.call_args[1].get('theme') == 'dark'
    
    @patch('smartcash.ui.dataset.split.components.split_ui.create_ratio_section')
    @patch('smartcash.ui.dataset.split.components.split_ui.create_path_section')  
    @patch('smartcash.ui.dataset.split.components.split_ui.create_advanced_section')
    def test_create_split_ui_components_section_integration(
        self,
        mock_advanced,
        mock_path,
        mock_ratio
    ):
        """Test integration between different sections."""
        # Setup mock returns with specific components
        ratio_components = {
            'ratio_section': Mock(),
            'train_ratio': Mock(),
            'val_ratio': Mock(),
            'test_ratio': Mock()
        }
        path_components = {
            'path_section': Mock(),
            'train_dir': Mock(),
            'val_dir': Mock(),
            'test_dir': Mock()
        }
        advanced_components = {
            'advanced_section': Mock(),
            'seed': Mock(),
            'shuffle': Mock(),
            'stratify': Mock()
        }
        
        mock_ratio.return_value = ratio_components
        mock_path.return_value = path_components
        mock_advanced.return_value = advanced_components
        
        with patch('smartcash.ui.dataset.split.components.split_ui.create_main_container') as mock_main:
            with patch('smartcash.ui.dataset.split.components.split_ui.create_header_container'):
                with patch('smartcash.ui.dataset.split.components.split_ui.create_form_container') as mock_form:
                    with patch('smartcash.ui.dataset.split.components.split_ui.create_action_container') as mock_action:
                        with patch('smartcash.ui.dataset.split.components.split_ui.create_footer_container'):
                            mock_form.return_value = {'container': Mock()}
                            mock_action.return_value = {
                                'container': Mock(),
                                'buttons': {
                                    'save_button': Mock(),
                                    'reset_button': Mock()
                                }
                            }
                            mock_main.return_value = Mock()
                            
                            result = create_split_ui_components()
                            
                            # Verify all components are included in result
                            assert 'train_ratio' in result
                            assert 'val_ratio' in result
                            assert 'test_ratio' in result
                            assert 'train_dir' in result
                            assert 'val_dir' in result
                            assert 'test_dir' in result
                            assert 'seed' in result
                            assert 'shuffle' in result
                            assert 'stratify' in result
    
    @patch('smartcash.ui.dataset.split.components.split_ui.widgets')
    def test_log_accordion_creation(self, mock_widgets):
        """Test log accordion widget creation."""
        # Setup mock widgets
        mock_output = Mock()
        mock_accordion = Mock()
        mock_widgets.Output.return_value = mock_output
        mock_widgets.Accordion.return_value = mock_accordion
        
        with patch('smartcash.ui.dataset.split.components.split_ui.create_ratio_section'):
            with patch('smartcash.ui.dataset.split.components.split_ui.create_path_section'):
                with patch('smartcash.ui.dataset.split.components.split_ui.create_advanced_section'):
                    with patch('smartcash.ui.dataset.split.components.split_ui.create_main_container'):
                        with patch('smartcash.ui.dataset.split.components.split_ui.create_header_container'):
                            with patch('smartcash.ui.dataset.split.components.split_ui.create_form_container'):
                                with patch('smartcash.ui.dataset.split.components.split_ui.create_action_container'):
                                    with patch('smartcash.ui.dataset.split.components.split_ui.create_footer_container'):
                                        result = create_split_ui_components()
                                        
                                        # Verify log components were created
                                        mock_widgets.Output.assert_called_once()
                                        mock_widgets.Accordion.assert_called_once()
                                        
                                        # Verify log components are in result
                                        assert result['log_output'] == mock_output
                                        assert result['log_accordion'] == mock_accordion
                                        
                                        # Verify accordion title was set
                                        mock_accordion.set_title.assert_called_once_with(0, 'Log Messages')
    
    def test_action_button_configuration(self):
        """Test action button configuration."""
        with patch('smartcash.ui.dataset.split.components.split_ui.create_ratio_section'):
            with patch('smartcash.ui.dataset.split.components.split_ui.create_path_section'):
                with patch('smartcash.ui.dataset.split.components.split_ui.create_advanced_section'):
                    with patch('smartcash.ui.dataset.split.components.split_ui.create_main_container'):
                        with patch('smartcash.ui.dataset.split.components.split_ui.create_header_container'):
                            with patch('smartcash.ui.dataset.split.components.split_ui.create_form_container'):
                                with patch('smartcash.ui.dataset.split.components.split_ui.create_action_container') as mock_action:
                                    with patch('smartcash.ui.dataset.split.components.split_ui.create_footer_container'):
                                        mock_action.return_value = {
                                            'container': Mock(),
                                            'buttons': {
                                                'save_button': Mock(),
                                                'reset_button': Mock()
                                            }
                                        }
                                        
                                        create_split_ui_components()
                                        
                                        # Verify action container was called with correct buttons
                                        call_args = mock_action.call_args[1]
                                        buttons = call_args['buttons']
                                        
                                        assert len(buttons) == 2
                                        
                                        # Check save button configuration
                                        save_btn = next(b for b in buttons if b['button_id'] == 'save_button')
                                        assert save_btn['text'] == '💾 Save'
                                        assert save_btn['style'] == 'success'
                                        assert save_btn['tooltip'] == 'Save the current configuration'
                                        assert save_btn['order'] == 1
                                        
                                        # Check reset button configuration
                                        reset_btn = next(b for b in buttons if b['button_id'] == 'reset_button')
                                        assert reset_btn['text'] == '🔄 Reset'
                                        assert reset_btn['style'] == 'warning'
                                        assert reset_btn['tooltip'] == 'Reset to default values'
                                        assert reset_btn['order'] == 2
    
    def test_header_container_configuration(self):
        """Test header container configuration."""
        with patch('smartcash.ui.dataset.split.components.split_ui.create_ratio_section'):
            with patch('smartcash.ui.dataset.split.components.split_ui.create_path_section'):
                with patch('smartcash.ui.dataset.split.components.split_ui.create_advanced_section'):
                    with patch('smartcash.ui.dataset.split.components.split_ui.create_main_container'):
                        with patch('smartcash.ui.dataset.split.components.split_ui.create_header_container') as mock_header:
                            with patch('smartcash.ui.dataset.split.components.split_ui.create_form_container'):
                                with patch('smartcash.ui.dataset.split.components.split_ui.create_action_container'):
                                    with patch('smartcash.ui.dataset.split.components.split_ui.create_footer_container'):
                                        create_split_ui_components()
                                        
                                        # Verify header was called with correct parameters
                                        mock_header.assert_called_once_with(
                                            title="Dataset Split Configuration",
                                            description="Configure how to split your dataset into train/validation/test sets"
                                        )
    
    def test_form_container_configuration(self):
        """Test form container configuration."""
        with patch('smartcash.ui.dataset.split.components.split_ui.create_ratio_section'):
            with patch('smartcash.ui.dataset.split.components.split_ui.create_path_section'):
                with patch('smartcash.ui.dataset.split.components.split_ui.create_advanced_section'):
                    with patch('smartcash.ui.dataset.split.components.split_ui.create_main_container'):
                        with patch('smartcash.ui.dataset.split.components.split_ui.create_header_container'):
                            with patch('smartcash.ui.dataset.split.components.split_ui.create_form_container') as mock_form:
                                with patch('smartcash.ui.dataset.split.components.split_ui.create_action_container'):
                                    with patch('smartcash.ui.dataset.split.components.split_ui.create_footer_container'):
                                        from smartcash.ui.components.form_container import LayoutType
                                        
                                        mock_form.return_value = {'container': Mock()}
                                        
                                        create_split_ui_components()
                                        
                                        # Verify form container was called with correct layout
                                        call_args = mock_form.call_args[1]
                                        assert call_args['layout_type'] == LayoutType.COLUMN
                                        assert 'form_rows' in call_args
                                        
                                        # Verify form rows include all sections plus log accordion
                                        form_rows = call_args['form_rows']
                                        assert len(form_rows) == 4  # ratio, path, advanced, log_accordion
    
    def test_footer_container_configuration(self):
        """Test footer container configuration."""
        with patch('smartcash.ui.dataset.split.components.split_ui.create_ratio_section'):
            with patch('smartcash.ui.dataset.split.components.split_ui.create_path_section'):
                with patch('smartcash.ui.dataset.split.components.split_ui.create_advanced_section'):
                    with patch('smartcash.ui.dataset.split.components.split_ui.create_main_container'):
                        with patch('smartcash.ui.dataset.split.components.split_ui.create_header_container'):
                            with patch('smartcash.ui.dataset.split.components.split_ui.create_form_container'):
                                with patch('smartcash.ui.dataset.split.components.split_ui.create_action_container'):
                                    with patch('smartcash.ui.dataset.split.components.split_ui.create_footer_container') as mock_footer:
                                        create_split_ui_components()
                                        
                                        # Verify footer was called with correct parameters
                                        call_args = mock_footer.call_args[1]
                                        assert call_args['show_progress'] is False
                                        assert 'log_accordion' in call_args
    
    @patch('smartcash.ui.dataset.split.components.split_ui.create_ratio_section')
    def test_component_creation_error_handling(self, mock_ratio):
        """Test error handling during component creation."""
        mock_ratio.side_effect = Exception("Section creation failed")
        
        with pytest.raises(Exception, match="Section creation failed"):
            create_split_ui_components()
    
    def test_component_structure_completeness(self):
        """Test that all required components are present in the result."""
        with patch('smartcash.ui.dataset.split.components.split_ui.create_ratio_section') as mock_ratio:
            with patch('smartcash.ui.dataset.split.components.split_ui.create_path_section') as mock_path:
                with patch('smartcash.ui.dataset.split.components.split_ui.create_advanced_section') as mock_advanced:
                    with patch('smartcash.ui.dataset.split.components.split_ui.create_main_container') as mock_main:
                        with patch('smartcash.ui.dataset.split.components.split_ui.create_header_container') as mock_header:
                            with patch('smartcash.ui.dataset.split.components.split_ui.create_form_container') as mock_form:
                                with patch('smartcash.ui.dataset.split.components.split_ui.create_action_container') as mock_action:
                                    with patch('smartcash.ui.dataset.split.components.split_ui.create_footer_container') as mock_footer:
                                        # Setup comprehensive mock returns
                                        mock_ratio.return_value = {
                                            'ratio_section': Mock(),
                                            'train_ratio': Mock(),
                                            'val_ratio': Mock(),
                                            'test_ratio': Mock()
                                        }
                                        mock_path.return_value = {
                                            'path_section': Mock(),
                                            'train_dir': Mock(),
                                            'val_dir': Mock(),
                                            'test_dir': Mock()
                                        }
                                        mock_advanced.return_value = {
                                            'advanced_section': Mock(),
                                            'seed': Mock(),
                                            'shuffle': Mock(),
                                            'stratify': Mock()
                                        }
                                        mock_header.return_value = Mock()
                                        mock_form.return_value = {'container': Mock()}
                                        mock_action.return_value = {
                                            'container': Mock(),
                                            'buttons': {
                                                'save_button': Mock(),
                                                'reset_button': Mock()
                                            }
                                        }
                                        mock_footer.return_value = Mock()
                                        mock_main.return_value = Mock()
                                        
                                        result = create_split_ui_components()
                                        
                                        # Check all required components are present
                                        required_components = [
                                            'main_container',
                                            'header_container',
                                            'form_container',
                                            'action_container',
                                            'footer_container',
                                            'form_components',
                                            'form_rows',
                                            'log_output',
                                            'log_accordion',
                                            'save_button',
                                            'reset_button',
                                            'ratio_section',
                                            'path_section',
                                            'advanced_section',
                                            'train_ratio',
                                            'val_ratio',
                                            'test_ratio',
                                            'train_dir',
                                            'val_dir',
                                            'test_dir',
                                            'seed',
                                            'shuffle',
                                            'stratify'
                                        ]
                                        
                                        for component in required_components:
                                            assert component in result, f"Missing required component: {component}"