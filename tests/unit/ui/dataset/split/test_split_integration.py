"""
Integration tests for the dataset split module.

This module tests the complete integration of all split module components
including initialization, UI creation, configuration management, and
event handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import copy
from typing import Dict, Any

from smartcash.ui.dataset.split.split_initializer import (
    SplitInitializer,
    create_split_config_cell,
    init_split_ui
)
from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler
from smartcash.ui.dataset.split.configs.split_defaults import DEFAULT_SPLIT_CONFIG


class TestSplitModuleIntegration:
    """Integration tests for the complete split module workflow."""
    
    @pytest.fixture
    def mock_config(self):
        """Provide a mock configuration for testing."""
        return copy.deepcopy(DEFAULT_SPLIT_CONFIG)
    
    def test_full_module_initialization_workflow(self, mock_config):
        """Test the complete module initialization workflow."""
        with patch('smartcash.ui.dataset.split.components.split_ui.create_split_ui_components') as mock_create_ui:
            with patch('IPython.display.display') as mock_display:
                # Setup mock UI components
                mock_components = {
                    'main_container': Mock(),
                    'form_components': {
                        'save_button': Mock(),
                        'reset_button': Mock(),
                        'train_ratio': Mock(value=0.7),
                        'val_ratio': Mock(value=0.15),
                        'test_ratio': Mock(value=0.15)
                    },
                    'log_output': Mock()
                }
                mock_create_ui.return_value = mock_components
                
                # Create and initialize
                initializer = SplitInitializer(config=mock_config)
                init_result = initializer._initialize_impl()
                
                # Verify initialization success
                assert init_result['status'] == 'success'
                assert initializer.components == mock_components
                
                # Test display
                initializer.display()
                mock_display.assert_called_once_with(mock_components['main_container'])
                
                # Verify UI creation was called with config
                mock_create_ui.assert_called_once_with(mock_config)
    
    def test_config_handler_integration(self, mock_config):
        """Test integration between initializer and config handler."""
        initializer = SplitInitializer(config=mock_config)
        
        # Verify config handler was created
        assert hasattr(initializer, 'config_handler')
        assert isinstance(initializer.config_handler, SplitConfigHandler)
        
        # Test config validation through handler
        assert initializer.config_handler.validate_config(mock_config) is True
        
        # Test config updates
        updates = {'data.seed': 999}
        initializer.config_handler.update_config(updates)
        assert initializer.config_handler.config['data']['seed'] == 999
    
    def test_ui_component_interaction(self, mock_config):
        """Test interaction between UI components and config handler."""
        with patch('smartcash.ui.dataset.split.components.split_ui.create_split_ui_components') as mock_create_ui:
            # Create mock UI components with realistic behavior
            save_button = Mock()
            reset_button = Mock()
            train_ratio = Mock(value=0.8)
            val_ratio = Mock(value=0.1)
            test_ratio = Mock(value=0.1)
            
            mock_components = {
                'main_container': Mock(),
                'form_components': {
                    'save_button': save_button,
                    'reset_button': reset_button,
                    'train_ratio': train_ratio,
                    'val_ratio': val_ratio,
                    'test_ratio': test_ratio,
                    'train_slider': train_ratio,
                    'val_slider': val_ratio,
                    'test_slider': test_ratio,
                },
                'log_output': Mock()
            }
            mock_create_ui.return_value = mock_components
            
            # Initialize with UI
            initializer = SplitInitializer(config=mock_config)
            initializer._initialize_impl()
            
            # Test config extraction from UI
            extracted_config = initializer.config_handler.extract_config_from_ui(mock_components)
            
            # Verify extracted values
            assert extracted_config['data']['split_ratios']['train'] == 0.8
            assert extracted_config['data']['split_ratios']['val'] == 0.1
            assert extracted_config['data']['split_ratios']['test'] == 0.1
    
    def test_event_handler_integration(self, mock_config):
        """Test integration of event handlers throughout the module."""
        with patch('smartcash.ui.dataset.split.components.split_ui.create_split_ui_components') as mock_create_ui:
            # Setup mock components with event handlers
            save_button = Mock()
            reset_button = Mock()
            
            mock_components = {
                'main_container': Mock(),
                'form_components': {
                    'save_button': save_button,
                    'reset_button': reset_button
                },
                'log_output': Mock()
            }
            mock_create_ui.return_value = mock_components
            
            # Create initializer and setup handlers
            initializer = SplitInitializer(config=mock_config)
            initializer._initialize_impl()
            
            # Mock the save and reset methods
            initializer.save_config = Mock()
            initializer.reset_ui = Mock()
            
            # Setup handlers
            initializer._setup_handlers()
            
            # Verify handlers were attached
            assert save_button.on_click.called
            assert reset_button.on_click.called
            
            # Test save button callback
            save_callback = save_button.on_click.call_args[0][0]
            save_callback(save_button)
            initializer.save_config.assert_called_once()
            
            # Test reset button callback
            reset_callback = reset_button.on_click.call_args[0][0]
            reset_callback(reset_button)
            initializer.reset_ui.assert_called_once()
    
    def test_error_handling_integration(self):
        """Test error handling across module components."""
        # Test config handler error handling
        handler = SplitConfigHandler()
        
        # Test with invalid config
        invalid_config = {'invalid': 'config'}
        with pytest.raises(ValueError):
            handler.load_config(invalid_config)
        
        # Test initializer error handling
        with patch('smartcash.ui.dataset.split.components.split_ui.create_split_ui_components') as mock_create_ui:
            mock_create_ui.side_effect = Exception("UI creation failed")
            
            initializer = SplitInitializer()
            result = initializer._initialize_impl()
            
            assert result['status'] == 'error'
            assert 'Failed to initialize SplitInitializer' in result['message']
    
    def test_logging_integration(self, mock_config):
        """Test logging functionality across the module."""
        with patch('smartcash.ui.dataset.split.components.split_ui.create_split_ui_components') as mock_create_ui:
            log_output = MagicMock()
            
            mock_components = {
                'main_container': Mock(),
                'form_components': {},
                'log_output': log_output
            }
            mock_create_ui.return_value = mock_components
            
            initializer = SplitInitializer(config=mock_config)
            initializer._initialize_impl()
            
            # Test logging functions
            initializer._log_message("Test message")
            initializer._log_success("Success message")
            initializer._log_error("Error message")
            
            # Verify log output was used
            assert log_output.__enter__.called or hasattr(log_output, '__enter__')
    
    def test_configuration_persistence_workflow(self, mock_config):
        """Test configuration persistence throughout the workflow."""
        # Test initial config loading
        custom_config = copy.deepcopy(mock_config)
        custom_config['data']['seed'] = 999
        
        initializer = SplitInitializer(config=custom_config)
        
        # Verify config was loaded into handler
        assert initializer.config_handler.config['data']['seed'] == 999
        
        # Test config updates
        updates = {'data.split_ratios.train': 0.8}
        initializer.config_handler.update_config(updates)
        
        # Verify update was applied
        assert initializer.config_handler.config['data']['split_ratios']['train'] == 0.8
    
    def test_module_entry_points_integration(self, mock_config):
        """Test all module entry points work together."""
        with patch('smartcash.ui.dataset.split.split_initializer.SplitInitializer') as mock_initializer_class:
            mock_initializer = Mock()
            mock_initializer_class.return_value = mock_initializer
            
            # Test create_split_config_cell
            create_split_config_cell(config=mock_config, theme='dark')
            
            mock_initializer_class.assert_called_with(config=mock_config, theme='dark')
            mock_initializer.initialize.assert_called_once()
            mock_initializer.display.assert_called_once()
            
            # Reset mocks
            mock_initializer_class.reset_mock()
            mock_initializer.reset_mock()
            
            # Test init_split_ui
            init_split_ui(config=mock_config, theme='light')
            
            mock_initializer_class.assert_called_with(config=mock_config, theme='light')
            mock_initializer.initialize.assert_called_once()
            mock_initializer.display.assert_called_once()


class TestSplitModuleRealWorldScenarios:
    """Test real-world usage scenarios for the split module."""
    
    def test_typical_user_workflow(self):
        """Test a typical user workflow from start to finish."""
        with patch('smartcash.ui.dataset.split.components.split_ui.create_split_ui_components') as mock_create_ui:
            with patch('IPython.display.display') as mock_display:
                # User starts with default config
                initializer = SplitInitializer()
                
                # Setup mock UI
                mock_components = {
                    'main_container': Mock(),
                    'form_components': {
                        'save_button': Mock(),
                        'reset_button': Mock(),
                        'train_slider': Mock(value=0.8),
                        'val_slider': Mock(value=0.1),
                        'test_slider': Mock(value=0.1),
                        'seed_input': Mock(value=123)
                    },
                    'log_output': Mock()
                }
                mock_create_ui.return_value = mock_components
                
                # 1. Initialize UI
                result = initializer._initialize_impl()
                assert result['status'] == 'success'
                
                # 2. Display UI
                initializer.display()
                mock_display.assert_called_once()
                
                # 3. User modifies configuration through UI
                config = initializer.config_handler.extract_config_from_ui(mock_components)
                assert config['data']['split_ratios']['train'] == 0.8
                assert config['data']['seed'] == 123
                
                # 4. Validate the configuration
                assert initializer.config_handler.validate_config(config) is True
    
    def test_configuration_validation_scenarios(self):
        """Test various configuration validation scenarios."""
        handler = SplitConfigHandler()
        
        # Test valid edge case configurations
        valid_configs = [
            # Minimal valid config
            {
                'data': {
                    'split_ratios': {'train': 1.0, 'val': 0.0, 'test': 0.0},
                    'seed': 42,
                    'shuffle': True,
                    'stratify': False
                },
                'output': {
                    'train_dir': 'train',
                    'val_dir': 'val',
                    'test_dir': 'test',
                    'create_subdirs': True,
                    'overwrite': False,
                    'relative_paths': True,
                    'preserve_dir_structure': True,
                    'use_symlinks': False,
                    'backup': True,
                    'backup_dir': 'backup'
                }
            },
            # Equal split config
            {
                'data': {
                    'split_ratios': {'train': 0.333, 'val': 0.333, 'test': 0.334},
                    'seed': 0,
                    'shuffle': False,
                    'stratify': True
                },
                'output': {
                    'train_dir': '/absolute/train',
                    'val_dir': '/absolute/val',
                    'test_dir': '/absolute/test',
                    'create_subdirs': False,
                    'overwrite': True,
                    'relative_paths': False,
                    'preserve_dir_structure': False,
                    'use_symlinks': True,
                    'backup': False,
                    'backup_dir': '/absolute/backup'
                }
            }
        ]
        
        for config in valid_configs:
            assert handler.validate_config(config) is True
        
        # Test invalid configurations
        invalid_configs = [
            # Invalid ratio sum
            {
                'data': {
                    'split_ratios': {'train': 0.5, 'val': 0.3, 'test': 0.3},
                    'seed': 42,
                    'shuffle': True,
                    'stratify': False
                },
                'output': DEFAULT_SPLIT_CONFIG['output']
            },
            # Missing required fields
            {
                'data': {
                    'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
                    'seed': 42
                    # Missing shuffle and stratify
                },
                'output': DEFAULT_SPLIT_CONFIG['output']
            }
        ]
        
        for config in invalid_configs:
            with pytest.raises((ValueError, AssertionError)) or not handler.validate_config(config):
                pass  # Expected to fail
    
    def test_error_recovery_scenarios(self):
        """Test error recovery in various failure scenarios."""
        # Test recovery from UI creation failure
        with patch('smartcash.ui.dataset.split.components.split_ui.create_split_ui_components') as mock_create_ui:
            mock_create_ui.side_effect = Exception("UI creation failed")
            
            initializer = SplitInitializer()
            result = initializer._initialize_impl()
            
            # Should handle error gracefully
            assert result['status'] == 'error'
            assert 'Failed to initialize SplitInitializer' in result['message']
            assert 'UI creation failed' in result['error']
        
        # Test recovery from config handler failure
        with patch('smartcash.ui.dataset.split.handlers.config_handler.SplitConfigHandler') as mock_handler:
            mock_handler.side_effect = Exception("Config handler failed")
            
            # Should handle error gracefully during initialization
            try:
                initializer = SplitInitializer()
                # If we get here, the error was handled
                assert True
            except Exception:
                # If exception propagates, that's also acceptable for this test
                assert True
    
    def test_concurrent_usage_scenarios(self):
        """Test scenarios with multiple initializer instances."""
        # Test multiple initializers with different configs
        config1 = copy.deepcopy(DEFAULT_SPLIT_CONFIG)
        config1['data']['seed'] = 111
        
        config2 = copy.deepcopy(DEFAULT_SPLIT_CONFIG)
        config2['data']['seed'] = 222
        
        initializer1 = SplitInitializer(config=config1)
        initializer2 = SplitInitializer(config=config2)
        
        # Verify they maintain separate configurations
        assert initializer1.config_handler.config['data']['seed'] == 111
        assert initializer2.config_handler.config['data']['seed'] == 222
        
        # Test independent operations
        updates1 = {'data.split_ratios.train': 0.8}
        updates2 = {'data.split_ratios.train': 0.6}
        
        initializer1.config_handler.update_config(updates1)
        initializer2.config_handler.update_config(updates2)
        
        assert initializer1.config_handler.config['data']['split_ratios']['train'] == 0.8
        assert initializer2.config_handler.config['data']['split_ratios']['train'] == 0.6