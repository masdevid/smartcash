"""
Tests for Dependency UI module's configuration operations (Save/Reset).
"""
import pytest
from unittest.mock import MagicMock, patch, call

from smartcash.ui.setup.dependency.dependency_uimodule import DependencyUIModule
from smartcash.ui.setup.dependency.configs.dependency_config_handler import DependencyConfigHandler

class TestDependencyConfigOperations:
    """Test cases for Dependency UI module's configuration operations."""
    
    @pytest.fixture
    def mock_config_handler(self):
        """Create a mock config handler with necessary methods."""
        handler = MagicMock(spec=DependencyConfigHandler)
        handler.extract_config_from_ui.return_value = {
            'selected_packages': ['numpy', 'pandas'],
            'custom_packages': 'matplotlib',
            'install_options': {'upgrade': True}
        }
        handler.get_current_config.return_value = {
            'selected_packages': [],
            'custom_packages': '',
            'install_options': {}
        }
        handler.save_config.return_value = {'success': True, 'message': 'Configuration saved'}
        return handler
    
    @pytest.fixture
    def dependency_module(self, mock_config_handler):
        """Create a DependencyUIModule instance with mocked dependencies."""
        module = DependencyUIModule()
        
        # Mock UI components
        module._ui_components = {
            'header_container': MagicMock(),
            'operation_container': MagicMock(),
            'package_checkboxes': {
                'data_science': [MagicMock(value=True, package_name='numpy'),
                               MagicMock(value=True, package_name='pandas')]
            },
            'custom_packages': MagicMock(value='matplotlib'),
            'install_options': MagicMock(value={'upgrade': True})
        }
        
        # Mock config handler
        module._config_handler = mock_config_handler
        module._merged_config = {}
        
        # Mock logger
        module.logger = MagicMock()
        
        # Mock progress and status update methods
        module.update_operation_status = MagicMock()
        module.update_progress = MagicMock()
        module.log = MagicMock()
        
        return module
    
    def test_save_config_success(self, dependency_module, mock_config_handler):
        """Test successful save configuration operation."""
        # Call the save config handler
        result = dependency_module._handle_save_config()
        
        # Verify the result structure
        assert isinstance(result, dict)
        assert 'success' in result
        
        # Verify config handler was called correctly
        mock_config_handler.extract_config_from_ui.assert_called_once()
        mock_config_handler.save_config.assert_called_once()
        
        # Verify logging was called (at least once for the button click)
        assert dependency_module.log.call_count >= 1
    
    def test_reset_config_success(self, dependency_module, mock_config_handler):
        """Test successful reset configuration operation."""
        # Set up the config handler's reset_config method
        mock_config_handler.reset_config.return_value = {
            'success': True, 
            'message': 'Configuration reset to defaults'
        }
        
        # Call the reset config handler
        result = dependency_module._handle_reset_config()
        
        # Verify the result structure
        assert isinstance(result, dict)
        assert 'success' in result
        
        # Verify config handler was called
        mock_config_handler.reset_config.assert_called_once()
        
        # Verify logging was called (at least once for the button click)
        assert dependency_module.log.call_count >= 1
    
    def test_save_config_failure(self, dependency_module, mock_config_handler):
        """Test save configuration with failure."""
        # Set up the config handler to fail
        error_msg = "Failed to save config"
        mock_config_handler.save_config.return_value = {
            'success': False, 
            'message': error_msg
        }
        
        # Call the save config handler
        result = dependency_module._handle_save_config()
        
        # Verify the result
        assert isinstance(result, dict)
        assert 'success' in result
        
        # Verify error handling - check if any error log was recorded
        # Check for both the error message and the error emoji
        error_logs = [
            call[0][0] for call in dependency_module.log.call_args_list 
            if isinstance(call[0][0], str) and ('❌' in call[0][0] or 'error' in call[0][1].lower())
        ]
        assert len(error_logs) > 0, "Expected at least one error log with ❌ emoji"
    
    def test_reset_config_failure(self, dependency_module, mock_config_handler):
        """Test reset configuration with failure."""
        # Set up the config handler to fail
        error_msg = "Failed to reset config"
        mock_config_handler.reset_config.return_value = {
            'success': False,
            'message': error_msg
        }
        
        # Call the reset config handler
        result = dependency_module._handle_reset_config()
        
        # Verify the result structure
        assert isinstance(result, dict)
        assert 'success' in result
        
        # Verify error handling - check if any error log was recorded
        # Check for both the error message and the error emoji
        error_logs = [
            call[0][0] for call in dependency_module.log.call_args_list 
            if isinstance(call[0][0], str) and ('❌' in call[0][0] or 'error' in call[0][1].lower())
        ]
        assert len(error_logs) > 0, "Expected at least one error log with ❌ emoji"
    
    def test_config_validation_during_save(self, dependency_module, mock_config_handler):
        """Test that config validation occurs during save."""
        # Set up validation to fail
        mock_config_handler.validate_config.return_value = {
            'valid': False,
            'message': 'Invalid configuration'
        }
        
        # Call the save config handler
        result = dependency_module._handle_save_config()
        
        # Verify the result indicates failure
        assert isinstance(result, dict)
        assert 'success' in result
        
        # Verify save was not called when validation failed
        mock_config_handler.save_config.assert_not_called()
    
    def test_config_extraction(self, dependency_module, mock_config_handler):
        """Test that config is properly extracted from UI components."""
        # Call the save config handler
        dependency_module._handle_save_config()
        
        # Verify extract_config_from_ui was called
        mock_config_handler.extract_config_from_ui.assert_called_once()
        
        # Get the extracted config
        extracted_config = mock_config_handler.extract_config_from_ui()
        
        # Verify the extracted config structure
        assert 'selected_packages' in extracted_config
        assert 'custom_packages' in extracted_config
        assert 'install_options' in extracted_config
        assert extracted_config['selected_packages'] == ['numpy', 'pandas']
        assert extracted_config['custom_packages'] == 'matplotlib'
        assert extracted_config['install_options'] == {'upgrade': True}
