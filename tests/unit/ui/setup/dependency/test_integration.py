"""
File: tests/unit/ui/setup/dependency/test_integration.py
Integration tests for dependency module
"""
import pytest
from unittest.mock import MagicMock, patch, call

class TestDependencyIntegration:
    """Integration tests for dependency module"""
    
    @patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyUIHandler')
    @patch('smartcash.ui.setup.dependency.dependency_initializer.OperationHandlerFactory')
    @patch('smartcash.ui.setup.dependency.dependency_initializer.create_dependency_ui_components')
    def test_full_workflow(self, mock_create_ui, mock_factory, mock_handler_class):
        """Test complete workflow from initialization to operation execution"""
        # Setup UI components mock
        mock_ui_components = {
            'main_container': MagicMock(),
            'status_panel': MagicMock(),
            'progress_bar': MagicMock()
        }
        mock_create_ui.return_value = mock_ui_components
        
        # Setup UI handler mock
        mock_ui_handler = MagicMock()
        mock_ui_handler.update_status = MagicMock()
        mock_handler_class.return_value = mock_ui_handler
        
        # Setup operation handler mock
        mock_operation_handler = MagicMock()
        mock_operation_handler.execute.return_value = {
            'status': 'success',
            'message': 'Packages installed successfully',
            'details': {
                'installed': ['numpy', 'pandas'],
                'failed': []
            }
        }
        
        mock_factory.create_handler.return_value = mock_operation_handler
        
        # Import here to ensure mocks are in place
        from smartcash.ui.setup.dependency.dependency_initializer import DependencyInitializer
        
        # Initialize
        initializer = DependencyInitializer()
        
        # Test data
        test_config = {
            'packages': ['numpy', 'pandas'],
            'operation': 'install',
            'auto_confirm': True
        }
        
        # Execute
        result = initializer.initialize(config=test_config)
        
        # Verify initialization
        assert result['status'] == 'success'
        assert 'ui' in result
        assert 'handlers' in result
        
        # Verify UI components were created
        mock_create_ui.assert_called_once()
        
        # Verify operation execution
        mock_factory.create_handler.assert_called_once_with('install')
        mock_operation_handler.execute.assert_called_once_with(
            ['numpy', 'pandas'], 
            auto_confirm=True
        )
        
        # Verify UI updates
        assert mock_ui_handler.update_status.call_count >= 1
        
    @patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyUIHandler')
    @patch('smartcash.ui.setup.dependency.dependency_initializer.create_dependency_ui_components')
    def test_error_handling(self, mock_create_ui, mock_handler_class):
        """Test error handling during operation execution"""
        # Setup UI components mock
        mock_ui_components = {
            'main_container': MagicMock(),
            'status_panel': MagicMock(),
            'progress_bar': MagicMock()
        }
        mock_create_ui.return_value = mock_ui_components
        
        # Setup UI handler mock
        mock_ui_handler = MagicMock()
        mock_ui_handler.update_status = MagicMock()
        mock_ui_handler.show_error = MagicMock()
        mock_handler_class.return_value = mock_ui_handler
        
        # Mock operation to raise an exception
        with patch('smartcash.ui.setup.dependency.dependency_initializer.OperationHandlerFactory') as mock_factory:
            mock_operation_handler = MagicMock()
            test_error = Exception("Test error")
            mock_operation_handler.execute.side_effect = test_error
            mock_factory.create_handler.return_value = mock_operation_handler
            
            # Import here to ensure mocks are in place
            from smartcash.ui.setup.dependency.dependency_initializer import DependencyInitializer
            
            # Initialize
            initializer = DependencyInitializer()
            
            # Execute with test data that will cause an error
            test_config = {
                'packages': ['nonexistent-package'],
                'operation': 'install',
                'auto_confirm': False
            }
            
            result = initializer.initialize(config=test_config)
            
            # Verify error handling
            assert result['status'] == 'error'
            assert 'error' in result
            
            # Verify error was shown in UI
            mock_ui_handler.show_error.assert_called_once()
            
            # Verify operation handler was still called
            mock_factory.create_handler.assert_called_once_with('install')
            mock_operation_handler.execute.assert_called_once()
    
    def test_ui_components_initialization(self, dependency_initializer, mock_ui_handler):
        """Test that UI components are properly initialized"""
        # Setup
        test_config = {
            'packages': ['numpy'],
            'operation': 'install'
        }
        
        # Mock UI components creation
        mock_ui_components = {
            'main_container': MagicMock(),
            'status_panel': MagicMock(),
            'progress_bar': MagicMock()
        }
        
        with patch('smartcash.ui.setup.dependency.dependency_initializer.create_dependency_ui_components', 
                  return_value=mock_ui_components) as mock_create_ui:
            # Execute
            result = dependency_initializer.initialize(config=test_config)
            
            # Verify
            mock_create_ui.assert_called_once()
            assert dependency_initializer._ui_components == mock_ui_components
            assert result['ui'] == mock_ui_components['main_container']
