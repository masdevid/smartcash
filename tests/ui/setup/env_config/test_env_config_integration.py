"""End-to-end tests for environment configuration system."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, ANY, call, PropertyMock
from pathlib import Path
import json
import shutil
import tempfile

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from smartcash.ui.setup.env_config.env_config_initializer import EnvConfigInitializer
from smartcash.ui.setup.env_config.handlers.env_config_handler import EnvConfigHandler
from smartcash.ui.setup.env_config.handlers.config_handler import ConfigHandler
from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler


class TestEnvConfigIntegration(unittest.TestCase):
    """Integration tests for the environment configuration system."""
    
    def setUp(self):
        # Create a temporary directory for test configs
        self.test_config_dir = tempfile.mkdtemp(prefix='test_env_config_')
        
        # Create test UI components
        self.ui_components = {
            'ui': MagicMock(),
            'log_output': MagicMock(),
            'status_panel': MagicMock(),
            'setup_button': MagicMock(),
            'env_info_panel': MagicMock(),
            'config_editor': MagicMock(),
            'progress_bar': MagicMock(),
        }
        
        # Create mocks for all handler classes
        self.mock_config_handler = MagicMock(spec=ConfigHandler)
        self.mock_setup_handler = MagicMock(spec=SetupHandler)
        self.mock_env_config_handler = MagicMock(spec=EnvConfigHandler)
        
        # Set up handler relationships
        self.mock_env_config_handler.config_handler = self.mock_config_handler
        self.mock_env_config_handler.setup_handler = self.mock_setup_handler
        
        # Configure the mock to return our mock handlers
        self.mock_env_config_handler.initialize_environment = MagicMock()
        self.mock_env_config_handler.handle_setup_button_click = MagicMock()
        
        # Patch the handler classes to return our mocks
        self.config_handler_patcher = patch(
            'smartcash.ui.setup.env_config.handlers.config_handler.ConfigHandler',
            return_value=self.mock_config_handler
        )
        self.setup_handler_patcher = patch(
            'smartcash.ui.setup.env_config.handlers.setup_handler.SetupHandler',
            return_value=self.mock_setup_handler
        )
        
        # Patch the EnvConfigHandler class to return our mock instance
        self.env_config_handler_patcher = patch(
            'smartcash.ui.setup.env_config.handlers.env_config_handler.EnvConfigHandler',
            return_value=self.mock_env_config_handler
        )
        
        # Start all patches
        self.mock_config_handler_class = self.config_handler_patcher.start()
        self.mock_setup_handler_class = self.setup_handler_patcher.start()
        self.mock_env_config_handler_class = self.env_config_handler_patcher.start()
        
        # Patch the _init_handlers method to set up our mock handlers
        self.initializer = EnvConfigInitializer(config_handler_class=self.mock_config_handler_class)
        
        # Patch the required UI components
        self.ui_patcher = patch('smartcash.ui.setup.env_config.components.ui_components.create_env_config_ui')
        self.mock_create_ui = self.ui_patcher.start()
        self.mock_create_ui.return_value = self.ui_components
        
        # Patch the environment detector
        self.env_detector_patcher = patch('smartcash.ui.setup.env_config.components.env_info_panel.detect_environment_info')
        self.mock_detect_env = self.env_detector_patcher.start()
        self.mock_detect_env.return_value = {
            'os': 'test_os',
            'python': '3.9.0',
            'gpu': 'test_gpu',
            'memory': '16GB',
            'cuda': '11.7'
        }
        
        # Manually set the _env_config_handler attribute
        self.initializer._env_config_handler = self.mock_env_config_handler
    
    def tearDown(self):
        # Clean up the temporary directory
        try:
            shutil.rmtree(self.test_config_dir)
        except (OSError, IOError):
            pass
        
        # Stop all patches
        patches = [
            'ui_patcher',
            'env_detector_patcher',
            'config_handler_patcher',
            'setup_handler_patcher',
            'env_config_handler_patcher'
        ]
        
        for patch_name in patches:
            if hasattr(self, patch_name):
                getattr(self, patch_name).stop()
    
    @patch('smartcash.ui.utils.fallback_utils.create_fallback_ui')
    def test_initialization(self, mock_fallback_ui):
        """Test that the initializer properly initializes the environment config system."""
        # Setup the mock fallback UI
        mock_fallback_ui.return_value = MagicMock()
        
        # Initialize the system
        config = {
            'config_dir': str(self.test_config_dir),
            'auto_start': False,
            'ui_components': self.ui_components
        }
        
        # Call initialize with our patched handlers
        with patch.object(self.initializer, '_init_handlers') as mock_init_handlers:
            result = self.initializer.initialize(config=config)
            
            # Verify _init_handlers was called with the config
            mock_init_handlers.assert_called_once()
            
            # Verify the handler was initialized with correct parameters
            self.mock_env_config_handler_class.assert_called_once()
            self.assertTrue(hasattr(self.initializer, '_env_config_handler'))
            
            # Verify UI components were properly set up
            self.assertEqual(self.initializer.ui_components, self.ui_components)
            
            # Verify the UI creation was called with the right parameters
            self.mock_create_ui.assert_called_once()
            self.mock_detect_env.assert_called_once()
            
            # Verify the fallback UI was not called (since initialization should succeed)
            mock_fallback_ui.assert_not_called()
    
    @patch('smartcash.ui.utils.fallback_utils.create_fallback_ui')
    def test_environment_setup_flow(self, mock_fallback_ui):
        """Test the complete environment setup flow."""
        # Setup the mock fallback UI
        mock_fallback_ui.return_value = MagicMock()
        
        # Initialize the system
        config = {
            'config_dir': str(self.test_config_dir),
            'auto_start': True,
            'ui_components': self.ui_components
        }
        
        # Call initialize with our patched handlers
        with patch.object(self.initializer, '_init_handlers') as mock_init_handlers:
            self.initializer.initialize(config=config)
            
            # Verify _init_handlers was called
            mock_init_handlers.assert_called_once()
            
            # Verify the handler was initialized
            self.mock_env_config_handler_class.assert_called_once()
            self.assertTrue(hasattr(self.initializer, '_env_config_handler'))
            
            # Get the handler instance using the protected attribute
            env_config_handler = self.initializer._env_config_handler
            self.assertIsNotNone(env_config_handler, "EnvConfigHandler was not initialized")
            
            # Verify the setup button click handler was set up
            self.ui_components['setup_button'].on_click.assert_called_once()
            
            # Simulate button click if handler was set up
            if self.ui_components['setup_button'].on_click.called:
                click_args = self.ui_components['setup_button'].on_click.call_args
                click_handler = click_args[0][0]
                mock_button = MagicMock()
                click_handler(mock_button)
                
                # Verify the setup process was started through the handler
                env_config_handler.initialize_environment.assert_called_once()
                
            # Verify the fallback UI was not called (since initialization should succeed)
            mock_fallback_ui.assert_not_called()
    
    @patch('smartcash.ui.utils.fallback_utils.create_fallback_ui')
    def test_ui_components_after_init(self, mock_fallback_ui):
        """Test that UI components are properly updated after initialization."""
        # Setup the mock fallback UI
        mock_fallback_ui.return_value = MagicMock()
        
        # Initialize the system
        config = {
            'config_dir': str(self.test_config_dir),
            'auto_start': False,
            'ui_components': self.ui_components
        }
        
        # Call initialize with our patched handlers
        with patch.object(self.initializer, '_init_handlers') as mock_init_handlers:
            self.initializer.initialize(config=config)
            
            # Verify _init_handlers was called
            mock_init_handlers.assert_called_once()
            
            # Verify all required UI components exist
            required_components = ['ui', 'log_output', 'status_panel', 'setup_button', 
                                  'env_info_panel', 'config_editor', 'progress_bar']
            
            for component in required_components:
                self.assertIn(component, self.ui_components, 
                            f"Missing required UI component: {component}")
            
            # Test that components can be updated
            self.ui_components['setup_button'].description = 'Setup Environment'
            self.ui_components['status_panel'].value = 'Ready'
            
            # Verify the handler was properly initialized with UI components
            self.mock_env_config_handler_class.assert_called_once()
            self.assertEqual(self.initializer.ui_components, self.ui_components)
            
            # Verify the fallback UI was not called (since initialization should succeed)
            mock_fallback_ui.assert_not_called()


if __name__ == '__main__':
    unittest.main()
