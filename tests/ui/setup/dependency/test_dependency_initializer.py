"""
Tests for the dependency initializer module.

This module contains unit tests for the DependencyInitializer class and related functionality.
"""

# Standard library imports
import unittest
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any
from smartcash.common.logger import SmartCashLogger
import pytest

# Test imports
from smartcash.ui.setup.dependency.dependency_initializer import (
    DependencyInitializer,
    initialize_dependency_ui,
    _dependency_initializer
)
from smartcash.common.exceptions import (
    SmartCashError,
    ConfigError
)

# Test data
SAMPLE_CONFIG = {
    'module_name': 'test_module',
    'dependencies': {
        'numpy': {'version': '1.21.0', 'required': True},
        'pytest': {'version': '6.2.5', 'required': False}
    },
    'auto_update': True,
    'check_on_startup': True
}

class TestDependencyInitializer(unittest.TestCase):
    """Test cases for DependencyInitializer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.initializer = DependencyInitializer()
        self.mock_ui_components = {
            'ui': MagicMock(),
            'install_button': MagicMock(),
            'analyze_button': MagicMock(),
            'status_check_button': MagicMock(),
            'progress_tracker': MagicMock(),
            'status_panel': MagicMock(),
            'package_selector': MagicMock(),
            'custom_packages': MagicMock(),
            'config': {}
        }
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        # Setup
        expected_config = {
            'module_name': 'dependency',
            'dependencies': {
                'torch': {'version': 'latest', 'required': True},
                'torchvision': {'version': 'latest', 'required': True},
                'ultralytics': {'version': 'latest', 'required': True}
            },
            'auto_update': True,
            'check_on_startup': True
        }
        
        # Create a mock logger
        mock_logger = MagicMock()
        
        # Create a config provider function
        def mock_config_provider():
            return expected_config
        
        # Create a new instance with our mocks
        initializer = DependencyInitializer(logger=mock_logger, config_provider=mock_config_provider)
        
        # Execute
        result = initializer._get_default_config()
        
        # Verify
        self.assertEqual(result, expected_config)
        mock_logger.warning.assert_not_called()
        self.assertIn('module_name', result)
        self.assertIn('dependencies', result)
        self.assertIn('torch', result['dependencies'])
        self.assertTrue(result['auto_update'])
        self.assertTrue(result['check_on_startup'])
    
    def test_get_default_config_fallback(self):
        """Test fallback when default config cannot be loaded."""
        # Setup
        mock_logger = MagicMock()
        
        # Configure the mock to raise an ImportError
        def mock_config_provider():
            raise ImportError("Test error")
        
        # Create a new instance with our mocks
        initializer = DependencyInitializer(logger=mock_logger, config_provider=mock_config_provider)
        
        # Execute
        result = initializer._get_default_config()
        
        # Verify the fallback config is returned
        self.assertIn('dependencies', result)
        self.assertIn('torch', result['dependencies'])
        self.assertIn('torchvision', result['dependencies'])
        self.assertIn('ultralytics', result['dependencies'])
        self.assertTrue(result['auto_update'])
        self.assertTrue(result['check_on_startup'])
        
        # Verify the warning was logged
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("Could not load default config, using fallback", warning_msg)
    
    def test_get_ui_root(self):
        """Test getting UI root component."""
        # Setup
        ui_components = {'ui': 'test_ui'}
        
        # Execute
        result = self.initializer._get_ui_root(ui_components)
        
        # Verify
        self.assertEqual(result, 'test_ui')
    
    def test_get_ui_root_missing(self):
        """Test error when UI root component is missing."""
        # Setup
        ui_components = {}
        
        # Execute & Verify
        with self.assertRaises(KeyError):
            self.initializer._get_ui_root(ui_components)
    
    def test_create_ui_components(self):
        """Test UI component creation."""
        # Setup
        expected_ui = {'ui': MagicMock(), 'config': SAMPLE_CONFIG}
        
        # Create a mock for the module and function
        mock_ui_components = MagicMock()
        mock_ui_components.create_dependency_main_ui.return_value = expected_ui
        
        # Patch the import in the dependency_initializer module
        with patch.dict('sys.modules', {'smartcash.ui.setup.dependency.components.ui_components': mock_ui_components}):
            # Re-import the module to apply the patch
            import importlib
            import sys
            if 'smartcash.ui.setup.dependency.dependency_initializer' in sys.modules:
                importlib.reload(sys.modules['smartcash.ui.setup.dependency.dependency_initializer'])
            
            # Execute
            result = self.initializer._create_ui_components(SAMPLE_CONFIG)
            
            # Verify
            self.assertEqual(result, expected_ui)
            mock_ui_components.create_dependency_main_ui.assert_called_once_with(SAMPLE_CONFIG)
    
    @patch('smartcash.ui.setup.dependency.dependency_initializer.get_logger')
    def test_initialize_ui(self, mock_get_logger):
        """Test successful UI initialization."""
        # Setup
        # Create a mock logger
        mock_logger = MagicMock(spec=SmartCashLogger)
        mock_get_logger.return_value = mock_logger
        
        # Create a new initializer with the mock logger
        initializer = DependencyInitializer(logger=mock_logger)
        
        # Configure mocks on the instance
        expected_ui = self.mock_ui_components
        initializer._create_ui_components = MagicMock(return_value=expected_ui)
        initializer._setup_handlers = MagicMock(return_value=expected_ui)
        initializer._after_init_checks = MagicMock()
        
        # Execute
        result = initializer.initialize_ui(SAMPLE_CONFIG)
        
        # Verify the result and the order of method calls
        self.assertEqual(result, expected_ui)
        
        # Verify the order of calls
        initializer._create_ui_components.assert_called_once_with(SAMPLE_CONFIG)
        initializer._setup_handlers.assert_called_once_with(expected_ui, SAMPLE_CONFIG)
        initializer._after_init_checks.assert_called_once_with(ui_components=expected_ui, config=SAMPLE_CONFIG)
    
    @patch('smartcash.ui.setup.dependency.dependency_initializer.get_logger')
    def test_initialize_ui_error(self, mock_get_logger):
        """Test error handling during UI initialization."""
        # Setup
        # Create a mock logger that matches SmartCashLogger's interface
        mock_logger = MagicMock(spec=SmartCashLogger)
        mock_get_logger.return_value = mock_logger
        
        # Create a new initializer with the mock logger
        initializer = DependencyInitializer(logger=mock_logger)
        
        # Configure mocks on the instance
        test_exception = Exception("UI creation failed")
        initializer._create_ui_components = MagicMock(side_effect=test_exception)
        initializer._setup_handlers = MagicMock()
        initializer._after_init_checks = MagicMock()
        
        # Execute and verify the exception is raised
        with self.assertRaises(RuntimeError) as context:
            initializer.initialize_ui(SAMPLE_CONFIG)
            
        # Verify the error message and that the exception was chained
        self.assertIn("Failed to initialize dependency UI: UI creation failed", str(context.exception))
        self.assertIsInstance(context.exception.__cause__, Exception)
        self.assertEqual(str(context.exception.__cause__), "UI creation failed")
        
        # Verify the error was logged
        mock_logger.error.assert_called_once_with("Failed to initialize dependency UI: UI creation failed")
        
        # Verify other methods were not called
        initializer._setup_handlers.assert_not_called()
        initializer._after_init_checks.assert_not_called()
    
    @patch('smartcash.ui.setup.dependency.dependency_initializer.get_logger')
    def test_initialize(self, mock_get_logger):
        """Test full initialization process."""
        # Setup
        # Create a mock logger
        mock_logger = MagicMock(spec=SmartCashLogger)
        mock_get_logger.return_value = mock_logger
        
        # Create a new initializer with the mock logger
        initializer = DependencyInitializer(logger=mock_logger)
        
        # Configure mocks on the instance
        ui_components = self.mock_ui_components
        mock_handlers = {'handler1': MagicMock(), 'handler2': MagicMock()}
        
        # Mock the instance methods
        initializer.initialize_ui = MagicMock(return_value=ui_components)
        initializer._setup_handlers = MagicMock(return_value=mock_handlers)
        initializer._after_init_checks = MagicMock()
        
        # Execute
        result = initializer.initialize(SAMPLE_CONFIG)
        
        # Verify the result
        self.assertIsInstance(result, dict)
        self.assertIn('ui', result)
        self.assertIn('handlers', result)
        self.assertEqual(result['ui'], ui_components)
        self.assertEqual(result['handlers'], mock_handlers)
        
        # Verify the method calls
        initializer.initialize_ui.assert_called_once_with(SAMPLE_CONFIG)
        initializer._setup_handlers.assert_called_once_with(ui_components, SAMPLE_CONFIG)
        initializer._after_init_checks.assert_called_once_with(ui_components=ui_components, config=SAMPLE_CONFIG)

    @patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyValidator')
    @patch('smartcash.ui.setup.dependency.utils.package.status.analyze_packages')
    def test_after_init_checks_success(self, mock_analyze_packages, mock_dependency_validator_class):
        """Test _after_init_checks with successful package analysis."""
        # Setup
        mock_ui_components = {
            'status_panel': MagicMock(),
            'package_selector': MagicMock()
        }
        
        mock_config = {
            'dependencies': {
                'numpy': {'required': True, 'version': '1.21.0'},
                'pytest': {'required': False}
            },
            'install_options': {
                'force_reinstall': False,
                'upgrade': True
            }
        }
        
        # Mock the DependencyValidator instance and its validate_config method
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_config.return_value = {
            'valid': True,
            'issues': []
        }
        mock_dependency_validator_class.return_value = mock_validator_instance
        
        # Mock analyze_packages to return a status dictionary with the expected structure
        mock_analyze_packages.return_value = {
            'numpy': {'installed': True, 'version': '1.21.0'},
            'pytest': {'installed': False}
        }
        
        # Execute
        self.initializer._after_init_checks(
            ui_components=mock_ui_components,
            config=mock_config
        )
        
        # Verify the validator was created and validate_config was called
        mock_dependency_validator_class.assert_called_once()
        mock_validator_instance.validate_config.assert_called_once_with(mock_config)
        
        # Verify analyze_packages was called with the correct arguments
        mock_analyze_packages.assert_called_once_with(mock_config)
        
        # Verify status panel was updated with the correct status text
        mock_ui_components['status_panel'].update.assert_called_once()
        
        # Get the actual call arguments
        call_args = mock_ui_components['status_panel'].update.call_args[1]
        status_text = call_args['value']
        
        # Check that the status text contains the expected package statuses
        self.assertIn('numpy: ✓', status_text)
        self.assertIn('pytest: ✗', status_text)

    @patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyValidator')
    @patch('smartcash.ui.setup.dependency.utils.package.status.analyze_packages')
    def test_after_init_checks_error(self, mock_analyze_packages, mock_dependency_validator_class):
        """Test _after_init_checks with error during analysis."""
        # Setup
        mock_ui_components = {
            'status_panel': MagicMock(),
            'package_selector': MagicMock()
        }
        mock_config = {
            'dependencies': {
                'numpy': {'required': True, 'version': '1.21.0'}
            },
            'install_options': {
                'force_reinstall': False,
                'upgrade': True
            }
        }
        
        # Mock the DependencyValidator instance and its validate_config method
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_config.return_value = {
            'valid': True,
            'issues': []
        }
        mock_dependency_validator_class.return_value = mock_validator_instance
        
        # Configure the mock to raise an exception
        mock_analyze_packages.side_effect = Exception("Analysis failed")
        
        # Execute and verify exception is raised
        with self.assertRaises(RuntimeError) as context:
            self.initializer._after_init_checks(
                ui_components=mock_ui_components,
                config=mock_config
            )
        
        # Verify the validator was created and validate_config was called
        mock_dependency_validator_class.assert_called_once()
        mock_validator_instance.validate_config.assert_called_once_with(mock_config)
        
        # Verify analyze_packages was called with the correct arguments
        mock_analyze_packages.assert_called_once_with(mock_config)
        
        # Verify status panel was updated with the error message
        mock_ui_components['status_panel'].update.assert_called_once()
        call_args = mock_ui_components['status_panel'].update.call_args[1]
        self.assertIn('Error during initialization: Analysis failed', call_args['value'])
        
        # Verify error message
        self.assertIn("Post-initialization check failed: Analysis failed", str(context.exception))
        
        # Verify status panel was updated with error message
        mock_ui_components['status_panel'].update.assert_called_once()
        call_args = mock_ui_components['status_panel'].update.call_args[1]
        self.assertIn('value', call_args)
        self.assertIn('Error during initialization: Analysis failed', call_args['value'])

    @patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyValidator')
    @patch('smartcash.ui.setup.dependency.utils.package.status.analyze_packages')
    def test_after_init_checks_no_ui_components(self, mock_analyze_packages, mock_dependency_validator_class):
        """Test _after_init_checks with missing UI components."""
        # Setup
        mock_config = {
            'dependencies': {},
            'install_options': {}
        }
        
        # Mock the DependencyValidator instance and its validate_config method
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_config.return_value = {
            'valid': True,
            'issues': [],
            'config': mock_config
        }
        mock_dependency_validator_class.return_value = mock_validator_instance
        
        # Mock analyze_packages to return an empty status dictionary
        mock_analyze_packages.return_value = {}
        
        # This should not raise any exceptions even with missing components
        self.initializer._after_init_checks(
            ui_components={},
            config=mock_config
        )


class TestInitializeDependencyUI(unittest.TestCase):
    """Test cases for the initialize_dependency_ui function."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Import the module and save the original state
        import smartcash.ui.setup.dependency.dependency_initializer as di_module
        self.di_module = di_module
        
        # Save the original singleton instance to restore in tearDown
        self.original_initializer = getattr(di_module, '_dependency_initializer', None)
        
        # Create test mocks
        self.mock_initializer = MagicMock()
        self.mock_ui_components = {
            'ui': MagicMock(),
            'install_button': MagicMock(),
            'analyze_button': MagicMock(),
            'status_check_button': MagicMock(),
            'progress_tracker': MagicMock(),
            'status_panel': MagicMock(),
            'package_selector': MagicMock(),
            'custom_packages': MagicMock(),
            'config': SAMPLE_CONFIG
        }
        self.mock_initializer.initialize_ui.return_value = self.mock_ui_components
        
        # Clear the singleton instance
        self.di_module._dependency_initializer = None
    
    def tearDown(self):
        """Clean up after each test method."""
        # Restore the original singleton instance
        self.di_module._dependency_initializer = self.original_initializer
    
    @patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyInitializer')
    @patch('smartcash.ui.setup.dependency.dependency_initializer.get_logger')
    def test_initialize_dependency_ui_success(self, mock_get_logger, mock_initializer_class):
        """Test successful initialization of dependency UI."""
        # Setup mocks
        mock_logger = MagicMock(spec=SmartCashLogger)
        mock_get_logger.return_value = mock_logger
        
        # Create a mock initializer instance
        mock_initializer = MagicMock()
        mock_initializer.initialize_ui.return_value = self.mock_ui_components
        mock_initializer_class.return_value = mock_initializer
        
        # Ensure the singleton is cleared before the test
        self.di_module._dependency_initializer = None
        
        # Call the function
        result = self.di_module.initialize_dependency_ui(SAMPLE_CONFIG)
        
        # Verify the returned UI components
        self.assertEqual(result, self.mock_ui_components)
        
        # Verify the module-level variable was set
        self.assertIs(self.di_module._dependency_initializer, mock_initializer)
        
        # Verify the initializer was created with the correct logger
        mock_initializer_class.assert_called_once_with(logger=mock_logger)
        
        # Verify initialize_ui was called with the correct config
        mock_initializer.initialize_ui.assert_called_once_with(SAMPLE_CONFIG)
        
        # Verify success was logged
        mock_logger.info.assert_called_once_with("Dependency UI initialized successfully")
    
    @patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyInitializer')
    @patch('smartcash.ui.setup.dependency.dependency_initializer.get_logger')
    def test_initialize_dependency_ui_error(self, mock_get_logger, mock_initializer_class):
        """Test error handling during dependency UI initialization."""
        # Setup mocks
        mock_logger = MagicMock(spec=SmartCashLogger)
        mock_get_logger.return_value = mock_logger
        
        # Create a mock initializer that will raise an exception
        mock_initializer = MagicMock()
        mock_initializer.initialize_ui.side_effect = Exception("UI creation failed")
        mock_initializer_class.return_value = mock_initializer
        
        # Ensure the singleton is cleared before the test
        self.di_module._dependency_initializer = None
        
        # Execute & Verify
        with self.assertRaises(RuntimeError) as context:
            self.di_module.initialize_dependency_ui(SAMPLE_CONFIG)
        
        # Verify the error was raised with the correct message
        self.assertIn("Failed to initialize dependency UI: UI creation failed", str(context.exception))
        
        # Verify initializer was created with the logger
        mock_initializer_class.assert_called_once_with(logger=mock_logger)
        
        # Verify error was logged
        mock_logger.error.assert_called_once()
        error_message = str(mock_logger.error.call_args[0][0])
        self.assertIn("Failed to initialize dependency UI: UI creation failed", error_message)
        
        # Verify singleton instance was not set
        self.assertIsNone(self.di_module._dependency_initializer)
    
    @patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyInitializer')
    @patch('smartcash.ui.setup.dependency.dependency_initializer.get_logger')
    def test_initialize_dependency_ui_singleton(self, mock_get_logger, mock_initializer_class):
        """Test that only one instance of DependencyInitializer is created."""
        # Setup mocks
        mock_logger = MagicMock(spec=SmartCashLogger)
        mock_get_logger.return_value = mock_logger
        
        # Create a mock initializer instance
        mock_initializer = MagicMock()
        mock_initializer.initialize_ui.return_value = self.mock_ui_components
        mock_initializer_class.return_value = mock_initializer
        
        # Ensure the singleton is cleared before the test
        self.di_module._dependency_initializer = None
        
        # First call
        result1 = self.di_module.initialize_dependency_ui(SAMPLE_CONFIG)
        
        # Verify first call
        self.assertEqual(result1, self.mock_ui_components)
        mock_initializer_class.assert_called_once()  # Instance created
        mock_initializer.initialize_ui.assert_called_once_with(SAMPLE_CONFIG)
        
        # Reset call counts for the next test
        mock_initializer.initialize_ui.reset_mock()
        
        # Second call with different config
        different_config = {'test': 'different'}
        result2 = self.di_module.initialize_dependency_ui(different_config)
        
        # Verify second call
        self.assertEqual(result2, self.mock_ui_components)
        mock_initializer_class.assert_called_once()  # No new instance created
        mock_initializer.initialize_ui.assert_called_once_with(different_config)  # But initialize_ui called with new config
        
        # Verify the singleton instance is set
        self.assertIsNotNone(self.di_module._dependency_initializer)
        self.assertIs(self.di_module._dependency_initializer, mock_initializer)
    
    def test_create_package_selector(self):
        """Test package selector creation."""
        # Skip this test for now as it requires additional setup
        self.skipTest("Skipping package selector test - requires additional setup")
        
        # The following is kept as a reference for future implementation
        # when the package selector component is properly set up
        try:
            from smartcash.ui.setup.dependency.components.ui_package_selector import (
                create_package_selector_grid
            )
            
            # Setup test data
            test_data = {
                'selected_packages': ['test_pkg'],
                'on_change': MagicMock()
            }
            
            # Execute
            result = create_package_selector_grid(**test_data)
            
            # Verify
            self.assertIsNotNone(result)
            self.assertIn('container', result)
            
        except ImportError:
            self.skipTest("Package selector module not available")


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests for dependency management."""
    
    @patch('smartcash.ui.setup.dependency.dependency_initializer.DependencyInitializer')
    def test_complete_flow(self, mock_initializer_class):
        """Test complete dependency management flow."""
        # Setup mocks
        mock_ui_components = {
            'ui': MagicMock(),
            'install_button': MagicMock(),
            'analyze_button': MagicMock(),
            'status_check_button': MagicMock(),
            'progress_tracker': MagicMock(),
            'status_panel': MagicMock(),
            'package_selector': MagicMock(),
            'custom_packages': MagicMock(),
            'config': SAMPLE_CONFIG
        }
        
        # Create a mock initializer
        mock_initializer = MagicMock()
        mock_initializer.initialize_ui.return_value = mock_ui_components
        mock_initializer_class.return_value = mock_initializer
        
        # Import here to avoid module-level side effects
        from smartcash.ui.setup.dependency.dependency_initializer import initialize_dependency_ui
        
        # Clear any existing singleton instance
        global _dependency_initializer
        _dependency_initializer = None
        
        # Execute
        result = initialize_dependency_ui(SAMPLE_CONFIG)
        
        # Verify initialization
        self.assertEqual(result, mock_ui_components)
        mock_initializer_class.assert_called_once()
        mock_initializer.initialize_ui.assert_called_once_with(SAMPLE_CONFIG)
        
        # Test singleton behavior
        mock_initializer_class.reset_mock()
        mock_initializer.initialize_ui.reset_mock()
        
        # Call again - should use existing instance
        result2 = initialize_dependency_ui(SAMPLE_CONFIG)
        mock_initializer_class.assert_not_called()
        mock_initializer.initialize_ui.assert_called_once_with(SAMPLE_CONFIG)


if __name__ == '__main__':
    unittest.main()
