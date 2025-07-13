"""
Tests for the VisualizationUIModule class.
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.dataset.visualization import VisualizationUIModule, create_visualization_module


class TestVisualizationUIModule(unittest.TestCase):
    """Test cases for VisualizationUIModule."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "dataset_path": "/test/path",
            "visualization_type": "histogram"
        }
        self.module = VisualizationUIModule(config=self.config)
        
    def test_initialization(self):
        """Test module initialization."""
        self.assertEqual(self.module.module_name, "visualization")
        self.assertEqual(self.module.parent_module, "dataset")
        self.assertEqual(self.module.get_config("dataset_path"), "/test/path")
        
    @patch('smartcash.ui.dataset.visualization.visualization_module.VisualizationUIHandler')
    @patch('smartcash.ui.dataset.visualization.components.visualization_ui.create_visualization_ui')
    def test_setup_components(self, mock_create_ui, mock_handler_class):
        """Test setting up UI components."""
        # Mock UI components
        mock_ui = {
            "header_container": widgets.VBox(),
            "form_container": widgets.VBox(),
            "action_container": widgets.VBox(),
            "summary_container": widgets.VBox(),
            "operation_container": widgets.VBox(),
            "footer_container": widgets.VBox(),
            "main_container": widgets.VBox()
        }
        mock_create_ui.return_value = mock_ui
        
        # Mock handler
        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler
        
        # Call the method
        self.module._setup_components()
        
        # Get the actual config that was passed to create_visualization_ui
        actual_config = mock_create_ui.call_args[0][0]
        
        # Assert the config contains our test values
        self.assertEqual(actual_config['dataset_path'], "/test/path")
        self.assertEqual(actual_config['visualization_type'], "histogram")
        
        # Assert the handler was created with the mock UI
        mock_handler_class.assert_called_once_with(
            ui_components=mock_ui,
            logger=self.module.logger
        )
        
        # Check components were registered
        self.assertIsNotNone(self.module.get_component("main_container"))
        self.assertIsNotNone(self.module.get_component("header_container"))
        
    @patch.object(VisualizationUIModule, 'update_status')
    def test_analyze_dataset(self, mock_update_status):
        """Test dataset analysis operation."""
        # Call the method
        result = self.module._analyze_dataset()
        
        # Assertions
        self.assertEqual(result["status"], "success")
        mock_update_status.assert_called_with("Analysis completed", "success")
        
    @patch.object(VisualizationUIModule, 'update_status')
    def test_export_visualization(self, mock_update_status):
        """Test export visualization operation."""
        # Call the method
        result = self.module._export_visualization()
        
        # Assertions
        self.assertEqual(result["status"], "success")
        mock_update_status.assert_called_with("Export completed", "success")
        
    @patch.object(VisualizationUIModule, 'update_status')
    def test_compare_datasets(self, mock_update_status):
        """Test compare datasets operation."""
        # Call the method
        result = self.module._compare_datasets()
        
        # Assertions
        self.assertEqual(result["status"], "success")
        mock_update_status.assert_called_with("Comparison completed", "success")
        
    @patch('IPython.display.display')
    def test_display(self, mock_display):
        """Test display method."""
        # Create a mock main container
        main_container = widgets.VBox()
        
        # Test successful display
        with patch.object(self.module, 'get_component', return_value=main_container):
            result = self.module.display()
            
            # Assert the display function was called with our container
            mock_display.assert_called_once()
            
            # Get the actual argument that was passed to display
            displayed_widget = mock_display.call_args[0][0]
            self.assertIs(displayed_widget, main_container)
            
            # Check the return value
            self.assertIs(result, main_container)
        
        # Test error case - no main container
        with patch.object(self.module, 'get_component', return_value=None):
            with patch.object(self.module.logger, 'warning') as mock_warning:
                result = self.module.display()
                
                # Verify the warning was logged
                mock_warning.assert_called_once_with(
                    "Main container not found in visualization module"
                )
                
                # Verify display wasn't called
                mock_display.assert_called_once()  # Still only called once from first test
                self.assertIsNone(result)


class TestModuleFactory(unittest.TestCase):
    """Test module factory functions."""
    
    @patch('smartcash.ui.core.ui_module_factory.UIModuleFactory.create_module')
    def test_create_visualization_module(self, mock_create):
        """Test create_visualization_module factory function."""
        config = {"test": "config"}
        create_visualization_module(config=config)
        
        # Assert the factory was called with correct parameters
        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        self.assertEqual(kwargs["module_name"], "visualization")
        self.assertEqual(kwargs["parent_module"], "dataset")
        self.assertEqual(kwargs["config"], config)
        self.assertEqual(kwargs["module_class"], VisualizationUIModule)


if __name__ == "__main__":
    unittest.main()
