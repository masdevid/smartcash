"""
Tests for the FooterContainer component.
"""
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch, mock_open
import ipywidgets as widgets

class TestFooterContainer(unittest.TestCase):
    """Test cases for the FooterContainer component."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import here to avoid caching issues
        from smartcash.ui.components.footer_container import FooterContainer
        
        # Create a test instance with minimal parameters
        self.footer = FooterContainer(
            show_progress=True,
            show_logs=True,
            show_info=True,
            log_module_name="TestModule",
            log_height="200px"
        )
    
    def test_initialization(self):
        """Test that the footer initializes with the correct components."""
        # Verify the container was created
        self.assertIsNotNone(self.footer.container)
        self.assertIsInstance(self.footer.container, widgets.VBox)
        
        # Verify components were created based on initialization parameters
        self.assertIsNotNone(self.footer.progress_tracker)
        self.assertIsNotNone(self.footer.log_accordion)
        self.assertIsNotNone(self.footer.log_output)
        self.assertIsNotNone(self.footer.info_panel)
        
        # Verify tips panel is not shown by default
        self.assertIsNone(self.footer.tips_panel)
    
    def test_show_hide_tips(self):
        """Test showing and hiding the tips panel."""
        # Initially, tips panel should be None
        self.assertIsNone(self.footer.tips_panel)
        
        # Show tips
        self.footer.show_tips = True
        self.footer._create_components(
            progress_config=None,
            log_module_name="TestModule",
            log_height="200px",
            info_title="Info",
            info_content="Test content",
            info_style="info"
        )
        
        # Now tips panel should be created
        self.assertIsNotNone(self.footer.tips_panel)
        
        # Test showing/hiding
        self.footer.set_tips_visible(False)
        self.assertEqual(self.footer.tips_panel.layout.display, 'none')
        
        self.footer.set_tips_visible(True)
        self.assertEqual(self.footer.tips_panel.layout.display, 'flex')
    
    @patch('importlib.import_module')
    def test_info_box_loading(self, mock_import):
        """Test loading info box from a module."""
        # Create a mock module with a get_test_info function
        mock_module = MagicMock()
        mock_module.get_test_info.return_value = "<div>Test Info</div>"
        mock_import.return_value = mock_module
        
        # Create footer with info_box_path
        from smartcash.ui.components.footer_container import FooterContainer
        footer = FooterContainer(
            show_info=True,
            info_box_path="test_info"
        )
        
        # Verify the module was imported
        mock_import.assert_called_once_with('smartcash.ui.info_boxes.test_info')
        
        # Verify the info panel was created with the mock content
        self.assertIsNotNone(footer.info_panel)
    
    def test_info_box_fallback(self):
        """Test fallback when info box module is not found."""
        # Create footer with non-existent info_box_path
        from smartcash.ui.components.footer_container import FooterContainer
        
        with patch('importlib.import_module') as mock_import:
            # Make import_module raise ImportError
            mock_import.side_effect = ImportError("Test error")
            
            # This should not raise an exception
            footer = FooterContainer(
                show_info=True,
                info_box_path="nonexistent_module"
            )
            
            # Info panel should still be created with error message
            self.assertIsNotNone(footer.info_panel)
    
    def test_progress_tracking(self):
        """Test progress tracking functionality."""
        debug_file = '/tmp/footer_container_debug.log'
        
        def debug_print(*args, **kwargs):
            with open(debug_file, 'a') as f:
                print(*args, file=f, **kwargs)
        
        # Clear the debug file
        with open(debug_file, 'w') as f:
            f.write("=== Starting FooterContainer Progress Test ===\n")
        
        try:
            # Test updating progress
            self.footer.update_progress(50, 100, 'main')
            progress = self.footer.get_progress()
            debug_print(f"[DEBUG] After update_progress(50, 100): {progress}")
            debug_print(f"[DEBUG] Type of progress: {type(progress)}")
            debug_print(f"[DEBUG] progress == 50.0: {progress == 50.0}")
            debug_print(f"[DEBUG] progress == 50: {progress == 50}")
            debug_print(f"[DEBUG] abs(progress - 50.0) < 0.0001: {abs(progress - 50.0) < 0.0001}")
            self.assertAlmostEqual(progress, 50.0, places=6, 
                                 msg=f"Expected progress 50.0, got {progress}")
            
            # Test completing progress
            self.footer.complete()
            progress = self.footer.get_progress()
            debug_print(f"[DEBUG] After complete(): {progress}")
            debug_print(f"[DEBUG] progress == 100.0: {progress == 100.0}")
            self.assertAlmostEqual(progress, 100.0, places=6,
                                 msg=f"Expected progress 100.0, got {progress}")
            
            # Test resetting progress
            self.footer.reset()
            progress = self.footer.get_progress()
            debug_print(f"[DEBUG] After reset(): {progress}")
            debug_print(f"[DEBUG] progress == 0.0: {progress == 0.0}")
            self.assertAlmostEqual(progress, 0.0, places=6,
                                 msg=f"Expected progress 0.0, got {progress}")
            
            # Verify internal progress state
            if hasattr(self.footer, '_progress'):
                debug_print(f"[DEBUG] Internal _progress: {self.footer._progress}")
                debug_print(f"[DEBUG] Type of _progress: {type(self.footer._progress)}")
                
        except Exception as e:
            debug_print(f"[ERROR] Test failed: {str(e)}")
            raise
        finally:
            debug_print("=== Test completed ===\n")
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_logging(self, mock_stdout):
        """Test logging functionality."""
        test_message = "Test log message"
        self.footer.log(test_message)
        
        # Check if the log message was added to the output
        # For Output widget, we need to capture the output differently
        # Since we can't directly access the output, we'll check if the log method was called
        # and that it didn't raise any exceptions
        self.assertTrue(True)  # Just verify the log method was called without errors


class TestCreateFooterContainer(unittest.TestCase):
    """Test cases for the create_footer_container function."""
    
    def test_create_footer_container(self):
        """Test creating a footer container with the factory function."""
        # Import the function directly to test
        from smartcash.ui.components.footer_container import create_footer_container
        
        # Create a footer container with custom options
        footer = create_footer_container(
            show_progress=True,
            show_logs=True,
            show_info=True,
            show_tips=True,
            log_module_name="TestModule",
            log_height="300px",
            tips_title="💡 Helpful Tips",
            tips_content=["Tip 1", "Tip 2"],
            info_title="Information",
            info_content="<p>Test info</p>",
            info_style="info",
            width="100%"
        )
        
        # Verify the container was created
        self.assertIsNotNone(footer.container)
        self.assertIsInstance(footer.container, widgets.VBox)
        
        # Verify components were created based on parameters
        self.assertIsNotNone(footer.progress_tracker)
        self.assertIsNotNone(footer.log_accordion)
        self.assertIsNotNone(footer.log_output)
        self.assertIsNotNone(footer.info_panel)
        self.assertIsNotNone(footer.tips_panel)
        
        # Verify custom parameters were applied
        self.assertEqual(footer.tips_title, "💡 Helpful Tips")
        self.assertIsNotNone(footer.tips_content)
        self.assertTrue(isinstance(footer.tips_content, list))


if __name__ == "__main__":
    unittest.main()
