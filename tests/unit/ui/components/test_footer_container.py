"""
Tests for the FooterContainer component.
"""
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch, mock_open
import ipywidgets as widgets
from smartcash.ui.components.info.info_component import InfoBox

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
        
        # Verify the log accordion is an InfoAccordion
        from smartcash.ui.components.info.info_component import InfoAccordion
        self.assertIsInstance(self.footer.log_accordion, InfoAccordion)
        
        # Verify tips panel is not shown by default
        self.assertIsNone(self.footer.tips_panel)
    
    def test_show_hide_tips(self):
        """Test showing and hiding the tips panel."""
        # Initially, tips panel should be None
        self.assertIsNone(self.footer.tips_panel)
        
        # Show tips - this should create the tips panel
        self.footer.show_tips = True
        self.footer.set_tips_visible(True)
        
        # Now tips panel should be created
        self.assertIsNotNone(self.footer.tips_panel)
        
        # Test hiding
        self.footer.set_tips_visible(False)
        self.assertEqual(self.footer.tips_panel.layout.display, 'none')
        
        # Test showing again
        self.footer.set_tips_visible(True)
        self.assertEqual(self.footer.tips_panel.layout.display, 'flex')
    
    def test_info_box_loading(self):
        """Test creating an info box with content."""
        # Create a footer with info content
        from smartcash.ui.components.footer_container import FooterContainer
        footer = FooterContainer(
            show_info=True,
            info_content="test content",
            info_title="Test Title",
            info_style="info"
        )
        
        # Verify the info box was created with the correct parameters
        from smartcash.ui.components.info.info_component import InfoBox
        self.assertIsInstance(footer.info_panel, InfoBox)
        self.assertEqual(footer.info_panel.content, "test content")
        self.assertEqual(footer.info_panel.title, "Test Title")
    
    def test_info_box_fallback(self):
        """Test info box with minimal parameters."""
        # Create footer with minimal info parameters
        from smartcash.ui.components.footer_container import FooterContainer
        from smartcash.ui.components.info.info_component import InfoBox
        
        # This should not raise an exception
        footer = FooterContainer(
            show_info=True,
            info_content="Test content"
        )
        
        # Info panel should be created with default values
        self.assertIsNotNone(footer.info_panel)
        self.assertIsInstance(footer.info_panel, InfoBox)
        self.assertEqual(footer.info_panel.content, "Test content")
    
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
        from smartcash.ui.components.log_accordion import LogLevel
        
        # Test logging at different levels
        self.footer.log("Test info message", LogLevel.INFO)
        self.footer.log("Test warning message", LogLevel.WARNING)
        self.footer.log("Test error message", LogLevel.ERROR)
        
        # Verify the log output was updated
        self.assertIsNotNone(self.footer.log_output)
        
        # Check stdout for the logged messages
        output = mock_stdout.getvalue()
        self.assertIn("Test info message", output)
        self.assertIn("Test warning message", output)
        self.assertIn("Test error message", output)


    def test_update_info(self):
        """Test updating the info panel content."""
        # Test updating with string content
        self.footer.update_info("New Title", "New Content", "success")
        self.assertEqual(self.footer.info_panel.title, "New Title")
        self.assertEqual(self.footer.info_content, "New Content")
        self.assertEqual(self.footer.info_panel.style, "success")
        
        # Test updating with widget content
        test_widget = widgets.HTML("<p>Widget Content</p>")
        self.footer.update_info("Widget Title", test_widget, "warning")
        self.assertEqual(self.footer.info_panel.title, "Widget Title")
        
        # Check the content based on its type
        content = self.footer.info_panel.content
        if hasattr(content, 'value'):
            # Handle case where content is a widget with a value attribute
            self.assertEqual(content.value, "<p>Widget Content</p>")
        elif isinstance(content, str):
            # Handle case where content is a raw string
            self.assertEqual(content, "<p>Widget Content</p>")
        else:
            # For any other case, check if it's the same widget
            self.assertEqual(content, test_widget)
            
        self.assertEqual(self.footer.info_panel.style, "warning")
    
    def test_show_component(self):
        """Test showing and hiding components."""
        # Test showing/hiding progress component
        self.footer.show_component('progress', False)
        self.assertFalse(self.footer.show_progress)
        self.assertEqual(self.footer.progress_component.layout.display, 'none')
        
        self.footer.show_component('progress', True)
        self.assertTrue(self.footer.show_progress)
        self.assertEqual(self.footer.progress_component.layout.display, 'flex')
        
        # Test showing/hiding logs component
        self.footer.show_component('logs', False)
        self.assertFalse(self.footer.show_logs)
        self.assertEqual(self.footer.log_accordion.layout.display, 'none')
        
        # Test showing/hiding info component
        self.footer.show_component('info', False)
        self.assertFalse(self.footer.show_info)
        self.assertEqual(self.footer.info_panel.layout.display, 'none')
    
    def test_add_remove_class(self):
        """Test adding and removing CSS classes."""
        # Test adding a class
        self.footer.add_class('test-class')
        self.assertIn('test-class', self.footer.container._dom_classes)
        
        # Test removing a class
        self.footer.remove_class('test-class')
        self.assertNotIn('test-class', self.footer.container._dom_classes)
        
        # Test removing non-existent class (should not raise)
        self.footer.remove_class('non-existent')
    
    def test_info_content_property(self):
        """Test the info_content property."""
        # Test with string content
        self.footer.update_info("Test", "Test Content", "info")
        self.assertEqual(self.footer.info_content, "Test Content")
        
        # Test with widget content
        test_widget = widgets.HTML("<p>Widget Content</p>")
        self.footer.update_info("Test", test_widget, "info")
        self.assertEqual(self.footer.info_content, test_widget.value)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with all components disabled
        empty_footer = self.footer.__class__(
            show_progress=False,
            show_logs=False,
            show_info=False
        )
        self.assertIsNotNone(empty_footer.container)
        
        # Test updating progress when progress is disabled
        empty_footer.update_progress(50, 100)
        
        # Test logging when logs are disabled
        empty_footer.log("Test log")
        
        # Test showing non-existent component (should not raise)
        self.footer.show_component('nonexistent', True)


class TestCreateFooterContainer(unittest.TestCase):
    """Test cases for the create_footer_container function."""
    
    def test_create_footer_container(self):
        """Test creating a footer container with the factory function."""
        from smartcash.ui.components.footer_container import create_footer_container, FooterContainer
        from smartcash.ui.components.info.info_component import InfoBox, InfoAccordion
        
        # Test with default parameters
        footer = create_footer_container()
        self.assertIsInstance(footer, FooterContainer)
        
        # Test with custom parameters
        footer = create_footer_container(
            show_progress=True,
            show_logs=True,
            show_info=True,
            log_module_name="TestModule",
            log_height="200px",
            info_title="Information",
            info_content="Test Content",
            info_style="info"
        )
        self.assertIsInstance(footer, FooterContainer)
        
        # Verify the components were created with the correct parameters
        self.assertIsNotNone(footer.progress_tracker)
        self.assertIsInstance(footer.log_accordion, InfoAccordion)
        self.assertIsNotNone(footer.log_output)
        self.assertIsInstance(footer.info_panel, InfoBox)
        self.assertEqual(footer.info_panel.title, "Information")
        self.assertEqual(footer.info_panel.content, "Test Content")
        
        # Tips panel should be None by default
        self.assertIsNone(footer.tips_panel)
        
        # Verify custom parameters were applied
        self.assertEqual(footer.info_panel.title, "Information")
        self.assertTrue(hasattr(footer, 'tips_title'))
        self.assertTrue(hasattr(footer, 'tips_content'))
        self.assertTrue(isinstance(footer.tips_content, list))


if __name__ == "__main__":
    unittest.main()
