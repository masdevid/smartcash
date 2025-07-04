"""Tests for info components."""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

from smartcash.ui.components.info import (
    InfoBox,
    InfoAccordion,
    TabbedInfo,
    create_info_accordion,
    style_info_content,
    create_tabbed_info
)


class TestInfoBox(unittest.TestCase):
    """Tests for the InfoBox component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.content = "Test content"
        self.title = "Test Title"
        self.style = "success"
        self.component = InfoBox(
            content=self.content,
            title=self.title,
            style=self.style
        )
    
    def test_initialization(self):
        """Test component initialization."""
        self.assertEqual(self.component.content, self.content)
        self.assertEqual(self.component.title, self.title)
        self.assertEqual(self.component.style, self.style)
        self.assertFalse(self.component._initialized)
    
    def test_show(self):
        """Test showing the component."""
        widget = self.component.show()
        self.assertIsInstance(widget, widgets.HTML)
        self.assertTrue(self.component._initialized)
    
    def test_update_content(self):
        """Test updating the content."""
        self.component.show()
        new_content = "Updated content"
        self.component.update_content(new_content)
        self.assertEqual(self.component.content, new_content)
    
    def test_update_style(self):
        """Test updating the style."""
        self.component.show()
        new_style = "warning"
        self.component.update_style(new_style)
        self.assertEqual(self.component.style, new_style)


class TestInfoAccordion(unittest.TestCase):
    """Tests for the InfoAccordion component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.title = "Test Accordion"
        self.content = "Test content"
        self.component = InfoAccordion(
            title=self.title,
            content=self.content
        )
    
    def test_initialization(self):
        """Test component initialization."""
        self.assertEqual(self.component.accordion_title, self.title)
        self.assertEqual(self.component.content, self.content)
        self.assertIsNone(self.component.icon)
        self.assertFalse(self.component.open_by_default)
    
    def test_show(self):
        """Test showing the component."""
        widget = self.component.show()
        self.assertIsInstance(widget, widgets.Accordion)
        self.assertTrue(self.component._initialized)
    
    def test_set_open(self):
        """Test setting the accordion open/closed state."""
        self.component.show()
        self.component.set_open(True)
        self.assertEqual(self.component._ui_components['accordion'].selected_index, 0)
        self.component.set_open(False)
        self.assertIsNone(self.component._ui_components['accordion'].selected_index)


class TestTabbedInfo(unittest.TestCase):
    """Tests for the TabbedInfo component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tabs_content = {
            "Tab 1": "Content 1",
            "Tab 2": "Content 2"
        }
        self.component = TabbedInfo(tabs_content=self.tabs_content)
    
    def test_initialization(self):
        """Test component initialization."""
        self.assertEqual(self.component.tabs_content, self.tabs_content)
        self.assertEqual(self.component.style, "info")
    
    def test_show(self):
        """Test showing the component."""
        widget = self.component.show()
        self.assertIsInstance(widget, widgets.Tab)
        self.assertEqual(len(widget.children), 2)
        self.assertEqual(widget.get_title(0), "Tab 1")
        self.assertEqual(widget.get_title(1), "Tab 2")
    
    def test_add_tab(self):
        """Test adding a new tab."""
        self.component.show()
        self.component.add_tab("Tab 3", "Content 3")
        self.assertEqual(len(self.component.tabs_content), 3)
        self.assertIn("Tab 3", self.component.tabs_content)


class TestLegacyFunctions(unittest.TestCase):
    """Tests for legacy functions (backward compatibility)."""
    
    def test_create_info_accordion(self):
        """Test the legacy create_info_accordion function."""
        accordion = create_info_accordion("Test", "Content")
        self.assertIsInstance(accordion, widgets.Accordion)
    
    def test_style_info_content(self):
        """Test the legacy style_info_content function."""
        styled = style_info_content("Test")
        self.assertIn("Test", styled)
        self.assertIn("background-color", styled)
    
    def test_create_tabbed_info(self):
        """Test the legacy create_tabbed_info function."""
        tabs = create_tabbed_info({"Tab 1": "Content 1"})
        self.assertIsInstance(tabs, widgets.Tab)


if __name__ == "__main__":
    unittest.main()
