"""Integration tests for FooterContainer component.

Tests the functionality and integration of the FooterContainer component with
its child components and dependencies.
"""
import pytest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

# Import the container we're testing
from smartcash.ui.components.footer_container import FooterContainer

# Fixtures
@pytest.fixture
def footer_container():
    """Create a FooterContainer instance for testing."""
    return FooterContainer(
        component_name="test_footer",
        version="1.0.0",
        show_copyright=True,
        additional_links=[
            {"label": "Documentation", "url": "https://docs.example.com"},
            {"label": "Support", "url": "https://support.example.com"}
        ]
    )

class TestFooterContainer:
    """Test suite for FooterContainer integration."""
    
    def test_initialization(self, footer_container):
        """Test basic initialization with parameters."""
        assert footer_container is not None
        assert footer_container.component_name == "test_footer"
        assert footer_container.version == "1.0.0"
        assert footer_container.show_copyright is True
        assert len(footer_container.additional_links) == 2
    
    def test_ui_components_creation(self, footer_container):
        """Test that all UI components are created correctly."""
        # Check main container
        assert hasattr(footer_container, 'container')
        assert footer_container.container is not None
        
        # Check footer elements
        assert hasattr(footer_container, 'version_widget')
        assert hasattr(footer_container, 'copyright_widget')
        assert hasattr(footer_container, 'links_container')
        
        # Check links
        assert len(footer_container.links_container.children) == 2  # 2 additional links
    
    def test_version_updates(self, footer_container):
        """Test version update functionality."""
        # Test version update
        new_version = "2.0.0"
        footer_container.update_version(new_version)
        
        assert footer_container.version == new_version
        assert new_version in footer_container.version_widget.value
    
    def test_copyright_visibility(self, footer_container):
        """Test copyright visibility toggle."""
        # Initially visible
        assert footer_container.copyright_widget.layout.visibility == 'visible'
        
        # Hide copyright
        footer_container.toggle_copyright(False)
        assert footer_container.copyright_widget.layout.visibility == 'hidden'
        
        # Show copyright
        footer_container.toggle_copyright(True)
        assert footer_container.copyright_widget.layout.visibility == 'visible'
    
    def test_links_management(self, footer_container):
        """Test adding and removing links."""
        # Initial links count
        initial_links = len(footer_container.links_container.children)
        
        # Add a new link
        footer_container.add_link("New Link", "https://new.example.com")
        assert len(footer_container.links_container.children) == initial_links + 1
        
        # Remove a link
        footer_container.remove_link("New Link")
        assert len(footer_container.links_container.children) == initial_links
    
    def test_custom_styling(self, footer_container):
        """Test custom styling options."""
        # Apply custom styles
        custom_styles = {
            'background': '#f0f0f0',
            'padding': '10px 20px',
            'border_top': '1px solid #ddd'
        }
        
        footer_container.update_style(**custom_styles)
        
        # Verify styles were applied
        container_style = footer_container.container.layout
        assert container_style.background == '#f0f0f0'
        assert container_style.padding == '10px 20px'
        assert container_style.border_top == '1px solid #ddd'
    
    def test_compact_mode(self):
        """Test compact mode styling."""
        # Create footer with compact mode
        footer = FooterContainer(
            component_name="compact_footer",
            compact=True
        )
        
        # Verify compact styles
        container_style = footer.container.layout
        assert container_style.padding == '5px 10px'
        assert 'font-size: 0.8em' in footer.container.style
    
    def test_error_handling(self, footer_container):
        """Test error handling for invalid inputs."""
        # Test invalid version format
        with pytest.raises(ValueError):
            footer_container.update_version("invalid_version")
        
        # Test adding invalid link
        with pytest.raises(ValueError):
            footer_container.add_link("Invalid", "not_a_url")
        
        # Test removing non-existent link
        with pytest.raises(ValueError):
            footer_container.remove_link("non_existent_link")
    
    def test_custom_content(self):
        """Test adding custom content to the footer."""
        # Create footer with custom content
        custom_widget = widgets.HTML("<div>Custom Footer Content</div>")
        footer = FooterContainer(
            component_name="custom_footer",
            custom_content=custom_widget
        )
        
        # Verify custom content is included
        assert custom_widget in footer.container.children
    
    def test_theme_support(self, footer_container):
        """Test theme support and styling."""
        # Apply dark theme
        footer_container.set_theme("dark")
        
        # Verify dark theme styles
        assert 'dark' in footer_container.container.style
        assert 'color: #ffffff' in footer_container.container.style
        
        # Apply light theme
        footer_container.set_theme("light")
        assert 'light' in footer_container.container.style
        assert 'color: #333333' in footer_container.container.style
