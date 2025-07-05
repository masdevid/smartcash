"""Integration tests for FormContainer component.

Tests the functionality and integration of the FormContainer component with
its child components and dependencies.
"""
import pytest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

# Import the container we're testing
from smartcash.ui.components.form_container import FormContainer, FormItem

# Fixtures
@pytest.fixture
def form_container():
    """Create a FormContainer instance for testing."""
    return FormContainer(
        component_name="test_form",
        layout="vertical",
        grid_columns=2,
        grid_template_areas=None,
        spacing="10px"
    )

class TestFormContainer:
    """Test suite for FormContainer integration."""
    
    def test_initialization(self, form_container):
        """Test basic initialization with parameters."""
        assert form_container is not None
        assert form_container.component_name == "test_form"
        assert form_container.layout == "vertical"
        assert form_container.grid_columns == 2
        assert form_container.spacing == "10px"
    
    def test_add_form_item(self, form_container):
        """Test adding form items to the container."""
        # Create test widgets
        text_input = widgets.Text(description="Name:")
        number_input = widgets.IntText(description="Age:")
        
        # Add form items
        form_container.add_item("name", text_input)
        form_container.add_item("age", number_input)
        
        # Verify items were added
        assert "name" in form_container.items
        assert "age" in form_container.items
        assert len(form_container.container.children) == 2
    
    def test_grid_layout(self):
        """Test grid layout configuration."""
        # Create a form with grid layout
        container = FormContainer(
            component_name="grid_form",
            layout="grid",
            grid_columns=3,
            grid_template_areas="""
                "name name email"
                "age . phone"
                "address address address"
            """
        )
        
        # Add items to the grid
        container.add_item("name", widgets.Text(description="Name:"), grid_area="name")
        container.add_item("age", widgets.IntText(description="Age:"), grid_area="age")
        container.add_item("email", widgets.Text(description="Email:"), grid_area="email")
        container.add_item("phone", widgets.Text(description="Phone:"), grid_area="phone")
        container.add_item("address", widgets.Textarea(description="Address:"), grid_area="address")
        
        # Verify grid configuration
        assert container.container.layout.grid_template_areas == '''
                "name name email"
                "age . phone"
                "address address address"
            '''
        assert container.container.layout.grid_template_columns == '1fr 1fr 1fr'
    
    def test_form_validation(self, form_container):
        """Test form validation functionality."""
        # Add required fields
        required_text = widgets.Text(description="Required:")
        form_container.add_item("required_field", required_text, required=True)
        
        # Test validation
        assert form_container.validate() is False  # Should be invalid (empty required field)
        
        # Set a value and validate again
        required_text.value = "Test value"
        assert form_container.validate() is True  # Should be valid now
    
    def test_form_data(self, form_container):
        """Test getting and setting form data."""
        # Add test fields
        text_input = widgets.Text(description="Name:")
        number_input = widgets.IntText(description="Age:")
        
        form_container.add_item("name", text_input)
        form_container.add_item("age", number_input)
        
        # Set form data
        test_data = {
            "name": "John Doe",
            "age": 30
        }
        form_container.set_data(test_data)
        
        # Verify data was set
        assert text_input.value == "John Doe"
        assert number_input.value == 30
        
        # Get form data
        form_data = form_container.get_data()
        assert form_data["name"] == "John Doe"
        assert form_data["age"] == 30
    
    def test_form_reset(self, form_container):
        """Test form reset functionality."""
        # Add a field with initial value
        text_input = widgets.Text(description="Test:", value="Initial")
        form_container.add_item("test", text_input)
        
        # Change the value
        text_input.value = "Changed"
        
        # Reset the form
        form_container.reset()
        
        # Verify value was reset
        assert text_input.value == "Initial"
    
    def test_custom_styling(self, form_container):
        """Test custom styling options."""
        # Apply custom styles
        form_container.update_style(
            background="#f5f5f5",
            padding="20px",
            border_radius="8px"
        )
        
        # Verify styles were applied
        assert form_container.container.layout.background == "#f5f5f5"
        assert form_container.container.layout.padding == "20px"
        assert form_container.container.layout.border_radius == "8px"
    
    def test_form_item_validation(self):
        """Test FormItem validation."""
        # Create a test widget
        test_widget = widgets.Text(description="Test:")
        
        # Test valid FormItem creation
        form_item = FormItem(
            widget=test_widget,
            width="200px",
            height="40px",
            flex="1",
            grid_area="test"
        )
        
        assert form_item.widget == test_widget
        assert form_item.width == "200px"
        assert form_item.height == "40px"
        assert form_item.flex == "1"
        assert form_item.grid_area == "test"
        
        # Test invalid width
        with pytest.raises(ValueError):
            FormItem(widget=test_widget, width="invalid")
            
        # Test invalid height
        with pytest.raises(ValueError):
            FormItem(widget=test_widget, height="invalid")
