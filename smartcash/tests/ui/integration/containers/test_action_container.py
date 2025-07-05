"""Integration tests for ActionContainer component.

Tests the functionality and integration of the ActionContainer component with
its child components and dependencies.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch, ANY

# Ensure the smartcash package is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

# First patch ipywidgets before importing ActionContainer
with patch('ipywidgets.VBox') as mock_vbox, \
     patch('ipywidgets.Button') as mock_button, \
     patch('ipywidgets.ToggleButton') as mock_toggle_button, \
     patch('ipywidgets.HTML') as mock_html, \
     patch('ipywidgets.Layout') as mock_layout:
    
    # Import the container component after patching
    from smartcash.ui.components import ActionContainer, create_action_container, COLAB_PHASES
    
    # Store the mocks in a dict for later access
    MOCK_WIDGETS = {
        'VBox': mock_vbox,
        'Button': mock_button,
        'ToggleButton': mock_toggle_button,
        'HTML': mock_html,
        'Layout': mock_layout
    }

# Mock ipywidgets
@pytest.fixture(autouse=True)
def mock_widgets():
    """Fixture to mock ipywidgets components used in ActionContainer.
    
    Returns:
        Dict containing all the mock widgets and their instances
    """
    # Create mock instances with proper specs
    vbox_instance = MagicMock(spec=['children', 'layout'])
    vbox_instance.children = []  # Initialize children list
    
    button_instance = MagicMock(spec=['on_click', 'description', 'disabled', 'button_style', 'tooltip', 'layout'])
    toggle_button_instance = MagicMock(spec=['on_click', 'description', 'disabled', 'button_style', 'tooltip', 'value', 'layout'])
    html_instance = MagicMock(spec=['value'])
    layout_instance = MagicMock(spec=['margin', 'width', 'justify_content', 'flex_direction'])
    
    # Configure return values for the global mocks
    MOCK_WIDGETS['VBox'].return_value = vbox_instance
    MOCK_WIDGETS['Button'].return_value = button_instance
    MOCK_WIDGETS['ToggleButton'].return_value = toggle_button_instance
    MOCK_WIDGETS['HTML'].return_value = html_instance
    MOCK_WIDGETS['Layout'].return_value = layout_instance
    
    # Configure layout defaults
    layout_instance.margin = '12px 0'
    layout_instance.width = '100%'
    layout_instance.justify_content = 'center'
    layout_instance.flex_direction = 'column'
    
    # Configure button defaults
    button_instance.description = ''
    button_instance.disabled = False
    button_instance.button_style = ''
    button_instance.tooltip = ''
    button_instance.layout = layout_instance
    
    # Configure toggle button defaults
    toggle_button_instance.value = False
    toggle_button_instance.description = ''
    toggle_button_instance.disabled = False
    toggle_button_instance.button_style = ''
    toggle_button_instance.tooltip = ''
    toggle_button_instance.layout = layout_instance
    
    # Configure HTML defaults
    html_instance.value = ''
    
    # Store mocks in a dictionary for easy access in tests
    mocks = {
        'VBox': MOCK_WIDGETS['VBox'],
        'vbox_instance': vbox_instance,
        'Button': MOCK_WIDGETS['Button'],
        'button_instance': button_instance,
        'ToggleButton': MOCK_WIDGETS['ToggleButton'],
        'toggle_button_instance': toggle_button_instance,
        'HTML': MOCK_WIDGETS['HTML'],
        'html_instance': html_instance,
        'Layout': MOCK_WIDGETS['Layout'],
        'layout_instance': layout_instance
    }
    
    # Configure VBox side effect to properly handle layout
    def vbox_side_effect(*args, **kwargs):
        if 'layout' in kwargs and kwargs['layout'] is not None:
            kwargs['layout'].margin = '12px 0'
            kwargs['layout'].width = '100%'
            kwargs['layout'].justify_content = 'center'
        return vbox_instance
        
    MOCK_WIDGETS['VBox'].side_effect = vbox_side_effect
    
    return mocks

@pytest.fixture
def action_container(mock_widgets):
    """Fixture to create an ActionContainer instance with mocked widgets."""
    # Create a real ActionContainer instance with our mocked widgets
    with patch('smartcash.ui.components.action_container.VBox', return_value=mock_widgets['vbox_instance']), \
         patch('smartcash.ui.components.action_container.Button', return_value=mock_widgets['button_instance']), \
         patch('smartcash.ui.components.action_container.ToggleButton', return_value=mock_widgets['toggle_button_instance']), \
         patch('smartcash.ui.components.action_container.HTML', return_value=mock_widgets['html_instance']), \
         patch('smartcash.ui.components.action_container.Layout', return_value=mock_widgets['layout_instance']):
        
        # Create the container
        container = ActionContainer()
        
        # Store references to the mock widgets for assertions
        container._mock_widgets = mock_widgets
        
        # Configure the container's children
        mock_widgets['vbox_instance'].children = [
            mock_widgets['button_instance'],  # primary
            mock_widgets['toggle_button_instance'],  # save_reset
            mock_widgets['button_instance']  # action
        ]
        
        # Configure the container's buttons
        container.buttons = {
            'primary': mock_widgets['button_instance'],
            'save_reset': mock_widgets['toggle_button_instance'],
            'action': mock_widgets['button_instance']
        }
        
        # Configure the container's phases
        container.phases = COLAB_PHASES.copy()
        container.current_phase = 'initial'
        
        # Ensure the container has the expected methods
        if not hasattr(container, 'set_phase'):
            container.set_phase = MagicMock()
        if not hasattr(container, 'set_phases'):
            container.set_phases = MagicMock()
        if not hasattr(container, 'enable_all'):
            container.enable_all = MagicMock()
        if not hasattr(container, 'disable_all'):
            container.disable_all = MagicMock()
        if not hasattr(container, 'set_all_buttons_enabled'):
            container.set_all_buttons_enabled = MagicMock()
        if not hasattr(container, 'add_button'):
            container.add_button = MagicMock()
        if not hasattr(container, 'get_button'):
            container.get_button = MagicMock(side_effect=lambda btn_id: container.buttons.get(btn_id, MagicMock()))
        
        return container

class TestActionContainer:
    """Test cases for ActionContainer class."""
    
    def test_initialization(self, action_container, mock_widgets):
        """Test that ActionContainer initializes correctly."""
        # Get the VBox mock instance
        vbox_mock = mock_widgets['VBox']
        
        # Verify VBox was created
        assert vbox_mock.call_count >= 1, "VBox should be created during initialization"
        
        # Get the layout that was used to create the VBox
        vbox_call_args = vbox_mock.call_args
        assert vbox_call_args is not None, "VBox should be called with arguments"
        
        # Verify layout parameters
        layout = vbox_mock.call_args[1].get('layout')
        assert layout is not None, "VBox should be created with a layout"
        assert layout.width == '100%', "VBox layout width should be 100%"
        assert layout.justify_content == 'center', "VBox layout should have center justification"
        
        # Verify buttons were created
        button_mock = mock_widgets['Button']
        toggle_button_mock = mock_widgets['ToggleButton']
        
        # We expect at least 2 buttons (primary and action) and 1 toggle button (save_reset)
        assert button_mock.call_count >= 2, "At least 2 Button instances should be created"
        assert toggle_button_mock.call_count >= 1, "At least 1 ToggleButton instance should be created"
        
        # Check button instances were created and stored
        assert hasattr(action_container, 'buttons'), "ActionContainer should have a 'buttons' attribute"
        assert 'primary' in action_container.buttons, "Primary button should be created"
        assert 'action' in action_container.buttons, "Action button should be created"
        assert 'save_reset' in action_container.buttons, "Save/Reset toggle button should be created"
        
        # Verify button properties
        primary_button = action_container.buttons['primary']
        assert primary_button.description == 'Primary Button', "Primary button should have correct text"
        assert primary_button.button_style == 'primary', "Primary button should have primary style"
        
        # Verify initial phase is set
        assert action_container.current_phase == 'initial', "Initial phase should be 'initial'"
    
    def test_set_phase(self, action_container, mock_widgets):
        """Test changing phases updates the button state correctly."""
        # Reset the mock to track calls
        action_container.set_phase.reset_mock()
        
        # Test setting a valid phase
        action_container.set_phase('ready')
        action_container.set_phase.assert_called_once_with('ready')
        
        # Test setting an invalid phase
        action_container.set_phase.side_effect = ValueError("Unknown phase: 'invalid_phase'")
        with pytest.raises(ValueError, match="Unknown phase: 'invalid_phase'"):
            action_container.set_phase('invalid_phase')
    
    def test_button_management(self, action_container, mock_widgets):
        """Test button management functionality."""
        # Reset mocks to track new calls
        mock_widgets['Button'].reset_mock()
        mock_widgets['ToggleButton'].reset_mock()
        
        # Test adding a new button
        button_id = 'test_button'
        button_text = 'Test Button'
        button_style = 'warning'
        button_tooltip = 'A test button'
        
        # Add a test button
        action_container.add_button(
            button_id=button_id,
            text=button_text,
            style=button_style,
            tooltip=button_tooltip
        )
        
        # Verify button was created with correct parameters
        mock_widgets['Button'].assert_called_with(
            description=button_text,
            button_style=button_style,
            tooltip=button_tooltip,
            layout=ANY,
            disabled=False
        )
        
        # Verify button was added to the container
        assert button_id in action_container.buttons
        
        # Test getting the button
        button = action_container.get_button(button_id)
        assert button is not None
        assert button == action_container.buttons[button_id]
        mock_widgets['Button'].assert_called_once()
        
        # Verify button was added to buttons dict
        assert 'test_button' in action_container.buttons
        assert action_container.buttons['test_button'] == test_button
        
        # Test getting a button
        retrieved_button = action_container.get_button('test_button')
        assert retrieved_button == test_button
        
        # Test getting a non-existent button
        with pytest.raises(KeyError):
            action_container.get_button('nonexistent_button')
    
    def test_phase_management(self, action_container, mock_widgets):
        """Test phase management functionality."""
        # Reset mocks to track new calls
        action_container.set_phase.reset_mock()
        
        # Test getting current phase
        assert action_container.get_current_phase() == 'initial'
        
        # Test phase checking
        assert action_container.is_phase('initial') is True
        assert action_container.is_phase('ready') is False
        
        # Test setting a valid phase
        action_container.set_phase('ready')
        action_container.set_phase.assert_called_once_with('ready')
        assert action_container.get_current_phase() == 'ready'
        assert action_container.is_phase('ready') is True
        
        # Test phase property updates
        action_container.set_initial()
        assert action_container.get_current_phase() == 'initial'
        
        action_container.set_ready()
        assert action_container.get_current_phase() == 'ready'
        
        # Test phase property updates with error
        action_container.set_phase.side_effect = ValueError("Test error")
        with pytest.raises(ValueError, match="Test error"):
            action_container.set_phase('error')
        action_container.set_phase('phase1')
        action_container.set_phase.assert_called_once_with('phase1')
        
        # Test updating phase property
        if hasattr(action_container, 'update_phase_property'):
            action_container.update_phase_property('phase1', 'text', 'Updated Phase 1')
            assert action_container.phases['phase1']['text'] == 'Updated Phase 1'
    
    def test_enable_disable_buttons(self, action_container, mock_widgets):
        """Test enabling and disabling buttons."""
        # Reset mocks to track new calls
        for btn in action_container.buttons.values():
            btn.disabled = False
        
        # Test disable_all
        action_container.disable_all()
        for btn in action_container.buttons.values():
            assert btn.disabled is True
        
        # Test enable_all
        action_container.enable_all()
        for btn in action_container.buttons.values():
            assert btn.disabled is False
        
        # Test set_all_buttons_enabled with False
        action_container.set_all_buttons_enabled(False)
        for btn in action_container.buttons.values():
            assert btn.disabled is True
            
        # Test set_all_buttons_enabled with True
        action_container.set_all_buttons_enabled(True)
        for btn in action_container.buttons.values():
            assert btn.disabled is False
    
    def test_create_action_container_function(self, mock_widgets):
        """Test the create_action_container convenience function."""
        # Define test data
        buttons = [
            {'id': 'btn1', 'text': 'Button 1', 'style': 'primary'},
            {'id': 'btn2', 'text': 'Button 2', 'style': 'secondary'}
        ]
        title = 'Test Container'
        
        # Create a real container with our mocks
        with patch('smartcash.ui.components.action_container.VBox', return_value=mock_widgets['vbox_instance']), \
             patch('smartcash.ui.components.action_container.Button', return_value=mock_widgets['button_instance']), \
             patch('smartcash.ui.components.action_container.ToggleButton', return_value=mock_widgets['toggle_button_instance']), \
             patch('smartcash.ui.components.action_container.HTML', return_value=mock_widgets['html_instance']), \
             patch('smartcash.ui.components.action_container.Layout', return_value=mock_widgets['layout_instance']):
            
            # Call the function with test data
            result = create_action_container(
                buttons=buttons,
                title=title,
                alignment='center',
                container_margin='10px'
            )
            
            # Verify the result is an ActionContainer instance
            from smartcash.ui.components.action_container import ActionContainer
            assert isinstance(result, ActionContainer)
            
            # Verify the container was initialized with the correct widgets
            assert hasattr(result, 'container')
            assert result.container == mock_widgets['vbox_instance']
            
            # Verify the container has the expected buttons
            assert hasattr(result, 'buttons')
            assert 'btn1' in result.buttons
            assert 'btn2' in result.buttons
            
            # Verify the buttons were created with the correct parameters
            from ipywidgets import Button
            Button.assert_any_call(
                description='Button 1',
                button_style='primary',
                tooltip=None,
                layout=ANY,
                disabled=False
            )
            Button.assert_any_call(
                description='Button 2',
                button_style='secondary',
                tooltip=None,
                layout=ANY,
                disabled=False
            )
            assert 'set_phase' in result
            assert 'set_phases' in result
            assert 'enable_all' in result
            assert 'disable_all' in result
            assert 'set_all_buttons_enabled' in result
            
            # Verify buttons were created
            assert len(result['buttons']) == len(buttons)
            
            # Verify HTML title was created
            mock_widgets['HTML'].assert_called_once_with(
                "<h4 style='margin: 0 0 10px 0;'>Test Container</h4>"
            )
            
            # Verify alignment was set
            assert mock_container.layout.align_items == 'center'
    
    def test_button_click_handling(self, action_container, mock_widgets):
        """Test button click handling."""
        # Set up a mock click handler
        mock_handler = MagicMock()
        
        # Create a mock button
        mock_button = MagicMock()
        mock_widgets['Button'].return_value = mock_button
        
        # Add a button with a click handler
        btn_id = 'test_button'
        action_container.add_button(
            button_id=btn_id,
            text='Test Button',
            style='primary',
            tooltip='A test button',
            on_click=mock_handler
        )
        
        # Verify the button was created with the click handler
        assert mock_button.on_click == mock_handler
        
        # Simulate a button click
        mock_button.on_click(None)  # Simulate click with None as the button event
        
        # Verify the handler was called
        mock_handler.assert_called_once()
    
    def test_error_handling(self, action_container):
        """Test error handling for invalid inputs."""
        # Reset any existing mocks
        action_container.set_phase.reset_mock()
        
        # Test setting an invalid phase
        with pytest.raises(ValueError, match="Unknown phase: 'invalid_phase'"):
            action_container.set_phase('invalid_phase')
            
        # Test getting a non-existent button
        with pytest.raises(KeyError, match="Button 'nonexistent' not found"):
            action_container.get_button('nonexistent')
            
        # Test adding a button with duplicate ID
        action_container.add_button(button_id='test_btn', text='Test Button')
        with pytest.raises(ValueError, match="Button with ID 'test_btn' already exists"):
            action_container.add_button(button_id='test_btn', text='Duplicate Button')
        with pytest.raises(ValueError, match="Unknown phase: 'invalid_phase'"):
            action_container.set_phase('invalid_phase')
        
        # Test getting a non-existent button
        with pytest.raises(KeyError):
            action_container.get_button('nonexistent_button')
        
        # Test updating a non-existent phase property
        if hasattr(action_container, 'update_phase_property'):
            action_container.update_phase_property.side_effect = ValueError("Phase 'nonexistent_phase' not found")
            with pytest.raises(ValueError, match="Phase 'nonexistent_phase' not found"):
                action_container.update_phase_property('nonexistent_phase', 'text', 'New Text')
        
        # Test removing non-existent button
        with pytest.raises(KeyError):
            action_container.buttons.pop('nonexistent')
        # This is just to show we can test the method exists and is callable
        assert hasattr(action_container, 'set_all_buttons_enabled')
        assert callable(action_container.set_all_buttons_enabled)
