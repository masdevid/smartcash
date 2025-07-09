"""
Unit tests for action_container.py
"""
import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

# Import the component to test
from smartcash.ui.components.action_container import ActionContainer, create_action_container, COLAB_PHASES


class TestActionContainer(unittest.TestCase):
    """Test cases for the ActionContainer class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a test instance of ActionContainer with default phases
        self.container = ActionContainer(container_margin="10px 0")
        
        # Get the actual phases used in the container
        self.test_phases = self.container.phases
        
        # Add a test phase if it doesn't exist
        if 'installing_deps' not in self.test_phases:
            self.test_phases['installing_deps'] = {
                'text': 'Installing Dependencies',
                'style': 'info',
                'disabled': True
            }
        
        # Initialize action button for testing
        self.container.buttons['action'] = widgets.Button(
            description='Test Action',
            disabled=False,
            layout=widgets.Layout(width='auto')
        )
        
        # Create test buttons
        self.test_buttons = [
            {
                'id': 'test1',
                'text': 'Test Button 1',
                'style': 'primary',
                'tooltip': 'Test tooltip 1',
                'disabled': False
            },
            {
                'id': 'test2',
                'text': 'Test Button 2',
                'style': 'danger',
                'tooltip': 'Test tooltip 2',
                'disabled': True
            }
        ]
    
    def tearDown(self):
        """Tear down the test environment."""
        pass
    
    def test_initialization(self):
        """Test ActionContainer initialization."""
        # Check if container is created
        self.assertIsInstance(self.container.container, widgets.VBox)
    
        # Check initial state
        self.assertEqual(self.container.current_phase, 'initial')
        
        # Check if all expected phases exist
        for phase in ['initial', 'installing_deps', 'complete']:
            self.assertIn(phase, self.container.phases)
    
        # Check if default buttons are initialized
        self.assertIsNotNone(self.container.buttons['primary'])
        self.assertIsNotNone(self.container.buttons['save_reset'])
        self.assertIsNotNone(self.container.buttons['action'])
    
    def test_add_button(self):
        """Test adding a button to the container."""
        # Add a test button
        button_id = 'test_button'
        self.container.add_button(
            button_id,
            'Test',
            'primary',
            tooltip='Test button',
            disabled=False
        )
        
        # Get the button using get_button
        button = self.container.get_button(button_id)
        
        # Check if button was added and has correct properties
        self.assertIsNotNone(button)
        self.assertEqual(button.description, 'Test')
        self.assertEqual(button.tooltip, 'Test button')
        self.assertFalse(button.disabled)
    
    def test_set_phases(self):
        """Test setting custom phases."""
        # Define test phases
        test_phases = {
            'start': {'text': 'Start', 'style': 'primary'},
            'middle': {'text': 'Middle', 'style': 'warning'},
            'end': {'text': 'End', 'style': 'success'}
        }
        
        # Set new phases
        self.container.set_phases(test_phases)
        
        # Check if phases were updated
        self.assertEqual(len(self.container.phases), len(test_phases))
        for phase_id, phase_config in test_phases.items():
            self.assertIn(phase_id, self.container.phases)
            self.assertEqual(self.container.phases[phase_id]['text'], phase_config['text'])
        
        # Check if current phase is set to the first phase
        self.assertEqual(self.container.current_phase, 'start')
    
    def test_set_phase(self):
        """Test changing the current phase."""
        # Set up a test phase
        test_phase = 'installing_deps'
        self.container.set_phase(test_phase)
        
        # Check if phase was updated
        self.assertEqual(self.container.current_phase, test_phase)
        
        # Check if primary button was updated if it exists
        if self.container.buttons['primary']:
            phase_config = COLAB_PHASES[test_phase]
            self.assertEqual(self.container.buttons['primary'].description, phase_config['text'])
            self.assertEqual(self.container.buttons['primary'].disabled, phase_config['disabled'])
    
    def test_enable_disable_buttons(self):
        """Test enabling and disabling all buttons."""
        # Get default buttons
        primary_btn = self.container.buttons['primary']
        save_reset_btn = self.container.buttons['save_reset']
        action_btn = self.container.buttons['action']
        
        # Test disable
        self.container.disable_all()
        self.assertTrue(primary_btn.disabled)
        self.assertTrue(save_reset_btn.disabled)
        self.assertTrue(action_btn.disabled)
        
        # Test enable
        self.container.enable_all()
        self.assertFalse(primary_btn.disabled)
        self.assertFalse(save_reset_btn.disabled)
        self.assertFalse(action_btn.disabled)
    
    def test_get_button_nonexistent(self):
        """Test getting a non-existent button."""
        button = self.container.get_button('nonexistent')
        self.assertIsNone(button)


class TestCreateActionContainer(unittest.TestCase):
    """Test cases for the create_action_container function."""
    
    def test_create_action_container(self):
        """Test creating an action container with buttons."""
        # Define test buttons
        test_buttons = [
            {
                'id': 'btn1',
                'text': 'Button 1',
                'style': 'primary',
                'tooltip': 'First button',
                'disabled': False,
                'order': 1
            },
            {
                'id': 'btn2',
                'text': 'Button 2',
                'style': 'warning',
                'tooltip': 'Second button',
                'disabled': True,
                'order': 2
            }
        ]
        
        # Create container
        result = create_action_container(
            buttons=test_buttons,
            title='Test Container',
            alignment='center',
            container_margin='10px 0',
            show_save_reset=True
        )
        
        # Check if container was created with expected structure
        self.assertIn('container', result)
        self.assertIsInstance(result['container'], widgets.VBox)
        
        # Check if buttons were added to the container
        container_children = result['container'].children
        self.assertGreaterEqual(len(container_children), 2)  # At least title and buttons
        
        # Check title
        self.assertIsInstance(container_children[0], widgets.HTML)
        self.assertIn('Test Container', container_children[0].value)
        
        # Check buttons
        self.assertTrue(any(isinstance(child, widgets.Button) for child in container_children))
        
        # Check if save/reset buttons are present
        has_save_reset = any(hasattr(child, 'children') and 
                            any('save' in str(c).lower() and 'reset' in str(c).lower()
                               for c in getattr(child, 'children', []))
                            for child in container_children)
        self.assertTrue(has_save_reset, "Save/Reset buttons not found in container")


if __name__ == '__main__':
    unittest.main()
