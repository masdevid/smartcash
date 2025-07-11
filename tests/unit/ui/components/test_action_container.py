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
    
    def test_add_primary_button(self):
        """Test adding a primary button to the container."""
        # Clear existing buttons
        self.container = ActionContainer(container_margin="10px 0")
        
        # Set a simple phase configuration for testing
        self.container.set_phases({
            'test_phase': {
                'text': 'Test Primary',
                'style': 'primary',
                'disabled': False
            }
        })
        
        # Get the default primary button and update its properties
        primary_button = self.container.buttons['primary']
        self.assertIsNotNone(primary_button, "Primary button should be initialized by default")
        
        # Update the primary button's properties
        primary_button.description = 'Test Primary'
        primary_button.tooltip = 'Test primary button'
        primary_button.disabled = False
        
        # Set the current phase to our test phase
        self.container.set_phase('test_phase')
        
        # Get the button using get_button
        button = self.container.get_button('primary')
        
        # Check if button exists and has correct properties
        self.assertIsNotNone(button)
        self.assertEqual(button.description, 'Test Primary')
        self.assertEqual(button.tooltip, 'Test primary button')
        self.assertFalse(button.disabled)
        
        # Verify it's the primary button
        self.assertEqual(self.container.buttons['primary'], button)
    
    def test_add_action_buttons(self):
        """Test adding action buttons to the container."""
        # Create a new container and remove the default primary button
        self.container = ActionContainer(container_margin="10px 0")
        if self.container.buttons['primary'] is not None:
            self.container.buttons['primary'].close()
            self.container.buttons['primary'] = None
        
        # Add action buttons
        button1_id = 'action1'
        button2_id = 'action2'
        
        button1 = self.container.add_button(
            button_id=button1_id,
            text='Action 1',
            style='success',
            tooltip='First action button',
            disabled=False,
            order=1
        )
        
        button2 = self.container.add_button(
            button_id=button2_id,
            text='Action 2',
            style='danger',
            tooltip='Second action button',
            disabled=True,
            order=2
        )
        
        # Check if buttons were added and have correct properties
        self.assertIsNotNone(button1)
        self.assertEqual(button1.description, 'Action 1')
        self.assertEqual(button1.tooltip, 'First action button')
        self.assertFalse(button1.disabled)
        
        self.assertIsNotNone(button2)
        self.assertEqual(button2.description, 'Action 2')
        self.assertEqual(button2.tooltip, 'Second action button')
        self.assertTrue(button2.disabled)
        
        # Verify they're in the action buttons dictionary
        self.assertIsInstance(self.container.buttons['action'], dict)
        self.assertIn(button1_id, self.container.buttons['action'])
        self.assertIn(button2_id, self.container.buttons['action'])
        
        # Verify the container's action buttons are properly set
        self.assertEqual(len(self.container.buttons['action']), 2)
        self.assertEqual(self.container.buttons['action'][button1_id], button1)
        self.assertEqual(self.container.buttons['action'][button2_id], button2)
    
    def test_button_mutual_exclusion(self):
        """Test that primary and action buttons are mutually exclusive."""
        # Test 1: Adding action buttons when primary button exists
        self.container = ActionContainer(container_margin="10px 0")
        
        # Primary button exists by default, try to add an action button
        with self.assertRaises(ValueError) as context:
            self.container.add_button(
                button_id='action1',
                text='Action',
                style='success'
            )
        self.assertIn("Cannot add action buttons when a primary button exists", str(context.exception))
        
        # Test 2: Adding primary button when action buttons exist
        self.container = ActionContainer(container_margin="10px 0")
        
        # First, remove the default primary button
        if self.container.buttons['primary'] is not None:
            self.container.buttons['primary'].close()
            self.container.buttons['primary'] = None
        
        # Add an action button
        self.container.add_button(
            button_id='action1',
            text='Action',
            style='success'
        )
        
        # Verify we have action buttons and no primary button
        self.assertIsInstance(self.container.buttons['action'], dict)
        self.assertIn('action1', self.container.buttons['action'])
        self.assertIsNone(self.container.buttons['primary'])
        
        # Now try to add a primary button - should raise ValueError
        with self.assertRaises(ValueError) as context:
            # Try to add a primary button
            self.container.add_button(
                button_id='primary_btn',
                text='Primary',
                style='primary'
            )
            
        self.assertIn("Cannot add primary button when action buttons exist", str(context.exception))
    
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
    
    def test_create_action_container_with_primary(self):
        """Test creating an action container with a primary button."""
        # Define test buttons (only primary)
        test_buttons = [
            {
                'id': 'primary',
                'text': 'Primary Button',
                'style': 'primary',
                'tooltip': 'Main action',
                'disabled': False
            }
        ]
        
        # Create container
        result = create_action_container(
            buttons=test_buttons,
            title='Test Primary Container',
            container_margin='10px 0',
            show_save_reset=True
        )
        
        # Check if container was created with expected structure
        self.assertIn('container', result)
        self.assertIsInstance(result['container'], widgets.VBox)
        
        # Check if the primary button was created
        self.assertIsNotNone(result['primary_button'])
        self.assertEqual(result['primary_button'].description, 'Primary Button')
        
        # Check container children
        container_children = result['container'].children
        self.assertGreaterEqual(len(container_children), 2)  # Title and buttons
        
    def test_create_action_container_with_actions(self):
        """Test creating an action container with action buttons."""
        # Define test action buttons
        test_buttons = [
            {
                'id': 'action1',
                'text': 'Action 1',
                'style': 'success',
                'tooltip': 'First action',
                'disabled': False,
                'order': 1
            },
            {
                'id': 'action2',
                'text': 'Action 2',
                'style': 'danger',
                'tooltip': 'Second action',
                'disabled': True,
                'order': 2
            }
        ]
        
        # Create container
        result = create_action_container(
            buttons=test_buttons,
            title='Test Actions Container',
            container_margin='10px 0',
            show_save_reset=True
        )
        
        # Check if container was created with expected structure
        self.assertIn('container', result)
        self.assertIsInstance(result['container'], widgets.VBox)
        
        # Check if action buttons were created
        self.assertIn('action1', result['buttons'])
        self.assertIn('action2', result['buttons'])
        self.assertEqual(result['buttons']['action1'].description, 'Action 1')
        self.assertEqual(result['buttons']['action2'].description, 'Action 2')
        
        # Primary button should be None since we're using action buttons
        self.assertIsNone(result['primary_button'])
        
        # Check container children
        container_children = result['container'].children
        self.assertGreaterEqual(len(container_children), 2)  # Title and buttons
        
        # Check if save/reset buttons are present and first in the container
        has_save_reset = False
        for i, child in enumerate(container_children):
            if hasattr(child, 'children'):
                for c in getattr(child, 'children', []):
                    if hasattr(c, 'children'):
                        if any('save' in str(btn).lower() and 'reset' in str(btn).lower() 
                             for btn in c.children if hasattr(btn, 'description')):
                            has_save_reset = True
                            # Save/reset should be in the first section
                            self.assertEqual(i, 0, "Save/Reset buttons should be first in container")
                            break
        
        self.assertTrue(has_save_reset, "Save/Reset buttons not found in container")
        
        # Check title is present (can be after save/reset)
        has_title = any(isinstance(child, widgets.HTML) and 'Test Container' in child.value 
                       for child in container_children)
        self.assertTrue(has_title, "Title not found in container")
        
        # Check if action buttons are present
        has_action_buttons = any(isinstance(child, widgets.Button) or 
                               (hasattr(child, 'children') and 
                                any(isinstance(c, widgets.Button) for c in getattr(child, 'children', [])))
                               for child in container_children)
        self.assertTrue(has_action_buttons, "Action buttons not found in container")


if __name__ == '__main__':
    unittest.main()
