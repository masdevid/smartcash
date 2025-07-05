"""
Tests for the FormContainer component.
"""
import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add the project root to the Python path for direct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

# Import the form_container module
from smartcash.ui.components import form_container

# Import the components we need
create_form_container = form_container.create_form_container
LayoutType = form_container.LayoutType
FormItem = form_container.FormItem

# Import widgets for testing
import ipywidgets as widgets

class TestFormContainer(unittest.TestCase):
    """Test cases for the FormContainer component."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock commonly used widgets
        self.mock_widget = MagicMock(spec=widgets.Widget)
        # Setup mock layout
        self.mock_layout = MagicMock()
        self.mock_widget.layout = self.mock_layout
        self.mock_button = MagicMock(spec=widgets.Button)
        
        # Patch ipywidgets
        self.patchers = [
            patch('ipywidgets.VBox'),
            patch('ipywidgets.HBox'),
            patch('ipywidgets.GridBox'),
            patch('ipywidgets.Button', return_value=self.mock_button)
        ]
        
        for patcher in self.patchers:
            patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures."""
        for patcher in self.patchers:
            patcher.stop()
    
    def test_create_form_container_basic(self):
        """Test creating a basic form container with default settings."""
        # Mock the widgets
        with patch('ipywidgets.VBox') as mock_vbox, \
             patch('ipywidgets.Layout') as mock_layout:
            
            # Setup mock return values
            mock_vbox.return_value = MagicMock()
            mock_vbox.return_value.layout = MagicMock()
            
            form = create_form_container()
            
            # Verify the form has the expected structure
            self.assertIn('container', form)
            self.assertIn('form_container', form)
            self.assertIn('add_item', form)
            self.assertIn('set_layout', form)
            
            # Verify VBox was created with correct layout
            mock_vbox.assert_called_once()
            mock_layout.assert_called_with(
                display='flex',
                flex_flow='column',
                gap='8px',
                margin='8px 0',
                padding='16px',
                width='100%',
                align_items='stretch'
            )
    
    def test_create_form_container_row_layout(self):
        """Test creating a form container with row layout."""
        with patch('ipywidgets.VBox') as mock_vbox, \
             patch('ipywidgets.Layout') as mock_layout:
            
            mock_vbox.return_value = MagicMock()
            mock_vbox.return_value.layout = MagicMock()
            
            form = create_form_container(layout_type=LayoutType.ROW)
            
            # Verify layout was created with row settings
            mock_layout.assert_called_once()
            call_kwargs = mock_layout.call_args[1]
            self.assertEqual(call_kwargs['flex_flow'], 'row wrap')
            self.assertEqual(call_kwargs['align_items'], 'flex-start')
    
    def test_create_form_container_grid_layout(self):
        """Test creating a form container with grid layout."""
        form = create_form_container(
            layout_type=LayoutType.GRID,
            grid_columns=2,
            grid_template_areas=['"a b"', '"c d"'],
            grid_auto_flow='row dense'
        )
    
    def test_add_item_to_form(self):
        """Test adding items to the form container."""
        with patch('ipywidgets.VBox'), \
             patch('ipywidgets.Layout'):
            
            # Create a mock form container
            mock_container = MagicMock()
            mock_container.children = []
            
            # Mock the create_form_container function
            with patch('smartcash.ui.components.form_container.create_form_container') as mock_create_form:
                mock_create_form.return_value = {
                    'form_container': mock_container,
                    'add_item': lambda x: mock_container.children.append(x.widget)
                }
                
                form = create_form_container()
                
                # Create test widgets
                mock_widget1 = MagicMock()
                mock_widget1.layout = MagicMock()
                mock_widget2 = MagicMock()
                mock_widget2.layout = MagicMock()
                
                # Add items to the form
                item1 = FormItem(mock_widget1, width='100%')
                item2 = FormItem(mock_widget2, width='50%')
                
                form['add_item'](item1)
                form['add_item'](item2)
                
                # Verify items were added
                self.assertEqual(len(mock_container.children), 2)
    
    def test_set_layout_dynamic(self):
        """Test dynamically changing the layout."""
        with patch('ipywidgets.VBox') as mock_vbox, \
             patch('ipywidgets.Layout') as mock_layout:
            
            # Setup mock container
            mock_container = MagicMock()
            mock_container.layout = MagicMock()
            mock_vbox.return_value = mock_container
            
            # Create form with column layout
            form = create_form_container(layout_type=LayoutType.COLUMN)
            
            # Get the set_layout function
            set_layout = form['set_layout']
            
            # Test changing to row layout
            mock_layout.reset_mock()
            set_layout(layout_type=LayoutType.ROW, gap='16px')
            
            # Verify layout was updated
            self.assertEqual(mock_container.layout.flex_flow, 'row wrap')
            self.assertEqual(mock_container.layout.gap, '16px')
            
            # Test changing to grid layout
            mock_layout.reset_mock()
            set_layout(layout_type=LayoutType.GRID, grid_columns=3, gap='8px')
            
            # Verify grid layout was applied
            self.assertEqual(mock_container.layout.display, 'grid')
            self.assertEqual(mock_container.layout.grid_template_columns, '1fr 1fr 1fr')
    
    def test_form_item_validation(self):
        """Test FormItem validation and normalization."""
        # Test align_items validation
        self.assertEqual(FormItem._validate_align_items('left'), 'flex-start')
        self.assertEqual(FormItem._validate_align_items('center'), 'center')
        self.assertEqual(FormItem._validate_align_items('invalid'), 'stretch')
    
    def test_form_container_with_custom_styles(self):
        """Test creating a form container with custom styles."""
        # Mock the widgets
        with patch('ipywidgets.VBox') as mock_vbox, \
             patch('ipywidgets.Layout') as mock_layout:
            
            # Setup mock return values
            mock_vbox.return_value = MagicMock()
            mock_vbox.return_value.layout = MagicMock()
            
            form = create_form_container(
                container_margin='20px',
                container_padding='10px',
                gap='16px',
                width='80%',
                height='500px'
            )
            
            # Verify layout was created with correct styles
            mock_layout.assert_called_once()
            call_kwargs = mock_layout.call_args[1]
            self.assertEqual(call_kwargs['margin'], '20px')
            self.assertEqual(call_kwargs['padding'], '10px')
            self.assertEqual(call_kwargs['gap'], '16px')
            self.assertEqual(call_kwargs['width'], '80%')
            self.assertEqual(call_kwargs['height'], '500px')
    
    def test_form_container_with_invalid_layout(self):
        """Test creating a form container with invalid layout type."""
        with patch('ipywidgets.VBox'), \
             patch('ipywidgets.Layout'):
            with self.assertRaises(KeyError):  # Enum raises KeyError for invalid values
                create_form_container(layout_type='INVALID_LAYOUT')
    
    def test_form_container_with_grid_missing_params(self):
        """Test grid layout with missing required parameters."""
        with patch('ipywidgets.VBox'), \
             patch('ipywidgets.Layout'):
            # Should raise ValueError if grid_columns is not provided for GRID layout
            with self.assertRaises(ValueError):
                create_form_container(layout_type=LayoutType.GRID)
    
    def test_form_container_with_custom_callbacks(self):
        """Test form container with custom save/reset callbacks."""
        with patch('ipywidgets.VBox'), \
             patch('ipywidgets.Layout'), \
             patch('ipywidgets.Button') as mock_button:
            
            # Setup mock button
            mock_save_btn = MagicMock()
            mock_reset_btn = MagicMock()
            mock_button.side_effect = [mock_save_btn, mock_reset_btn]
            
            save_called = False
            reset_called = False
            
            def on_save():
                nonlocal save_called
                save_called = True
                
            def on_reset():
                nonlocal reset_called
                reset_called = True
            
            form = create_form_container(
                on_save=on_save,
                on_reset=on_reset
            )
            
            # Verify buttons were created
            self.assertEqual(mock_button.call_count, 2)
            
            # Test save callback
            save_callback = mock_button.call_args_list[0].kwargs['on_click']
            save_callback(None)
            self.assertTrue(save_called)
            
            # Test reset callback
            reset_callback = mock_button.call_args_list[1].kwargs['on_click']
            reset_callback(None)
            self.assertTrue(reset_called)
    
    def test_form_container_with_callbacks(self):
        """Test form container with save and reset callbacks."""
        from smartcash.ui.components.form_container import create_form_container
        
        # Create mock callbacks
        save_callback = MagicMock()
        reset_callback = MagicMock()
        
        # Create form with callbacks
        form = create_form_container(
            on_save=save_callback,
            on_reset=reset_callback
        )
        
        # Trigger save button click
        save_button = form['save_button']
        save_button._click_handlers(save_button)
        
        # Verify save callback was called
        save_callback.assert_called_once()
        
        # Trigger reset button click
        reset_button = form['reset_button']
        reset_button._click_handlers(reset_button)
        
        # Verify reset callback was called
        reset_callback.assert_called_once()
    
    def test_form_container_styling(self):
        """Test form container with custom styling."""
        from smartcash.ui.components.form_container import create_form_container
        
        # Create form with custom styling
        form = create_form_container(
            container_margin="20px",
            container_padding="10px",
            form_spacing="15px"
        )
        
        # Verify container styling
        self.assertEqual(form['container'].layout.margin, '20px')
        self.assertEqual(form['container'].layout.padding, '10px')
        
        # Verify form container styling
        form_container = form['form_container']
        self.assertEqual(form_container.layout.margin, '0 0 15px 0')
    
    @patch('smartcash.ui.components.form_container.create_save_reset_buttons')
    @patch('smartcash.ui.components.form_container.widgets')
    def test_form_container_button_creation(self, mock_widgets, mock_create_buttons):
        """Test that save/reset buttons are created with correct parameters."""
        from smartcash.ui.components.form_container import create_form_container
        
        # Setup mock widgets
        mock_form_container = MagicMock()
        mock_button_container = MagicMock()
        
        # Create side effect for VBox to return different mocks for different calls
        def vbox_side_effect(*args, **kwargs):
            if not hasattr(vbox_side_effect, 'call_count'):
                vbox_side_effect.call_count = 0
            vbox_side_effect.call_count += 1
            
            if vbox_side_effect.call_count == 1:
                # First call is for the form container
                return mock_form_container
            else:
                # Second call is for the main container
                return MagicMock()
        
        mock_widgets.VBox.side_effect = vbox_side_effect
        
        # Setup mock buttons
        mock_save_button = MagicMock()
        mock_reset_button = MagicMock()
        
        mock_create_buttons.return_value = {
            'save_button': mock_save_button,
            'reset_button': mock_reset_button,
            'container': mock_button_container
        }
        
        # Create form with custom button labels and tooltips
        form = create_form_container(
            save_label="Save Changes",
            reset_label="Reset Form",
            save_tooltip="Click to save",
            reset_tooltip="Click to reset",
            alignment='center',
            with_sync_info=True,
            sync_message="Auto-save enabled"
        )
        
        # Verify create_save_reset_buttons was called with correct parameters
        args, kwargs = mock_create_buttons.call_args
        self.assertEqual(kwargs['save_label'], 'Save Changes')
        self.assertEqual(kwargs['reset_label'], 'Reset Form')
        self.assertEqual(kwargs['save_tooltip'], 'Click to save')
        self.assertEqual(kwargs['reset_tooltip'], 'Click to reset')
        self.assertEqual(kwargs['alignment'], 'center')
        self.assertTrue(kwargs['with_sync_info'])
        self.assertEqual(kwargs['sync_message'], 'Auto-save enabled')
        
        # Verify VBox was called twice (once for form container, once for main container)
        self.assertEqual(mock_widgets.VBox.call_count, 2)
        
        # Get the main container call (second call to VBox)
        main_container_call = mock_widgets.VBox.call_args_list[1]
        main_container_args = main_container_call[0]
        
        # Verify the main container has the form container and button container as children
        self.assertEqual(len(main_container_args[0]), 2)
        self.assertEqual(main_container_args[0][0], mock_form_container)  # Form container
        self.assertEqual(main_container_args[0][1], mock_button_container)  # Button container


class TestFormItem(unittest.TestCase):
    """Test cases for the FormItem class."""
    
    def test_form_item_initialization(self):
        """Test FormItem initialization with different parameters."""
        # Create a mock widget for testing
        mock_widget = MagicMock()
        mock_widget.layout = MagicMock()
        
        item = FormItem(mock_widget)
        self.assertIsInstance(item, FormItem)
        self.assertEqual(item.align_items, 'stretch')
        
        # Test with custom alignment
        item = FormItem(mock_widget, align_items='center')
        self.assertEqual(item.align_items, 'center')
        
        # Test with custom layout
        item = FormItem(
            mock_widget,
            width='100px',
            height='50px',
            align_self='center'
        )
        self.assertEqual(item.layout['width'], '100px')
        self.assertEqual(item.layout['height'], '50px')
        self.assertEqual(item.layout['align_self'], 'center')
    
    def test_form_item_align_items_validation(self):
        """Test align_items validation in FormItem."""
        # Test with valid values
        self.assertEqual(FormItem._validate_align_items('center'), 'center')
        self.assertEqual(FormItem._validate_align_items('flex-start'), 'flex-start')
        self.assertEqual(FormItem._validate_align_items('flex-end'), 'flex-end')
        
        # Test with aliases
        self.assertEqual(FormItem._validate_align_items('left'), 'flex-start')
        self.assertEqual(FormItem._validate_align_items('right'), 'flex-end')
        self.assertEqual(FormItem._validate_align_items('middle'), 'center')
        
        # Test with invalid values
        self.assertEqual(FormItem._validate_align_items('invalid'), 'stretch')
        self.assertEqual(FormItem._validate_align_items(''), 'stretch')
        self.assertEqual(FormItem._validate_align_items(None), 'stretch')


if __name__ == "__main__":
    unittest.main()
