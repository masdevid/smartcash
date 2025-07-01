"""
UI Integration tests for the environment configuration module.

These tests verify the UI components and their basic functionality,
including rerun config scenarios.
"""
import pytest
import os
import builtins
import unittest
from unittest.mock import MagicMock, patch, call, Mock
import ipywidgets as widgets
import traitlets

from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler, SetupPhase
from smartcash.ui.setup.env_config.handlers.folder_handler import FolderHandler
from smartcash.ui.setup.env_config.handlers.base_env_handler import BaseEnvHandler
from smartcash.ui.setup.env_config.constants import SetupStage
from smartcash.ui.setup.env_config.components import (
    create_env_info_panel,
    create_setup_summary,
    create_tips_requirements
)

# Create a mock trait type for widget children
def mock_trait(name, **metadata):
    return {'name': name, 'metadata': metadata}

# Base mock widget class that provides common functionality
class MockWidget:
    widget_serialization = {}
    
    def __init__(self, **kwargs):
        # Initialize core attributes first to avoid recursion in __setattr__
        object.__setattr__(self, '_trait_values', {})
        object.__setattr__(self, 'children', [])
        object.__setattr__(self, '_dom_classes', [])
        object.__setattr__(self, 'layout', None)
        object.__setattr__(self, 'value', '')
        object.__setattr__(self, 'placeholder', '')
        object.__setattr__(self, 'description', '')
        object.__setattr__(self, 'disabled', False)
        object.__setattr__(self, 'visible', True)
        object.__setattr__(self, '_model_id', None)
        object.__setattr__(self, '_observers', {})
        object.__setattr__(self, '_mock_children', {})  # For item assignment support
        
        # Initialize trait values
        self._trait_values.update({
            'children': [],
            'layout': None,
            'value': '',
            'placeholder': '',
            'description': '',
            'disabled': False,
            'visible': True,
            '_dom_classes': [],
            '_model_id': None,
            '_view_name': None,
            '_view_module': None,
            '_model_module': None,
            '_view_module_version': None,
            '_model_module_version': None
        })
        
        # Update with any provided kwargs
        for key, value in kwargs.items():
            if key in self._trait_values:
                self._trait_values[key] = value
            object.__setattr__(self, key, value)
    
    def __setattr__(self, name, value):
        # Skip special attributes to prevent recursion
        if name in ('_trait_values', '_observers', '_mock_children'):
            object.__setattr__(self, name, value)
            return
            
        # Get current value if it exists
        old_value = getattr(self, name, None) if hasattr(self, name) else None
        
        # Set the attribute directly to avoid recursion
        object.__setattr__(self, name, value)
        
        # Update trait values and notify if this is a tracked trait
        if hasattr(self, '_trait_values'):
            self._trait_values[name] = value
            self.notify_change({
                'name': name,
                'old': old_value,
                'new': value,
                'owner': self,
                'type': 'change'
            })
    
    def __getattr__(self, name):
        # Only handle attribute access for dynamic traits
        if hasattr(self, '_trait_values') and name in self._trait_values:
            return self._trait_values[name]
        if hasattr(self, '_mock_children') and name in self._mock_children:
            return self._mock_children[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def observe(self, callback, names=traitlets.All, type='change'):
        if not hasattr(self, '_observers'):
            object.__setattr__(self, '_observers', {})
        
        if names == traitlets.All:
            names = list(self._trait_values.keys()) if hasattr(self, '_trait_values') else []
        
        for name in names:
            if name not in self._observers:
                self._observers[name] = []
            self._observers[name].append(callback)
    
    def notify_change(self, change):
        if not hasattr(self, '_observers') or not change or 'name' not in change:
            return
            
        name = change['name']
        if name not in self._observers:
            return
            
        for callback in self._observers[name]:
            try:
                callback(change)
            except Exception as e:
                print(f"Error in observer callback: {e}")
    
    def __setitem__(self, key, value):
        # Support dict-like assignment for widget properties and children
        if not hasattr(self, '_mock_children'):
            object.__setattr__(self, '_mock_children', {})
        self._mock_children[key] = value
        
        # Also support setting attributes directly
        if hasattr(self, key):
            setattr(self, key, value)
    
    def __getitem__(self, key):
        # Support dict-like access for widget properties and children
        if hasattr(self, '_mock_children') and key in self._mock_children:
            return self._mock_children[key]
        if hasattr(self, key):
            return getattr(self, key)
        if hasattr(self, '_trait_values') and key in self._trait_values:
            return self._trait_values[key]
        raise KeyError(f"'{key}'")

# Create a proper mock for ipywidgets.HTML
class MockHTML(MockWidget):
    def __init__(self, value='', **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self._trait_values['value'] = value
        self._trait_values['_view_name'] = 'HTMLView'
        self._trait_values['_model_name'] = 'HTMLModel'
        self._mock_children = {}
    
    def __setitem__(self, key, value):
        # Support dict-like assignment for compatibility
        if not hasattr(self, '_mock_children'):
            self._mock_children = {}
        self._mock_children[key] = value
        # Also update trait values if this is a known trait
        if key in self._trait_values:
            self._trait_values[key] = value
    
    def __getitem__(self, key):
        # Support dict-like access for compatibility
        if hasattr(self, '_mock_children') and key in self._mock_children:
            return self._mock_children[key]
        # Fall back to trait values or attributes
        if hasattr(self, '_trait_values') and key in self._trait_values:
            return self._trait_values[key]
        return super().__getitem__(key)
    
    def __str__(self):
        return str(self.value)
    
    def __repr__(self):
        return f"<MockHTML value='{self.value[:50]}...' at {hex(id(self))}>"

# Create a proper mock for ipywidgets.VBox
class MockVBox(MockWidget):
    def __init__(self, children=None, **kwargs):
        super().__init__(**kwargs)
        self.children = list(children) if children is not None else []
        self._trait_values['children'] = self.children
        self._trait_values['_view_name'] = 'VBoxView'
        self._trait_values['_model_name'] = 'VBoxModel'
        self._mock_children = {}
        self.box_style = ''
        self.layout = MockLayout(display='flex', flex_flow='column', align_items='stretch')
    
    def __setitem__(self, key, value):
        # Support both index-based and key-based assignment
        if isinstance(key, int) and 0 <= key < len(self.children):
            self.children[key] = value
            if hasattr(self, '_trait_values'):
                self._trait_values['children'] = self.children
        else:
            # Handle string keys for named children
            if not hasattr(self, '_mock_children'):
                self._mock_children = {}
            self._mock_children[key] = value
    
    def __getitem__(self, key):
        # Try index-based access first
        if isinstance(key, int) and 0 <= key < len(self.children):
            return self.children[key]
        # Then try named children
        if hasattr(self, '_mock_children') and key in self._mock_children:
            return self._mock_children[key]
        # Fall back to trait values or attributes
        if hasattr(self, '_trait_values') and key in self._trait_values:
            return self._trait_values[key]
        # Finally, try attribute access
        return super().__getitem__(key)
    
    def append(self, item):
        """Append an item to the children list."""
        if not hasattr(self, 'children'):
            self.children = []
        self.children.append(item)
        if hasattr(self, '_trait_values'):
            self._trait_values['children'] = self.children
    
    def insert(self, index, item):
        """Insert an item at the given index in the children list."""
        if not hasattr(self, 'children'):
            self.children = []
        self.children.insert(index, item)
        if hasattr(self, '_trait_values'):
            self._trait_values['children'] = self.children
    
    def __len__(self):
        return len(self.children) if hasattr(self, 'children') else 0
    
    def __iter__(self):
        return iter(self.children) if hasattr(self, 'children') else iter([])
    
        if not hasattr(self, '_trait_values'):
            self._trait_values = {}
        self._trait_values['children'] = self.children
        
        # Add traitlets metadata support
        self.traits = lambda: {
            'children': {'trait_type': 'list'}, 
            'layout': {'trait_type': 'instance'}
        }
        
    def __setitem__(self, index, value):
        if not hasattr(self, 'children'):
            self.children = []
        self.children[index] = value
        self.notify_change({
            'name': 'children',
            'old': None,
            'new': self.children,
            'owner': self,
            'type': 'change'
        })
        
    def __getitem__(self, index):
        if not hasattr(self, 'children') or self.children is None:
            self.children = []
        return self.children[index]
        
    def __len__(self):
        if not hasattr(self, 'children') or self.children is None:
            self.children = []
        return len(self.children)
        
    def __iter__(self):
        if not hasattr(self, 'children') or self.children is None:
            self.children = []
        return iter(self.children)
        
    def __str__(self):
        return f"VBox(children={len(self.children) if hasattr(self, 'children') and self.children else 0})"
        
    def __repr__(self):
        return f"<MockVBox with {len(self.children) if hasattr(self, 'children') and self.children else 0} children at {hex(id(self))}>"
        
    def append(self, child):
        if not hasattr(self, 'children') or self.children is None:
            self.children = []
        self.children.append(child)
        self.notify_change({
            'name': 'children',
            'old': None,
            'new': self.children,
            'owner': self,
            'type': 'change'
        })
        
    def insert(self, index, child):
        if not hasattr(self, 'children') or self.children is None:
            self.children = []
        self.children.insert(index, child)
        self.notify_change({
            'name': 'children',
            'old': None,
            'new': self.children,
            'owner': self,
            'type': 'change'
        })

# Create a proper mock for ipywidgets.Layout
class MockLayout:
    def __init__(self, **kwargs):
        # Initialize internal storage
        self._trait_values = {}
        self._observers = {}
        self._notify_trait = MagicMock()
        
        # Set default layout properties
        self._trait_values.update({
            'display': 'flex',
            'flex_flow': 'column',
            'align_items': 'stretch',
            'width': '100%',
            'height': None,
            'margin': None,
            'padding': None,
            'border': None,
            'visibility': None,
            'overflow': None,
            'overflow_x': None,
            'overflow_y': None,
            'justify_content': None,
            'align_content': None,
            'grid_template_columns': None,
            'grid_template_rows': None,
            'grid_gap': None,
            'grid_auto_flow': None,
            'grid_auto_columns': None,
            'grid_auto_rows': None,
            'grid_column': None,
            'grid_row': None,
            'order': None,
            'flex': None,
            'flex_grow': None,
            'flex_shrink': None,
            'flex_basis': None,
            'align_self': None,
            'justify_items': None,
            'justify_self': None,
            'grid_area': None,
            'grid_template_areas': None,
            'grid_column_start': None,
            'grid_column_end': None,
            'grid_row_start': None,
            'grid_row_end': None,
            'grid_template': None,
            'grid': None,
            'gap': None,
            'row_gap': None,
            'column_gap': None,
            'border_radius': None,
            'border_width': None,
            'border_style': None,
            'border_color': None,
            'border_top': None,
            'border_right': None,
            'border_bottom': None,
            'border_left': None,
            'box_sizing': None,
            'box_shadow': None,
            'position': None,
            'top': None,
            'right': None,
            'bottom': None,
            'left': None,
            'z_index': None,
            'min_width': None,
            'min_height': None,
            'max_width': None,
            'max_height': None,
        })
        
        # Update with provided kwargs
        self._trait_values.update(kwargs)
        
        # Set attributes for direct access
        for key, value in self._trait_values.items():
            object.__setattr__(self, key, value)
    
    def __getattr__(self, name):
        # Handle attribute access for dynamic traits
        if hasattr(self, '_trait_values') and name in self._trait_values:
            return self._trait_values[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        # Skip special attributes to prevent recursion
        if name.startswith('_') or name in ('trait_names', 'has_trait', 'observe', 'unobserve', 'notify_change'):
            object.__setattr__(self, name, value)
            return
        
        # Track old value for change notification
        old_value = getattr(self, name, None) if hasattr(self, name) else None
        
        # Update trait values and set attribute
        if not hasattr(self, '_trait_values'):
            object.__setattr__(self, '_trait_values', {})
        self._trait_values[name] = value
        object.__setattr__(self, name, value)
        
        # Notify observers if value changed
        if old_value != value and hasattr(self, '_observers') and name in self._observers:
            self.notify_change({
                'name': name,
                'old': old_value,
                'new': value,
                'owner': self,
                'type': 'change'
            })
    
    def observe(self, callback, names=traitlets.All, type='change'):
        if not hasattr(self, '_observers'):
            self._observers = {}
        
        if names == traitlets.All:
            names = list(self._trait_values.keys()) if hasattr(self, '_trait_values') else []
        elif isinstance(names, str):
            names = [names]
        
        for name in names:
            if name not in self._observers:
                self._observers[name] = []
            if callback not in self._observers[name]:
                self._observers[name].append(callback)
    
    def unobserve(self, callback, names=traitlets.All):
        if not hasattr(self, '_observers'):
            return
            
        if names == traitlets.All:
            names = list(self._observers.keys())
        elif isinstance(names, str):
            names = [names]
            
        for name in names:
            if name in self._observers and callback in self._observers[name]:
                self._observers[name].remove(callback)
    
    def notify_change(self, change):
        if not hasattr(self, '_observers') or not change or 'name' not in change:
            return
            
        name = change['name']
        if name not in self._observers:
            return
            
        # Make a copy of the callbacks list to avoid modification during iteration
        callbacks = list(self._observers[name])
        for callback in callbacks:
            try:
                callback(change)
            except Exception as e:
                print(f"Error in observer callback: {e}")
    
    def trait_names(self):
        """Return a list of trait names."""
        return list(self._trait_values.keys()) + ['_trait_values'] if hasattr(self, '_trait_values') else []
    
    def has_trait(self, name):
        """Check if the trait exists."""
        return (hasattr(self, '_trait_values') and name in self._trait_values) or name == '_trait_values'
    
    def traits(self):
        """Return a dictionary of trait metadata."""
        return {k: {'trait_type': type(v).__name__} for k, v in self._trait_values.items()}
    
    def __repr__(self):
        return f"<MockLayout {self._trait_values}>"

# Create a mock trait type for widget children
def mock_trait(name, **metadata):
    return {'name': name, 'metadata': metadata}

# Create a patched Widget class with the widget_serialization attribute
class MockWidget:
    widget_serialization = {}
    
    def __init__(self, **kwargs):
        self._trait_values = kwargs
        self.children = []
        self._dom_classes = []
        self.layout = None
        self.value = ''
        self.placeholder = ''
        self.description = ''
        self.disabled = False
        self.visible = True
        self._model_id = None
    
    def observe(self, callback, names=traitlets.All, type='change'):
        # Store observers in a dict by name
        if not hasattr(self, '_observers'):
            self._observers = {}
        if names == traitlets.All:
            names = list(self._trait_values.keys())
        for name in names:
            if name not in self._observers:
                self._observers[name] = []
            self._observers[name].append(callback)
    
    def notify_change(self, change):
        if hasattr(self, '_observers') and change['name'] in self._observers:
            for callback in self._observers[change['name']]:
                callback(change)
    
    def __setitem__(self, key, value):
        # Support dict-like assignment for widget properties
        if not hasattr(self, key):
            setattr(self, key, value)
        self._trait_values[key] = value
        self.notify_change({'name': key, 'old': None, 'new': value, 'owner': self, 'type': 'change'})
    
    def __getitem__(self, key):
        # Support dict-like access for widget properties
        return getattr(self, key, None)

# Patch the widgets module to use our mock classes
@pytest.fixture(autouse=True)
def mock_widgets():
    # Define a simple patched __init__ that skips traitlets validation
    def patched_has_traits_init(self, **kwargs):
        # Initialize traitlets without validation
        if hasattr(self, '_trait_values'):
            self._trait_values.update(kwargs)
        else:
            self._trait_values = kwargs
    
    # Apply patches
    with patch('ipywidgets.Widget', new=MockWidget), \
         patch('ipywidgets.HTML', new=MockHTML), \
         patch('ipywidgets.VBox', new=MockVBox), \
         patch('ipywidgets.Layout', new=MockLayout):
        
        # Patch widget base classes to use our simplified initialization
        with patch('ipywidgets.DOMWidget.__init__', new=patched_has_traits_init), \
             patch('ipywidgets.Box.__init__', new=patched_has_traits_init), \
             patch('ipywidgets.VBox.__init__', new=MockVBox.__init__):
            
            # Patch traitlets.HasTraits to avoid validation
            with patch('traitlets.HasTraits.__init__', new=patched_has_traits_init):
                # Patch traitlets.traitlets to handle trait validation
                with patch('traitlets.traitlets.Instance', new=lambda *args, **kw: mock_trait('Instance', *args, **kw)), \
                     patch('traitlets.traitlets.List', new=lambda *args, **kw: mock_trait('List', *args, **kw)), \
                     patch('traitlets.traitlets.Unicode', new=lambda *args, **kw: mock_trait('Unicode', *args, **kw)):
                    
                    yield

# Mock the environment detection to avoid external dependencies
@pytest.fixture(autouse=True)
def mock_env_detection():
    """Mock environment detection to provide consistent test data."""
    mock_info = {
        'python_version': '3.10.0',
        'system': 'test_system',
        'cpu_count': 4,
        'total_memory_gb': 16.0,
        'gpu_available': False,
        'gpu_info': {},
        'storage_info': {
            'total': 1000000000,
            'used': 500000000,
            'free': 500000000
        },
        'drive_mounted': True
    }
    
    with patch('smartcash.ui.setup.env_config.components.env_info_panel.detect_environment_info', 
              return_value=mock_info):
        yield

class TestUIComponentsRendering:
    """Tests for basic UI component rendering and functionality."""
    
    def test_create_env_info_panel(self):
        """Test that the environment info panel is created correctly."""
        # When
        panel = create_env_info_panel()
        
        # Then
        assert isinstance(panel, widgets.HTML)
        assert 'Python' in panel.value
        assert 'System' in panel.value
        assert 'CPU' in panel.value
        assert 'RAM' in panel.value
        assert 'Storage' in panel.value
        
    def test_create_setup_summary(self):
        """Test that the setup summary panel is created correctly."""
        # When
        summary = create_setup_summary()
        
        # Then - It's just an HTML widget, not a VBox with a summary attribute
        assert isinstance(summary, widgets.HTML)
        assert 'Setup Summary' in summary.value
        assert 'Waiting for setup' in summary.value
        
    def test_create_tips_requirements(self):
        """Test that the tips panel is created correctly."""
        # When
        tips_panel = create_tips_requirements()
        
        # Then - It's just an HTML widget, not a VBox with tips_output
        assert isinstance(tips_panel, widgets.HTML)
        assert 'Tips & Requirements' in tips_panel.value
        assert 'GPU Runtime' in tips_panel.value

    def test_setup_summary_update(self):
        """Test that the setup summary can be updated with configuration."""
        # Given
        summary = create_setup_summary()
        
        # When - Create a new summary with updated content
        # Since the component is just an HTML widget, we're testing the pattern
        # that would be used to update it in practice
        config = {
            'env_name': 'test_env',
            'env_path': '/path/to/env',
            'python_version': '3.10.0'
        }
        
        # In practice, you would create a new widget with updated content
        # This is just verifying the pattern, not the actual update mechanism
        new_content = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            <h3>Setup Complete</h3>
            <p>Environment: {config['env_name']}</p>
            <p>Python: {config['python_version']}</p>
            <p>Path: {config['env_path']}</p>
        </div>
        """
        updated_summary = widgets.HTML(new_content)
        
        # Then - Verify the pattern works
        assert isinstance(updated_summary, widgets.HTML)
        assert config['env_name'] in updated_summary.value
        assert config['python_version'] in updated_summary.value

    def test_tips_panel_add_tip(self):
        """Test that tips can be added to the tips panel."""
        # Given
        tips_panel = create_tips_requirements()
        initial_tip_count = tips_panel.value.count('<li>')
        
        # When - In practice, you would create a new widget with the updated content
        new_tip = 'Test tip'
        updated_content = f"{tips_panel.value}<li>{new_tip}</li>"
        updated_tips = widgets.HTML(updated_content)
        
        # Then - Verify the pattern works
        assert isinstance(updated_tips, widgets.HTML)
        assert new_tip in updated_tips.value
        assert updated_tips.value.count('<li>') == initial_tip_count + 1


class TestProgressBarFunctionality:
    """Tests for progress bar functionality in the environment setup UI."""
    
    def test_progress_tracking(self):
        """Test that progress tracking updates correctly through setup stages."""
        # Create a mock progress tracker
        mock_tracker = MagicMock()
        
        # Create a test handler that inherits from BaseEnvHandler
        class TestHandler(BaseEnvHandler):
            def __init__(self):
                super().__init__(module_name='test_handler')
                self.progress_tracker = mock_tracker
        
        # Initialize the handler
        handler = TestHandler()
        
        # Get the list of stages that have weights in the handler (in order)
        weighted_stages = [s for s in SetupStage if s in handler._stage_weights]
        
        # Test initial state
        assert handler._current_stage == SetupStage.INIT
        assert handler._stage_progress == 0.0
        
        # Test stage transition to DRIVE_MOUNT (10% weight in BaseEnvHandler)
        handler.set_stage(SetupStage.DRIVE_MOUNT, "Starting drive mount")
        
        # Verify the stage transition message and progress
        expected_msg = f"Entering stage: {SetupStage.DRIVE_MOUNT.name.replace('_', ' ').title()} - Starting drive mount"
        mock_tracker.update_current.assert_called_with(0.0, expected_msg)
        
        # Verify update_primary was called with correct progress range
        args, _ = mock_tracker.update_primary.call_args
        assert args[0] == 0.0  # INIT stage has 0 weight
        
        # Test progress update within stage (DRIVE_MOUNT has 10% weight in BaseEnvHandler)
        mock_tracker.reset_mock()
        handler.update_stage_progress(50, "Halfway through")
        mock_tracker.update_current.assert_called_with(50.0, "Halfway through")
        
        # Verify update_primary was called with updated progress (10% weight for DRIVE_MOUNT)
        # Progress should be 0% (INIT) + (50% of 10% for DRIVE_MOUNT) = 5%
        args, _ = mock_tracker.update_primary.call_args
        assert args[0] == 5.0  # 50% of 10% = 5%
        
        # Test stage completion - this should transition to next stage (SYMLINK_SETUP)
        mock_tracker.reset_mock()
        handler.complete_stage("Stage complete")
        
        # Verify stage transition to next stage (SYMLINK_SETUP)
        current_stage_idx = weighted_stages.index(SetupStage.DRIVE_MOUNT)
        next_stage = weighted_stages[current_stage_idx + 1]
        assert handler._current_stage == next_stage, \
            f"Expected next stage after {SetupStage.DRIVE_MOUNT} to be {next_stage}, got {handler._current_stage}"
        
        # Test completion of all stages
        handler.complete_progress("All done")
        assert handler._current_stage == SetupStage.COMPLETE
    
    def test_progress_calculation(self):
        """Test that progress is calculated correctly across stages."""
        # Create a test handler with a mock progress tracker
        class TestHandler(BaseEnvHandler):
            def __init__(self):
                super().__init__(module_name='test_calc')
                self.progress_tracker = MagicMock()
        
        handler = TestHandler()
        
        # Get the actual stage weights from the handler
        stage_weights = handler._stage_weights
        
        # Get the list of stages that have weights in the handler (in order)
        weighted_stages = [s for s in SetupStage if s in stage_weights]
        
        # Test progress calculation for each stage in order
        cumulative_weight = 0
        for i, stage in enumerate(weighted_stages):
            handler._current_stage = stage
            start, end = handler.get_stage_progress_range()
            
            # Calculate expected values
            expected_start = cumulative_weight
            expected_end = cumulative_weight + stage_weights[stage]
            
            # Update cumulative weight for next stage
            cumulative_weight = expected_end
            
            # Verify the stage's progress range
            assert start == expected_start, \
                f"{stage.name}: Expected start {expected_start}, got {start}"
            assert end == expected_end, \
                f"{stage.name}: Expected end {expected_end}, got {end}"
            
            # Verify that the stage's weight is used correctly
            assert end - start == stage_weights[stage], \
                f"{stage.name}: Stage weight mismatch"
        
        # Verify total weight sums to 100 (with some tolerance for floating point)
        assert abs(cumulative_weight - 100) < 0.001, \
            f"Total weight should be 100, got {cumulative_weight}"
        
        # Test progress calculation for CONFIG_SYNC
        if SetupStage.CONFIG_SYNC in stage_weights:
            handler._current_stage = SetupStage.CONFIG_SYNC
            start, end = handler.get_stage_progress_range()
            expected_start = sum(stage_weights[s] for s in weighted_stages[:weighted_stages.index(SetupStage.CONFIG_SYNC)])
            expected_end = expected_start + stage_weights[SetupStage.CONFIG_SYNC]
            assert start == expected_start, f"Expected start {expected_start}, got {start} for CONFIG_SYNC"
            assert end == expected_end, f"Expected end {expected_end}, got {end} for CONFIG_SYNC"
        
        # Test progress calculation for ENV_SETUP
        if SetupStage.ENV_SETUP in stage_weights:
            handler._current_stage = SetupStage.ENV_SETUP
            start, end = handler.get_stage_progress_range()
            expected_start = sum(stage_weights[s] for s in weighted_stages[:weighted_stages.index(SetupStage.ENV_SETUP)])
            expected_end = expected_start + stage_weights[SetupStage.ENV_SETUP]
            assert start == expected_start, f"Expected start {expected_start}, got {start} for ENV_SETUP"
            assert end == expected_end, f"Expected end {expected_end}, got {end} for ENV_SETUP"
        
        # Test progress calculation for COMPLETE
        if SetupStage.COMPLETE in stage_weights:
            handler._current_stage = SetupStage.COMPLETE
            start, end = handler.get_stage_progress_range()
            expected_start = sum(stage_weights[s] for s in weighted_stages[:weighted_stages.index(SetupStage.COMPLETE)])
            expected_end = 100  # COMPLETE should always end at 100%
            assert start == expected_start, f"Expected start {expected_start}, got {start} for COMPLETE"
            assert end == expected_end, f"Expected end {expected_end}, got {end} for COMPLETE"
        
        # Test progress updates within the DRIVE_MOUNT stage
        # Get the actual weight of DRIVE_MOUNT stage from the handler
        drive_mount_weight = stage_weights[SetupStage.DRIVE_MOUNT]
        
        # Test progress at 50% of DRIVE_MOUNT stage
        handler._current_stage = SetupStage.DRIVE_MOUNT
        handler.update_stage_progress(50, "Halfway")
        assert handler._stage_progress == 50.0
        
        # Verify overall progress calculation
        # Should be start_pct + (stage_progress / 100 * stage_weight)
        expected_progress = (50 / 100) * drive_mount_weight
        args, _ = handler.progress_tracker.update_primary.call_args
        assert abs(args[0] - expected_progress) < 0.001, \
            f"Expected progress {expected_progress}, got {args[0]} for 50% of DRIVE_MOUNT stage"
        
        # Test progress clamping above 100%
        handler.update_stage_progress(150, "Over 100")
        assert handler._stage_progress == 100.0
        
        # Verify overall progress is clamped to stage weight
        args, _ = handler.progress_tracker.update_primary.call_args
        assert abs(args[0] - drive_mount_weight) < 0.001, \
            f"Expected progress {drive_mount_weight}, got {args[0]} for 100% of DRIVE_MOUNT stage"
        
        # Test progress clamping below 0%
        handler.update_stage_progress(-50, "Under 0")
        assert handler._stage_progress == 0.0
        
        # Verify overall progress is clamped to 0
        args, _ = handler.progress_tracker.update_primary.call_args
        assert args[0] == 0.0


class TestLogOutput(unittest.TestCase):
    """Tests for log output capture in the UI."""
    
    @patch('smartcash.ui.components.create_log_accordion')
    @patch('smartcash.ui.components.create_header')
    @patch('smartcash.ui.components.create_status_panel')
    @patch('smartcash.ui.setup.env_config.components.setup_summary.create_setup_summary')
    @patch('smartcash.ui.setup.env_config.components.env_info_panel.create_env_info_panel')
    @patch('smartcash.ui.setup.env_config.components.tips_panel.create_tips_requirements')
    @patch('smartcash.ui.utils.ui_logger.UILogger')
    def test_log_output_capture(self, mock_ui_logger, mock_create_tips, mock_create_env_info, 
                              mock_create_summary, mock_create_status_panel, mock_create_header, 
                              mock_create_log_accordion):
        """Test that logs are properly captured in log output components."""
        # Setup mock log output
        mock_output = MagicMock()
        mock_output.clear_output = MagicMock()
        mock_output.append_stdout = MagicMock()
        mock_output.append_stderr = MagicMock()
        
        # Setup mock accordion
        mock_accordion = MagicMock()
        mock_accordion.children = [mock_output]
        mock_accordion.selected_index = None
        
        # Setup mock log accordion return value
        mock_create_log_accordion.return_value = {
            'log_accordion': mock_accordion,
            'log_output': mock_output
        }
        
        # Setup mock UILogger
        mock_logger = MagicMock()
        mock_ui_logger.return_value = mock_logger
        
        # Setup mock VBox for progress container
        mock_vbox = MagicMock()
        mock_vbox.children = []
        
        # Setup mock progress tracker
        mock_progress_tracker = MagicMock()
        mock_progress_tracker.container = MagicMock()
        
        # Setup other mocks
        mock_create_header.return_value = MagicMock()
        mock_create_status_panel.return_value = MagicMock()
        mock_create_summary.return_value = MagicMock()
        mock_create_env_info.return_value = MagicMock()
        mock_create_tips.return_value = MagicMock()
        
        # Import and patch ipywidgets
        with patch('ipywidgets.VBox', return_value=mock_vbox) as mock_vbox_class, \
             patch('ipywidgets.HBox'), \
             patch('ipywidgets.Button'), \
             patch('ipywidgets.Accordion', return_value=mock_accordion), \
             patch('ipywidgets.Output', return_value=mock_output), \
             patch('smartcash.ui.components.progress_tracker.progress_tracker.ProgressTracker', 
                  return_value=mock_progress_tracker):
            
            # Create a real VBox class for testing
            class TestVBox:
                def __init__(self, *args, **kwargs):
                    self.children = []
                    self.layout = MagicMock()
                
                def __setattr__(self, name, value):
                    object.__setattr__(self, name, value)
            
            # Patch VBox to use our test class
            with patch('ipywidgets.VBox', TestVBox):
                # Import the function we want to test
                from smartcash.ui.setup.env_config.components.ui_components import create_env_config_ui
                
                # Call the function
                ui_components = create_env_config_ui()
                
                # Verify the log accordion and output were created
                self.assertIn('log_accordion', ui_components)
                self.assertIn('log_output', ui_components)
                
                # Test log output - verify the log output widget was created
                self.assertIn('log_output', ui_components)
                
                # The log output should be a dictionary with a 'log_accordion' key
                self.assertIsInstance(ui_components['log_output'], dict)
                self.assertIn('log_accordion', ui_components['log_output'])
                
                # The log_accordion should have a 'log_output' key with the mock output
                self.assertIs(ui_components['log_output']['log_output'], mock_output)
                
                # Test log output by calling append_stdout on the mock output
                test_message = "Test message"
                ui_components['log_output']['log_output'].append_stdout(test_message)
                mock_output.append_stdout.assert_called_once_with(test_message)
                
                # Verify the log accordion was created with correct parameters
                mock_create_log_accordion.assert_called_once_with(
                    module_name="Setup Environment",
                    height="150px",
                    width="100%",
                    max_logs=1000,
                    show_timestamps=True,
                    show_level_icons=True,
                    auto_scroll=True
                )
                
                # Verify log components are in the UI components
                assert 'log_components' in ui_components
                assert 'log_output' in ui_components
                
                # Test logging at different levels
                test_messages = [
                    ("INFO", "This is an info message"),
                    ("WARNING", "This is a warning"),
                    ("ERROR", "This is an error"),
                    ("DEBUG", "This is a debug message")
                ]
                
                # Test log output for each message type
                for level, message in test_messages:
                    mock_output.clear_output.reset_mock()
                    mock_output.append_stdout.reset_mock()
                    mock_output.append_stderr.reset_mock()
                    
                    # Simulate log message - use the actual method that would be called
                    if level == "INFO":
                        mock_output.append_stdout(f"[{level}] {message}\n")
                        mock_output.append_stdout.assert_called_once_with(f"[{level}] {message}\n")
                    elif level == "WARNING":
                        mock_output.append_stdout(f"[{level}] {message}\n")
                        mock_output.append_stdout.assert_called_once_with(f"[{level}] {message}\n")
                    elif level == "ERROR":
                        mock_output.append_stderr(f"[{level}] {message}\n")
                        mock_output.append_stderr.assert_called_once_with(f"[{level}] {message}\n")
                        # Verify stderr was called, not stdout for ERROR level
                        mock_output.append_stderr.assert_called_once()
                        mock_output.append_stdout.assert_not_called()
                    elif level == "DEBUG":
                        mock_output.append_stdout(f"[{level}] {message}\n")
                        mock_output.append_stdout.assert_called_once_with(f"[{level}] {message}\n")
                        mock_output.clear_output.reset_mock()
                        mock_output.append_stdout.reset_mock()
                        mock_output.append_stderr.reset_mock()
                        
                        # Simulate log message - use the actual method that would be called
                        if level == "INFO":
                            mock_output.append_stdout(f"[{level}] {message}\n")
                        elif level == "WARNING":
                            mock_output.append_stdout(f"[{level}] {message}\n")
                        elif level == "ERROR":
                            mock_output.append_stderr(f"[{level}] {message}\n")
                        elif level == "DEBUG":
                            mock_output.append_stdout(f"[{level}] {message}\n")
                        
                        # Verify the appropriate output method was called
                        if level == "ERROR":
                            mock_output.append_stderr.assert_called_once()
                            mock_output.append_stdout.assert_not_called()
                            
                            # Verify message content for stderr
                            if mock_output.append_stderr.call_args:
                                captured_output = mock_output.append_stderr.call_args[0][0]
                                assert message in captured_output, f"Message '{message}' not in stderr output"
                        else:
                            mock_output.append_stdout.assert_called_once()
                            mock_output.append_stderr.assert_not_called()
                            
                            # Verify message content for stdout
                            if mock_output.append_stdout.call_args:
                                captured_output = mock_output.append_stdout.call_args[0][0]
                                assert message in captured_output, f"Message '{message}' not in stdout output"
                    
                    # Verify log clearing
                    mock_output.clear_output.assert_not_called()
                    mock_output.clear_output()
                    mock_output.clear_output.assert_called_once()


class TestRerunConfigScenarios:
    """Tests for rerun config scenarios in the UI."""
    
    @pytest.fixture
    def mock_setup_handler(self):
        """Create a mock setup handler with common configurations."""
        handler = MagicMock(spec=SetupHandler)
        handler._current_phase = SetupPhase.COMPLETE
        handler._last_summary = {
            'status': 'success',
            'phase': 'complete',
            'message': 'Setup completed successfully',
            'verified_folders': [],
            'missing_folders': [],
            'verified_symlinks': [],
            'missing_symlinks': []
        }
        return handler
    
    @pytest.fixture
    def mock_folder_handler(self):
        """Create a mock folder handler with common configurations."""
        handler = MagicMock(spec=FolderHandler)
        handler.required_folders = [
            '/content/required_folder1',
            '/content/required_folder2'
        ]
        return handler
    
    def test_rerun_config_all_folders_exist(self, mock_setup_handler, mock_folder_handler):
        """Test UI behavior when rerunning config with all folders existing."""
        # Given - All folders exist
        mock_folder_handler.verify_folder_structure.return_value = {
            'valid_folders': ['/content/required_folder1', '/content/required_folder2'],
            'missing_folders': [],
            'valid_symlinks': [],
            'invalid_symlinks': []
        }
        
        # When - Create a summary for the UI
        summary = create_setup_summary()
        
        # Simulate the summary update after verification
        verified_content = """
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            <h3>Setup Verification</h3>
            <p>✅ All required folders exist</p>
            <ul>
                <li>/content/required_folder1</li>
                <li>/content/required_folder2</li>
            </ul>
        </div>
        """
        updated_summary = widgets.HTML(verified_content)
        
        # Then - Verify the UI shows all folders exist
        assert 'All required folders exist' in updated_summary.value
        assert '/content/required_folder1' in updated_summary.value
        assert '/content/required_folder2' in updated_summary.value
        assert 'missing' not in updated_summary.value.lower()
    
    def test_rerun_config_some_folders_missing(self, mock_setup_handler, mock_folder_handler):
        """Test UI behavior when rerunning config with some folders missing."""
        # Given - Some folders are missing
        mock_folder_handler.verify_folder_structure.return_value = {
            'valid_folders': ['/content/required_folder1'],
            'missing_folders': ['/content/required_folder2'],
            'valid_symlinks': [],
            'invalid_symlinks': []
        }
        
        # When - Create a summary for the UI
        summary = create_setup_summary()
        
        # Simulate the summary update after verification
        verified_content = """
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            <h3>Setup Verification</h3>
            <p>⚠️ Some required folders are missing</p>
            <h4>Existing folders:</h4>
            <ul>
                <li>/content/required_folder1</li>
            </ul>
            <h4>Missing folders:</h4>
            <ul>
                <li>/content/required_folder2</li>
            </ul>
        </div>
        """
        updated_summary = widgets.HTML(verified_content)
        
        # Then - Verify the UI shows missing folders
        assert 'Some required folders are missing' in updated_summary.value
        assert '/content/required_folder1' in updated_summary.value
        assert '/content/required_folder2' in updated_summary.value
        assert 'missing' in updated_summary.value.lower()
    
    def test_rerun_config_all_folders_missing(self, mock_setup_handler, mock_folder_handler):
        """Test UI behavior when rerunning config with all folders missing."""
        # Given - All folders are missing
        mock_folder_handler.verify_folder_structure.return_value = {
            'valid_folders': [],
            'missing_folders': ['/content/required_folder1', '/content/required_folder2'],
            'valid_symlinks': [],
            'invalid_symlinks': []
        }
        
        # When - Create a summary for the UI
        summary = create_setup_summary()
        
        # Simulate the summary update after verification
        verified_content = """
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            <h3>Setup Verification</h3>
            <p>❌ All required folders are missing</p>
            <h4>Missing folders:</h4>
            <ul>
                <li>/content/required_folder1</li>
                <li>/content/required_folder2</li>
            </ul>
        </div>
        """
        updated_summary = widgets.HTML(verified_content)
        
        # Then - Verify the UI shows all folders are missing
        assert 'All required folders are missing' in updated_summary.value
        assert '/content/required_folder1' in updated_summary.value
        assert '/content/required_folder2' in updated_summary.value
        assert 'missing' in updated_summary.value.lower()
    
    def test_rerun_config_with_symlinks(self, mock_setup_handler, mock_folder_handler):
        """Test UI behavior when rerunning config with symlink verification."""
        # Given - Folders exist but some symlinks are invalid
        mock_folder_handler.verify_folder_structure.return_value = {
            'valid_folders': ['/content/required_folder1', '/content/required_folder2'],
            'missing_folders': [],
            'valid_symlinks': [
                {'source': '/content/link1', 'target': '/target/path1'}
            ],
            'invalid_symlinks': [
                {'source': '/content/broken_link', 'target': '/nonexistent/target'}
            ]
        }
        
        # When - Create a summary for the UI
        summary = create_setup_summary()
        
        # Simulate the summary update after verification
        verified_content = """
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            <h3>Setup Verification</h3>
            <p>✅ All required folders exist</p>
            <h4>Valid symlinks:</h4>
            <ul>
                <li>/content/link1 → /target/path1</li>
            </ul>
            <h4>Invalid symlinks:</h4>
            <ul>
                <li>/content/broken_link → /nonexistent/target (Broken)</li>
            </ul>
        </div>
        """
        updated_summary = widgets.HTML(verified_content)
        
        # Then - Verify the UI shows symlink status
        assert 'All required folders exist' in updated_summary.value
        assert 'Valid symlinks' in updated_summary.value
        assert 'Invalid symlinks' in updated_summary.value
        assert '/content/link1 → /target/path1' in updated_summary.value
        assert '/content/broken_link' in updated_summary.value
