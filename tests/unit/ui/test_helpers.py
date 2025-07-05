"""Test helpers for UI component tests."""
import unittest
from unittest.mock import MagicMock, patch
import contextlib
import ipywidgets as widgets

@contextlib.contextmanager
def patch_ui_dependencies():
    """Patch common UI dependencies like error handling and base components."""
    with patch('smartcash.ui.components.base_component.ErrorContext', new_callable=MagicMock), \
         patch('smartcash.ui.components.base_component.BaseUIComponent', new_callable=MagicMock):
        yield

@contextlib.contextmanager
def patch_ipywidgets_for_form_container():
    """
    Context manager to patch ipywidgets for form_container tests.

    This patch targets 'smartcash.ui.components.form_container.widgets'
    and uses real ipywidgets classes while still tracking their usage.
    """
    # Create wrapper classes that track instantiation
    class TrackedVBox(widgets.VBox):
        def __init__(self, *args, **kwargs):
            self._tracked_args = args
            self._tracked_kwargs = kwargs
            super().__init__(*args, **kwargs)
    
    class TrackedHBox(widgets.HBox):
        def __init__(self, *args, **kwargs):
            self._tracked_args = args
            self._tracked_kwargs = kwargs
            super().__init__(*args, **kwargs)
    
    class TrackedGridBox(widgets.GridBox):
        def __init__(self, *args, **kwargs):
            self._tracked_args = args
            self._tracked_kwargs = kwargs
            super().__init__(*args, **kwargs)
    
    class TrackedLayout(widgets.Layout):
        def __init__(self, *args, **kwargs):
            self._tracked_args = args
            self._tracked_kwargs = kwargs
            # Initialize with all potential layout attributes to avoid AttributeErrors
            self.gap = kwargs.pop('gap', None)
            self.padding = kwargs.pop('padding', None)
            self.flex_flow = kwargs.pop('flex_flow', None)
            self.grid_template_columns = kwargs.pop('grid_template_columns', None)
            self.grid_template_areas = kwargs.pop('grid_template_areas', None)
            self.grid_auto_flow = kwargs.pop('grid_auto_flow', None)
            
            # Call parent's __init__ with remaining kwargs
            super().__init__(*args, **kwargs)
            
            # Store any additional layout attributes that were set by the parent
            for name, value in kwargs.items():
                if not hasattr(self, name):
                    setattr(self, name, value)
            
        def __setattr__(self, name, value):
            # Allow dynamic attribute assignment for any layout property
            if name not in ['_tracked_args', '_tracked_kwargs'] and not name.startswith('trait_'):
                object.__setattr__(self, name, value)
            super().__setattr__(name, value)
    
    # Create mocks that will track usage but return real widget instances
    mock_vbox = MagicMock(side_effect=TrackedVBox)
    mock_hbox = MagicMock(side_effect=TrackedHBox)
    mock_gridbox = MagicMock(side_effect=TrackedGridBox)
    mock_layout = MagicMock(side_effect=TrackedLayout)
    mock_widget = MagicMock(side_effect=widgets.Widget)
    
    # Patch the widgets module in form_container
    with patch('smartcash.ui.components.form_container.widgets.VBox', mock_vbox), \
         patch('smartcash.ui.components.form_container.widgets.HBox', mock_hbox), \
         patch('smartcash.ui.components.form_container.widgets.GridBox', mock_gridbox), \
         patch('smartcash.ui.components.form_container.widgets.Layout', mock_layout), \
         patch('smartcash.ui.components.form_container.widgets.Widget', mock_widget):
        
        yield {
            "VBox": mock_vbox,
            "HBox": mock_hbox,
            "GridBox": mock_gridbox,
            "Layout": mock_layout,
            "Widget": mock_widget,
            "TrackedVBox": TrackedVBox,
            "TrackedHBox": TrackedHBox,
            "TrackedGridBox": TrackedGridBox,
            "TrackedLayout": TrackedLayout
        }

class BaseTestCase(unittest.TestCase):
    """Base class for UI component tests with common patching."""

    def setUp(self):
        """Set up patches for UI dependencies and ipywidgets."""
        super().setUp()
        
        # Start UI dependencies patch context manager
        self.ui_patcher = patch_ui_dependencies()
        self.ui_patcher_ctx = self.ui_patcher.__enter__()
        
        # Start ipywidgets patch context manager
        self.ipywidgets_patcher = patch_ipywidgets_for_form_container()
        mock_widgets = self.ipywidgets_patcher.__enter__()
        
        # Store the actual widget classes for isinstance checks
        self.VBox = mock_widgets['TrackedVBox']
        self.HBox = mock_widgets['TrackedHBox']
        self.GridBox = mock_widgets['TrackedGridBox']
        self.Layout = mock_widgets['TrackedLayout']
        self.Widget = widgets.Widget  # Keep the original Widget for isinstance checks
        
        # Store the mocks for call assertions
        self.MockVBox = mock_widgets['VBox']
        self.MockHBox = mock_widgets['HBox']
        self.MockGridBox = mock_widgets['GridBox']
        self.MockLayout = mock_widgets['Layout']
        self.MockWidget = mock_widgets['Widget']

    def tearDown(self):
        """Tear down patches."""
        # Exit context managers in reverse order of entry
        self.ipywidgets_patcher.__exit__(None, None, None)
        self.ui_patcher.__exit__(None, None, None)
        super().tearDown()
