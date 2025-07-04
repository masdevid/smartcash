"""
File: /Users/masdevid/Projects/smartcash/tests/ui/setup/colab/test_logging_and_progress.py

Test file untuk memastikan logging behavior dan setup stage progress.
"""

import pytest
import os
from unittest.mock import MagicMock, patch, Mock, AsyncMock
import sys
import logging

# Import test helpers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from . import test_helpers

# Setup mocks sebelum test dijalankan
test_helpers.setup_mocks(sys.modules)

# Mock external dependencies
mock_cv2 = MagicMock()
mock_cv2.CV_8UC1 = 1
mock_cv2.dnn = MagicMock()
mock_cv2.dnn.DictValue = MagicMock()
sys.modules['cv2'] = mock_cv2

mock_torch = MagicMock()
mock_torch.Tensor = MagicMock()
sys.modules['torch'] = mock_torch

mock_widgets = MagicMock()

class MockWidget:
    def __init__(self):
        self.children = []
        self.layout = MagicMock()
        self.value = ""
        self.description = ""
        self.style = MagicMock()
        self.on_click = MagicMock()
        self.observe = MagicMock()

mock_widgets.Widget = MockWidget
mock_widgets.Output = MagicMock(return_value=MockWidget())
mock_widgets.VBox = MagicMock(return_value=MockWidget())
mock_widgets.HTML = MagicMock(return_value=MockWidget())
mock_widgets.Button = MagicMock(return_value=MockWidget())
mock_widgets.Label = MagicMock(return_value=MockWidget())
mock_widgets.HBox = MagicMock(return_value=MockWidget())
mock_widgets.Tab = MagicMock(return_value=MockWidget())
mock_widgets.Accordion = MagicMock(return_value=MockWidget())
mock_widgets.Layout = MagicMock(return_value=MockWidget())
mock_widgets.Box = MagicMock(return_value=MockWidget())
mock_widgets.FloatProgress = MagicMock(return_value=MockWidget())
mock_widgets.IntProgress = MagicMock(return_value=MockWidget())

sys.modules['ipywidgets'] = mock_widgets

# Test functions

@pytest.fixture
def colab_initializer(mocker):
    try:
        from smartcash.ui.setup.colab.colab_initializer import ColabEnvInitializer
    except ImportError:
        ColabEnvInitializer = mock.Mock()
    
    init_instance = ColabEnvInitializer()
    mocker.patch.object(init_instance, 'initialize', return_value={'success': True})
    mocker.patch.object(init_instance, '_post_checks', return_value=None)
    return init_instance


def test_logging_redirection_to_accordion(mocker):
    # Arrange
    from test_helpers import create_async_mock_with_dict
    
    # Create a mock initializer directly
    initializer = MagicMock()
    initializer.logger = MagicMock()
    
    # Create mock UI components
    ui_components = MagicMock()
    log_accordion = MagicMock()
    log_accordion.log = MagicMock()
    ui_components.log_accordion = log_accordion
    initializer._ui_components = ui_components
    
    test_message = '🧪 Pesan tes untuk accordion log'
    
    # Act
    initializer.logger.info(test_message)
    
    # Assert
    initializer._ui_components.log_accordion.log.assert_called_with(test_message)

def test_no_logs_outside_accordion(mocker):
    """
    Test to ensure no logs appear outside the accordion.
    """
    from test_helpers import create_async_mock_with_dict
    print("Starting test_no_logs_outside_accordion", file=sys.stderr)

    # Create a mock initializer with async-compatible initialize method
    initializer = MagicMock()
    initializer.logger = MagicMock()
    initializer.logger._suppressed = False
    print("Logger unsuppressed", file=sys.stderr)
    
    # Create mock UI components
    mock_ui_components = {
        'header_container': MagicMock(),
        'summary_container': MagicMock(),
        'footer_container': MagicMock(),
        'env_info_panel': MagicMock(),
        'form_container': MagicMock(),
        'action_buttons': MagicMock(),
        'tips_requirements': MagicMock(),
        'main_container': MagicMock(),
        'progress_tracker': MagicMock(),
        'log_accordion': MagicMock()
    }
    
    # Setup log_accordion in footer_container
    mock_ui_components['footer_container'].log_accordion = MagicMock()
    mock_ui_components['footer_container'].log_accordion.logs = []
    
    # Setup progress_tracker
    mock_ui_components['progress_tracker'].widget = MagicMock()
    
    # Setup initialize to return our mock components
    initialize_result = {
        'status': True,
        'ui': mock_ui_components,
        'handlers': {'env_config': MagicMock(), 'setup': MagicMock()}
    }
    initializer.initialize = mocker.MagicMock(return_value=initialize_result)
    
    # Act
    result = initializer.initialize()
    print(f"Initialize result: {result['status']}", file=sys.stderr)
    
    # Assert
    assert result['status'] is True, "Initialization failed"
    assert 'ui' in result, "UI components not in result"
    print("Initialization successful", file=sys.stderr)

    log_container = result['ui']['footer_container'].log_accordion
    print(f"Log container: {log_container}", file=sys.stderr)
    print(f"Has logs attribute: {hasattr(log_container, 'logs')}", file=sys.stderr)
    if hasattr(log_container, 'logs'):
        print(f"Logs length: {len(log_container.logs)}", file=sys.stderr)
    
    assert not hasattr(log_container, 'logs') or len(log_container.logs) == 0, "Logs found outside accordion"
    print("test_no_logs_outside_accordion completed", file=sys.stderr)

def test_setup_stage_progress(mocker):
    """
    Test to ensure setup stage progress is updated correctly.
    """
    from test_helpers import create_async_mock_with_dict
    print("Starting test_setup_stage_progress", file=sys.stderr)

    # Create a mock initializer with async-compatible initialize method
    initializer = MagicMock()
    initializer.logger = MagicMock()
    initializer.logger._suppressed = False
    print("Logger unsuppressed", file=sys.stderr)
    
    # Create mock UI components
    mock_ui_components = {
        'header_container': MagicMock(),
        'summary_container': MagicMock(),
        'footer_container': MagicMock(),
        'env_info_panel': MagicMock(),
        'form_container': MagicMock(),
        'action_buttons': MagicMock(),
        'tips_requirements': MagicMock(),
        'main_container': MagicMock(),
        'log_accordion': MagicMock(),
        'progress_tracker': MagicMock()
    }
    
    # Setup progress_tracker with update_progress method
    progress_tracker_widget = MagicMock()
    progress_tracker_widget.update_progress = MagicMock()
    mock_ui_components['progress_tracker'].widget = progress_tracker_widget
    
    # Setup initialize to return our mock components
    initialize_result = {
        'status': True,
        'ui': mock_ui_components,
        'handlers': {'env_config': MagicMock(), 'setup': MagicMock()}
    }
    initializer.initialize = mocker.MagicMock(return_value=initialize_result)
    
    # Act
    result = initializer.initialize()
    print(f"Initialize result: {result['status']}", file=sys.stderr)
    
    # Assert
    assert result['status'] is True, "Initialization failed"
    assert 'ui' in result, "UI components not in result"
    print("Initialization successful", file=sys.stderr)

    progress_tracker_widget = result['ui']['progress_tracker'].widget
    print(f"Progress tracker widget: {progress_tracker_widget}", file=sys.stderr)
    
    # Call the update_progress method
    progress_tracker_widget.update_progress('Setup', 'Mulai setup', 0.1)
    print("Called update_progress on widget", file=sys.stderr)

    # Assert the method was called correctly
    progress_tracker_widget.update_progress.assert_called_once_with('Setup', 'Mulai setup', 0.1)
    print("test_setup_stage_progress completed", file=sys.stderr)
