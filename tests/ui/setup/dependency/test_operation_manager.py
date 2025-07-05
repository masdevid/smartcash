"""
File: tests/ui/setup/dependency/test_operation_manager.py
Tests for the OperationManager class.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, create_autospec
from typing import Dict, Any, Optional, List, Tuple, Callable

from smartcash.ui.setup.dependency.operations.operation_manager import (
    OperationManager, 
    OperationContext,
    OperationType
)
from smartcash.ui.setup.dependency.operations.install_operation import InstallOperationHandler
from smartcash.ui.setup.dependency.operations.base_operation import BaseOperationHandler
from typing import Dict, Any, List

# Test data
TEST_PACKAGES = ["numpy", "pandas"]
TEST_OPERATION_TYPE = OperationType.INSTALL

def create_test_context(
    operation_type: OperationType = TEST_OPERATION_TYPE,
    packages: list = None,
    status_callback: Callable[[str, str], None] = None,
    progress_callback: Callable[[int, int], None] = None
) -> OperationContext:
    """Helper to create a test operation context."""
    return OperationContext(
        operation_type=operation_type,
        packages=packages or TEST_PACKAGES,
        requires_confirmation=operation_type in [OperationType.UNINSTALL, OperationType.UPDATE],
        status_callback=status_callback or (lambda msg, level="info": None),
        progress_callback=progress_callback or (lambda current, total: None)
    )

@pytest.fixture
def mock_ui_components() -> Dict[str, Any]:
    """Fixture providing mock UI components."""
    return {
        'status_label': MagicMock(),
        'progress_bar': MagicMock(),
        'log_output': MagicMock()
    }

@pytest.fixture
def operation_manager():
    """Create an OperationManager instance for testing."""
    manager = OperationManager(ui_components={})
    manager._operation_handlers = {
        OperationType.INSTALL: MagicMock(),
        OperationType.UPDATE: MagicMock(),
        OperationType.UNINSTALL: MagicMock(),
        OperationType.CHECK_STATUS: MagicMock()
    }
    return manager

@pytest.fixture
def mock_operation_handler():
    """Create a mock operation handler for testing."""
    # Create a subclass of InstallOperationHandler for testing
    class TestOperationHandler(InstallOperationHandler):
        def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
            super().__init__(ui_components, config)
            self._execute_operation = AsyncMock()
            self._execute_operation.return_value = {
                'success': True,
                'message': 'Success',
                'installed': 2,
                'total': 2,
                'duration': 1.5,
                'results': [
                    {'package': 'numpy', 'success': True, 'message': 'Installed'},
                    {'package': 'pandas', 'success': True, 'message': 'Installed'}
                ]
            }
        
        async def get_operations(self) -> List[Dict[str, Any]]:
            return [
                {'operation': 'install', 'package': 'numpy'},
                {'operation': 'install', 'package': 'pandas'}
            ]
        
        async def initialize(self) -> None:
            pass
            
        async def execute_operation(self, *args, **kwargs) -> Dict[str, Any]:
            return await self._execute_operation(*args, **kwargs)
    
    # Create a mock that will be patched onto the handler class
    mock_handler = MagicMock(spec=TestOperationHandler)
    
    # Configure the mock to return our test handler when instantiated
    def create_mock_handler(ui_components, config):
        handler = TestOperationHandler(ui_components, config)
        mock_handler.ui_components = ui_components
        mock_handler.config = config
        return handler
    
    mock_handler.side_effect = create_mock_handler
    
    return mock_handler

@pytest.mark.asyncio
async def test_execute_operation_success(operation_manager):
    """Test successful operation execution."""
    # Setup
    mock_handler = MagicMock()
    mock_handler.execute_operation.return_value = {
        'success': True,
        'message': 'Success',
        'installed': 2,
        'total': 2,
        'duration': 1.5,
        'results': []
    }
    
    # Set the mock handler for INSTALL operation
    operation_manager._operation_handlers[OperationType.INSTALL] = MagicMock(return_value=mock_handler)
    
    # Create a test context
    context = OperationContext(
        operation_type=OperationType.INSTALL,
        packages=['numpy', 'pandas'],
        requires_confirmation=False,
        status_callback=MagicMock(),
        progress_callback=MagicMock()
    )
    
    # Execute
    result = await operation_manager.execute_operation(context)
    
    # Verify
    assert result['success'] is True
    assert result['message'] == 'Success'
    assert result['installed'] == 2
    assert result['total'] == 2
    
    # Verify the handler was called with correct arguments
    operation_manager._operation_handlers[OperationType.INSTALL].assert_called_once_with(
        ui_components=operation_manager.ui_components,
        config={}
    )
    # Verify execute_operation was called
    mock_handler.execute_operation.assert_called_once()

@pytest.mark.asyncio
async def test_execute_operation_no_packages(operation_manager):
    """Test operation execution with no packages."""
    # Create a context with no packages
    context = OperationContext(
        operation_type=OperationType.INSTALL,
        packages=[],
        requires_confirmation=False,
        status_callback=MagicMock(),
        progress_callback=MagicMock()
    )
    
    # Execute
    result = await operation_manager.execute_operation(context)
    
    # Verify
    assert result['success'] is False
    assert "No packages provided" in result['message']
    
    # Verify handler was not called
    operation_manager._operation_handlers[OperationType.INSTALL].assert_not_called()

@pytest.mark.asyncio
async def test_execute_operation_handler_error(operation_manager):
    """Test operation execution when handler raises an exception."""
    # Setup mock handler that raises an exception
    mock_handler = MagicMock()
    mock_handler.execute_operation.side_effect = Exception("Test error")
    
    # Set the mock handler for INSTALL operation
    operation_manager._operation_handlers[OperationType.INSTALL] = MagicMock(return_value=mock_handler)
    
    # Create a test context
    context = OperationContext(
        operation_type=OperationType.INSTALL,
        packages=['numpy'],
        requires_confirmation=False,
        status_callback=MagicMock(),
        progress_callback=MagicMock()
    )
    
    # Execute
    result = await operation_manager.execute_operation(context)
    
    # Verify
    assert result['success'] is False
    assert "Test error" in result['message']
    
    # Verify the handler was called
    operation_manager._operation_handlers[OperationType.INSTALL].assert_called_once()
    mock_handler.execute_operation.assert_called_once()

@pytest.mark.asyncio
async def test_execute_operation_concurrent_check(operation_manager):
    """Test that concurrent operations are prevented."""
    # Create an event to control when the operation completes
    operation_done = asyncio.Event()
    
    # Create a mock handler with delayed execution
    async def delayed_execute(*args, **kwargs):
        await operation_done.wait()
        return {'success': True, 'message': 'Completed'}
    
    mock_handler = MagicMock()
    mock_handler.execute_operation.side_effect = delayed_execute
    
    # Set the mock handler for INSTALL operation
    operation_manager._operation_handlers[OperationType.INSTALL] = MagicMock(return_value=mock_handler)
    
    # Create a test context
    context = OperationContext(
        operation_type=OperationType.INSTALL,
        packages=['numpy'],
        requires_confirmation=False,
        status_callback=MagicMock(),
        progress_callback=MagicMock()
    )
    
    # Start first operation (it will block on the event)
    task = asyncio.create_task(operation_manager.execute_operation(context))
    
    # Give the task a moment to start
    await asyncio.sleep(0.1)
    
    # Try to start second operation while first is still running
    result = await operation_manager.execute_operation(context)
    assert result['success'] is False
    assert "already in progress" in result['message'].lower()
    
    # Complete the first operation
    operation_done.set()
    await task

def test_operation_handlers_initialized(operation_manager):
    """Test that operation handlers are properly initialized."""
    # Test that all operation types have a handler
    for op_type in OperationType:
        assert op_type in operation_manager._operation_handlers
        handler_class = operation_manager._operation_handlers[op_type]
        assert handler_class is not None
        assert issubclass(handler_class, BaseOperationHandler)

@pytest.mark.asyncio
async def test_update_operation_status(operation_manager, mock_ui_components):
    """Test updating operation status updates the UI components."""
    test_message = "Test message"
    test_level = "warning"
    
    # Test with status callback
    mock_callback = MagicMock()
    context = create_test_context(status_callback=mock_callback)
    
    # Set the current operation
    operation_manager._current_operation = context
    
    # Call the internal method directly
    operation_manager._update_status(test_message, test_level)
    
    # Verify the callback was called
    mock_callback.assert_called_once_with(test_message, test_level)
    
    # Test with UI components when no callback
    mock_ui_components['status_label'] = MagicMock()
    operation_manager._current_operation = create_test_context(status_callback=None)
    operation_manager._update_status(test_message, test_level)
    mock_ui_components['status_label'].value = test_message

def test_get_current_operation(operation_manager):
    """Test getting the current operation context."""
    assert operation_manager.get_current_operation() is None
    
    context = create_test_context()
    operation_manager._current_operation = context
    assert operation_manager.get_current_operation() is context
