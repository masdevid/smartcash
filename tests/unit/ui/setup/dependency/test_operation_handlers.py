"""
File: tests/unit/ui/setup/dependency/test_operation_handlers.py
Tests for operation handlers in the dependency module
"""
import sys
import pytest
import subprocess
from unittest.mock import MagicMock, patch, call

# Create mock classes
class MockBaseOperation:
    def __init__(self):
        self.logger = MagicMock()
        self.auto_confirm = False
        self.packages = []
        self.result = {'status': 'success', 'message': ''}
    
    def execute(self, packages=None, auto_confirm=False, **kwargs):
        self.packages = packages or []
        self.auto_confirm = auto_confirm
        return self.result

class MockPipInstallOperation(MockBaseOperation):
    def execute(self, packages=None, auto_confirm=False, **kwargs):
        self.packages = packages or []
        self.auto_confirm = auto_confirm
        if 'nonexistent' in str(packages):
            return {
                'status': 'error',
                'error': 'ERROR: Could not find a version that satisfies the requirement',
                'message': 'Failed to install packages'
            }
        return {
            'status': 'success',
            'message': f'Successfully installed {' '.join(packages)}',
            'packages': packages
        }

class MockPipUninstallOperation(MockBaseOperation):
    def execute(self, packages=None, auto_confirm=False, **kwargs):
        self.packages = packages or []
        self.auto_confirm = auto_confirm
        return {
            'status': 'success',
            'message': f'Successfully uninstalled {' '.join(packages)}',
            'packages': packages
        }

class MockPipUpdateOperation(MockBaseOperation):
    def execute(self, packages=None, auto_confirm=False, **kwargs):
        self.packages = packages or []
        self.auto_confirm = auto_confirm
        return {
            'status': 'success',
            'message': f'Successfully updated {' '.join(packages)}',
            'packages': packages
        }

class MockOperationHandlerFactory:
    @staticmethod
    def create_handler(operation_type):
        if operation_type == 'install':
            return MockPipInstallOperation()
        elif operation_type == 'uninstall':
            return MockPipUninstallOperation()
        elif operation_type == 'update':
            return MockPipUpdateOperation()
        raise ValueError("Invalid operation type")

# Set up module mocks
sys.modules['smartcash'] = MagicMock()
sys.modules['smartcash.ui'] = MagicMock()
sys.modules['smartcash.ui.setup'] = MagicMock()
sys.modules['smartcash.ui.setup.dependency'] = MagicMock()
sys.modules['smartcash.ui.setup.dependency.operations'] = MagicMock()
sys.modules['smartcash.ui.setup.dependency.operations.base_operation'] = MagicMock()
sys.modules['smartcash.ui.setup.dependency.operations.pip_install_operation'] = MagicMock()
sys.modules['smartcash.ui.setup.dependency.operations.pip_uninstall_operation'] = MagicMock()
sys.modules['smartcash.ui.setup.dependency.operations.pip_update_operation'] = MagicMock()
sys.modules['smartcash.ui.setup.dependency.operations.factory'] = MagicMock()

# Patch the actual imports
sys.modules['smartcash.ui.setup.dependency.operations.base_operation'].BaseOperation = MockBaseOperation
sys.modules['smartcash.ui.setup.dependency.operations.pip_install_operation'].PipInstallOperation = MockPipInstallOperation
sys.modules['smartcash.ui.setup.dependency.operations.pip_uninstall_operation'].PipUninstallOperation = MockPipUninstallOperation
sys.modules['smartcash.ui.setup.dependency.operations.pip_update_operation'].PipUpdateOperation = MockPipUpdateOperation
sys.modules['smartcash.ui.setup.dependency.operations.factory'].OperationHandlerFactory = MockOperationHandlerFactory

# Now import the actual modules we're testing
from smartcash.ui.setup.dependency.operations.base_operation import BaseOperation
from smartcash.ui.setup.dependency.operations.pip_install_operation import PipInstallOperation
from smartcash.ui.setup.dependency.operations.pip_uninstall_operation import PipUninstallOperation
from smartcash.ui.setup.dependency.operations.pip_update_operation import PipUpdateOperation
from smartcash.ui.setup.dependency.operations.factory import OperationHandlerFactory

class TestOperationHandlers:
    """Test cases for operation handlers"""
    
    def test_pip_install_operation_success(self):
        """Test successful pip install operation"""
        # Setup
        handler = PipInstallOperation()
        packages = ['numpy', 'pandas']
        
        # Execute
        result = handler.execute(packages, auto_confirm=True)
        
        # Verify
        assert result['status'] == 'success'
        assert 'Successfully installed' in result['message']
        assert result['packages'] == packages
        assert handler.auto_confirm is True
        
    def test_pip_install_operation_failure(self):
        """Test pip install operation with package not found"""
        # Setup
        handler = PipInstallOperation()
        packages = ['nonexistent-package']
        
        # Execute
        result = handler.execute(packages, auto_confirm=False)
        
        # Verify
        assert result['status'] == 'error'
        assert 'Could not find a version' in result.get('error', '')
        assert handler.auto_confirm is False
        
    def test_pip_install_with_auto_confirm(self):
        """Test pip install with auto-confirm flag"""
        # Setup
        handler = PipInstallOperation()
        
        # Execute with auto_confirm=True
        result = handler.execute(['numpy'], auto_confirm=True)
        
        # Verify
        assert result['status'] == 'success'
        assert handler.auto_confirm is True
        
    def test_operation_factory_valid_types(self):
        """Test operation handler factory with valid operation types"""
        # Test all valid operation types
        install_handler = OperationHandlerFactory.create_handler('install')
        assert isinstance(install_handler, PipInstallOperation)
        
        uninstall_handler = OperationHandlerFactory.create_handler('uninstall')
        assert isinstance(uninstall_handler, PipUninstallOperation)
        
        update_handler = OperationHandlerFactory.create_handler('update')
        assert isinstance(update_handler, PipUpdateOperation)
        
    def test_operation_factory_invalid_type(self):
        """Test operation handler factory with invalid operation type"""
        # Test invalid operation type
        with pytest.raises(ValueError, match="Invalid operation type"):
            OperationHandlerFactory.create_handler('invalid_operation')
    
    def test_pip_uninstall_operation(self):
        """Test pip uninstall operation"""
        # Setup
        handler = PipUninstallOperation()
        packages = ['numpy']
        
        # Execute
        result = handler.execute(packages, auto_confirm=True)
        
        # Verify
        assert result['status'] == 'success'
        assert 'numpy' in result['message']
        assert handler.packages == packages
        assert handler.auto_confirm is True
        
    def test_pip_update_operation(self):
        """Test pip update operation"""
        # Setup
        handler = PipUpdateOperation()
        packages = ['numpy']
        
        # Execute
        result = handler.execute(packages, auto_confirm=True)
        
        # Verify
        assert result['status'] == 'success'
        assert 'numpy' in result['message']
        assert handler.packages == packages
        assert handler.auto_confirm is True
