#!/usr/bin/env python3
"""
Test runner for dependency module tests with enhanced debugging
"""
import sys
import os
import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
import types
import importlib

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
src_root = project_root / 'smartcash'

# Add both project root and src root to path
for path in [str(project_root), str(src_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)
        logger.debug(f"Added to path: {path}")

# Log current Python path for debugging
logger.debug("Python path:")
for p in sys.path:
    logger.debug(f"  - {p}")

def log_module_creation(name):
    """Log when a module is created"""
    logger.debug(f"Creating mock module: {name}")

def create_mock_module(name, parent=None):
    """Create a mock module with the given name and optional parent"""
    log_module_creation(name)
    parts = name.split('.')
    module = types.ModuleType(name)
    module.__file__ = f"<mock_module>/{'/'.join(parts)}.py"
    module.__package__ = '.'.join(parts[:-1]) if len(parts) > 1 else parts[0]
    
    # Set __path__ for packages
    if len(parts) > 1:
        module.__path__ = [f"<mock_path>/{'/'.join(parts)}"]
    
    sys.modules[name] = module
    
    # Set up parent references
    if parent:
        parent_name = parent.__name__ if hasattr(parent, '__name__') else str(parent)
        parent_parts = parent_name.split('.')
        current = sys.modules
        
        # Traverse the parent path to set the attribute
        for part in parent_parts:
            if part not in current:
                logger.warning(f"Parent module {part} not found when creating {name}")
                break
            current = current[part].__dict__
        else:
            current[parts[-1]] = module
    
    return module

def setup_mock_environment():
    """Setup the complete mock environment with all required modules"""
    logger.info("Setting up mock environment...")
    
    # Import test helpers first
    from tests.unit.ui.setup.dependency.test_helpers import (
        MockConfigManager, MockModuleUIHandler, MockBaseUIHandler,
        mock_handle_errors, create_mock_package, setup_module_mock,
        get_default_dependency_config
    )
    
    # Clear any existing mock modules to avoid conflicts
    for module in list(sys.modules.keys()):
        if module.startswith('smartcash'):
            del sys.modules[module]
    
    # Create base packages in the correct order
    logger.debug("Creating base packages...")
    
    # Create each level of the package hierarchy explicitly
    smartcash = types.ModuleType('smartcash')
    sys.modules['smartcash'] = smartcash
    
    smartcash.ui = types.ModuleType('smartcash.ui')
    sys.modules['smartcash.ui'] = smartcash.ui
    
    smartcash.ui.core = types.ModuleType('smartcash.ui.core')
    sys.modules['smartcash.ui.core'] = smartcash.ui.core
    
    smartcash.ui.core.handlers = types.ModuleType('smartcash.ui.core.handlers')
    sys.modules['smartcash.ui.core.handlers'] = smartcash.ui.core.handlers
    
    smartcash.ui.core.initializers = types.ModuleType('smartcash.ui.core.initializers')
    sys.modules['smartcash.ui.core.initializers'] = smartcash.ui.core.initializers
    
    smartcash.ui.core.shared = types.ModuleType('smartcash.ui.core.shared')
    sys.modules['smartcash.ui.core.shared'] = smartcash.ui.core.shared
    
    smartcash.ui.setup = types.ModuleType('smartcash.ui.setup')
    sys.modules['smartcash.ui.setup'] = smartcash.ui.setup
    
    smartcash.ui.setup.dependency = types.ModuleType('smartcash.ui.setup.dependency')
    sys.modules['smartcash.ui.setup.dependency'] = smartcash.ui.setup.dependency
    
    # Set up __path__ for packages
    for module in [smartcash, smartcash.ui, smartcash.ui.core, smartcash.ui.setup, smartcash.ui.setup.dependency]:
        module.__path__ = [f"<mock_path>/{module.__name__.replace('.', '/')}"]
        module.__file__ = f"<mock_file>/{module.__name__.replace('.', '/')}/__init__.py"
        module.__package__ = module.__name__
    
    # Setup core package structure with proper __path__
    logger.debug("Setting up core package structure...")
    
    # Create all required subpackages explicitly
    packages_to_create = [
        'smartcash.ui.core.handlers',
        'smartcash.ui.core.initializers',
        'smartcash.ui.core.shared',
        'smartcash.ui.setup.dependency.handlers',
        'smartcash.ui.setup.dependency.configs',
        'smartcash.ui.setup.dependency.components',
        'smartcash.ui.setup.dependency.operations',
        'smartcash.ui.setup.dependency.operations.factory',
        'smartcash.ui.setup.dependency.operations.operation_manager',
    ]
    
    for pkg in packages_to_create:
        parts = pkg.split('.')
        for i in range(1, len(parts) + 1):
            current = '.'.join(parts[:i])
            if current not in sys.modules:
                module = types.ModuleType(current)
                module.__path__ = [f"<mock_path>/{current.replace('.', '/')}"]
                module.__file__ = f"<mock_file>/{current.replace('.', '/')}/__init__.py"
                module.__package__ = current
                sys.modules[current] = module
                
                # Add to parent's __dict__
                if i > 1:
                    parent = sys.modules['.'.join(parts[:i-1])]
                    setattr(parent, parts[i-1], module)
    
    # Setup shared config manager
    logger.debug("Setting up shared config manager...")
    shared_config_manager = types.ModuleType('smartcash.ui.core.shared.shared_config_manager')
    shared_config_manager.ConfigManager = MockConfigManager
    sys.modules['smartcash.ui.core.shared.shared_config_manager'] = shared_config_manager
    
    # Setup core handlers
    logger.debug("Setting up core handlers...")
    ui_handler = types.ModuleType('smartcash.ui.core.handlers.ui_handler')
    ui_handler.handle_errors = mock_handle_errors
    ui_handler.ModuleUIHandler = MockModuleUIHandler
    ui_handler.BaseUIHandler = MockBaseUIHandler
    sys.modules['smartcash.ui.core.handlers.ui_handler'] = ui_handler
    
    # Setup module initializer
    logger.debug("Setting up module initializer...")
    module_initializer = types.ModuleType('smartcash.ui.core.initializers.module_initializer')
    module_initializer.ModuleInitializer = type('MockModuleInitializer', (), {
        '__init__': lambda self, *args, **kwargs: None,
        'initialize': lambda self, *args, **kwargs: None,
        'cleanup': lambda self, *args, **kwargs: None,
    })
    sys.modules['smartcash.ui.core.initializers.module_initializer'] = module_initializer
    
    # Setup dependency_defaults
    logger.debug("Setting up dependency defaults...")
    dependency_defaults = types.ModuleType('smartcash.ui.setup.dependency.configs.dependency_defaults')
    dependency_defaults.get_default_dependency_config = get_default_dependency_config
    sys.modules['smartcash.ui.setup.dependency.configs.dependency_defaults'] = dependency_defaults
    
    # Setup operation type enum
    from enum import Enum
    class MockOperationType(Enum):
        INSTALL = 'install'
        UNINSTALL = 'uninstall'
        UPDATE = 'update'
    
    # Setup operation manager
    operation_manager = types.ModuleType('smartcash.ui.setup.dependency.operations.operation_manager')
    operation_manager.OperationType = MockOperationType
    sys.modules['smartcash.ui.setup.dependency.operations.operation_manager'] = operation_manager
    
    # Setup operation factory
    operation_factory = types.ModuleType('smartcash.ui.setup.dependency.operations.factory')
    operation_factory.OperationHandlerFactory = MagicMock()
    sys.modules['smartcash.ui.setup.dependency.operations.factory'] = operation_factory
    
    # Setup components
    components = types.ModuleType('smartcash.ui.setup.dependency.components.dependency_ui')
    components.create_dependency_ui_components = MagicMock(return_value={})
    sys.modules['smartcash.ui.setup.dependency.components.dependency_ui'] = components
    
    # Setup handlers
    handlers = types.ModuleType('smartcash.ui.setup.dependency.handlers.dependency_ui_handler')
    handlers.DependencyUIHandler = MagicMock()
    sys.modules['smartcash.ui.setup.dependency.handlers.dependency_ui_handler'] = handlers
    
    # Initialize the dependency module
    dependency_module = sys.modules['smartcash.ui.setup.dependency']
    dependency_module.__all__ = ['initialize_dependency_ui']
    dependency_module.initialize_dependency_ui = MagicMock()
    
    # Add __init__.py to all packages to make them proper packages
    for module in sys.modules.values():
        if hasattr(module, '__path__') and not hasattr(module, '__file__'):
            module.__file__ = module.__name__.replace('.', '/') + '/__init__.py'
    
    # Log current sys.modules for debugging
    logger.debug("Current sys.modules entries:")
    for mod in sorted(k for k in sys.modules if k.startswith('smartcash')):
        logger.debug(f"  - {mod}")
    
    # Try to import the module we want to test
    logger.info("Attempting to import dependency_initializer...")
    try:
        # First try direct import
        try:
            from smartcash.ui.setup.dependency import dependency_initializer
            logger.info(f"Successfully imported module: {dependency_initializer}")
            return True
        except ImportError as e:
            logger.warning(f"Direct import failed: {e}")
            
        # If direct import fails, try importing the module file directly
        module_path = project_root / 'smartcash' / 'ui' / 'setup' / 'dependency' / 'dependency_initializer.py'
        logger.info(f"Trying to import from: {module_path}")
        
        if not module_path.exists():
            raise ImportError(f"Module file not found at {module_path}")
            
        # Use importlib to load the module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'smartcash.ui.setup.dependency.dependency_initializer',
            str(module_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec from {module_path}")
            
        module = importlib.util.module_from_spec(spec)
        sys.modules['smartcash.ui.setup.dependency.dependency_initializer'] = module
        spec.loader.exec_module(module)
        
        logger.info(f"Successfully loaded module from file: {module}")
        return True
        
    except Exception as e:
        logger.error(f"Error importing module: {e}", exc_info=True)
        # Log detailed debug info
        logger.error("\n=== Debug Information ===")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Module search paths:")
        for i, path in enumerate(sys.path):
            logger.error(f"  {i}: {path}")
        logger.error("\nContents of smartcash.ui.setup.dependency:")
        try:
            import pkgutil
            module = sys.modules.get('smartcash.ui.setup.dependency')
            if module and hasattr(module, '__path__'):
                for item in pkgutil.iter_modules(module.__path__):
                    logger.error(f"  - {item.name}")
        except Exception as debug_e:
            logger.error(f"Error getting module contents: {debug_e}")
        raise

def run_tests():
    """Run the tests with enhanced error reporting"""
    logger.info("Starting test run...")
    
    try:
        # Setup mock environment
        if not setup_mock_environment():
            logger.error("Failed to set up mock environment")
            return 1
            
        # Import test module after setting up mocks
        logger.info("Importing test module...")
        from tests.unit.ui.setup.dependency import test_dependency_initializer
        
        # Run tests with coverage
        logger.info("Running tests...")
        return pytest.main([
            '-v',
            '--tb=short',
            '--cov=smartcash.ui.setup.dependency.dependency_initializer',
            '--cov-report=term-missing',
            '--log-cli-level=DEBUG',
            str(Path(__file__).parent / 'test_dependency_initializer.py')
        ])
    except Exception as e:
        logger.exception("Unexpected error during test execution:")
        return 1
    finally:
        # Clean up mock modules
        logger.info("Cleaning up mock modules...")
        for module in list(sys.modules.keys()):
            if module.startswith('smartcash'):
                del sys.modules[module]

if __name__ == '__main__':
    sys.exit(run_tests())
