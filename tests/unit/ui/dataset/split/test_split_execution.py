"""
Execution tests for the dataset split module.

This module tests the actual execution and instantiation of split module
components to ensure they work correctly in a real environment.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TestSplitModuleExecution:
    """Test actual execution of split module components."""
    
    def test_import_split_initializer(self):
        """Test importing the split initializer module."""
        try:
            from smartcash.ui.dataset.split.split_initializer import SplitInitializer
            assert SplitInitializer is not None
        except ImportError as e:
            pytest.fail(f"Failed to import SplitInitializer: {e}")
    
    def test_import_config_handler(self):
        """Test importing the config handler module."""
        try:
            from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler
            assert SplitConfigHandler is not None
        except ImportError as e:
            pytest.fail(f"Failed to import SplitConfigHandler: {e}")
    
    def test_import_default_config(self):
        """Test importing the default configuration."""
        try:
            from smartcash.ui.dataset.split.configs.split_defaults import DEFAULT_SPLIT_CONFIG
            assert DEFAULT_SPLIT_CONFIG is not None
            assert isinstance(DEFAULT_SPLIT_CONFIG, dict)
        except ImportError as e:
            pytest.fail(f"Failed to import DEFAULT_SPLIT_CONFIG: {e}")
    
    def test_import_ui_components(self):
        """Test importing UI component functions."""
        try:
            from smartcash.ui.dataset.split.components.split_ui import create_split_ui_components
            assert create_split_ui_components is not None
        except ImportError as e:
            pytest.fail(f"Failed to import create_split_ui_components: {e}")
    
    def test_instantiate_config_handler(self):
        """Test creating a config handler instance."""
        try:
            from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler
            
            # Test default instantiation
            handler = SplitConfigHandler()
            assert handler is not None
            assert hasattr(handler, 'config')
            assert hasattr(handler, 'validate_config')
            
        except Exception as e:
            pytest.fail(f"Failed to instantiate SplitConfigHandler: {e}")
    
    def test_instantiate_split_initializer(self):
        """Test creating a split initializer instance."""
        try:
            from smartcash.ui.dataset.split.split_initializer import SplitInitializer
            
            # Test default instantiation
            initializer = SplitInitializer()
            assert initializer is not None
            assert hasattr(initializer, 'config')
            assert hasattr(initializer, 'config_handler')
            assert hasattr(initializer, 'components')
            
        except Exception as e:
            pytest.fail(f"Failed to instantiate SplitInitializer: {e}")
    
    def test_config_handler_functionality(self):
        """Test basic config handler functionality."""
        try:
            from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler
            from smartcash.ui.dataset.split.configs.split_defaults import DEFAULT_SPLIT_CONFIG
            
            handler = SplitConfigHandler()
            
            # Test validation with default config
            is_valid = handler.validate_config(DEFAULT_SPLIT_CONFIG)
            assert is_valid is True
            
            # Test config property
            config = handler.config
            assert isinstance(config, dict)
            assert 'data' in config
            assert 'output' in config
            
        except Exception as e:
            pytest.fail(f"Config handler functionality test failed: {e}")
    
    def test_split_initializer_initialization(self):
        """Test split initializer initialization process."""
        try:
            from smartcash.ui.dataset.split.split_initializer import SplitInitializer
            from smartcash.ui.dataset.split.configs.split_defaults import DEFAULT_SPLIT_CONFIG
            
            # Test with default config
            initializer = SplitInitializer()
            
            # Test _initialize_impl method exists and is callable
            assert hasattr(initializer, '_initialize_impl')
            assert callable(initializer._initialize_impl)
            
            # Test with custom config
            custom_config = DEFAULT_SPLIT_CONFIG.copy()
            custom_config['data']['seed'] = 999
            
            initializer_custom = SplitInitializer(config=custom_config)
            assert initializer_custom.config == custom_config
            
        except Exception as e:
            pytest.fail(f"Split initializer initialization test failed: {e}")
    
    def test_entry_point_functions(self):
        """Test module entry point functions."""
        try:
            from smartcash.ui.dataset.split.split_initializer import (
                create_split_config_cell,
                get_split_config_components,
                init_split_ui,
                get_split_initializer
            )
            
            # Test that functions exist and are callable
            assert callable(create_split_config_cell)
            assert callable(get_split_config_components)
            assert callable(init_split_ui)
            assert callable(get_split_initializer)
            
            # Test get_split_initializer function
            initializer = get_split_initializer()
            assert initializer is not None
            
        except Exception as e:
            pytest.fail(f"Entry point functions test failed: {e}")
    
    def test_config_validation_execution(self):
        """Test actual config validation execution."""
        try:
            from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler
            
            handler = SplitConfigHandler()
            
            # Test valid config
            valid_config = {
                'data': {
                    'split_ratios': {
                        'train': 0.7,
                        'val': 0.15,
                        'test': 0.15
                    },
                    'seed': 42,
                    'shuffle': True,
                    'stratify': False
                },
                'output': {
                    'train_dir': 'data/train',
                    'val_dir': 'data/val',
                    'test_dir': 'data/test',
                    'create_subdirs': True,
                    'overwrite': False,
                    'relative_paths': True,
                    'preserve_dir_structure': True,
                    'use_symlinks': False,
                    'backup': True,
                    'backup_dir': 'data/backup'
                }
            }
            
            result = handler.validate_config(valid_config)
            assert result is True
            
            # Test invalid config (should raise exception or return False)
            invalid_config = valid_config.copy()
            invalid_config['data']['split_ratios']['train'] = 2.0  # Invalid ratio
            
            try:
                result = handler.validate_config(invalid_config)
                assert result is False
            except ValueError:
                # Expected for invalid ratios
                pass
            
        except Exception as e:
            pytest.fail(f"Config validation execution test failed: {e}")
    
    def test_config_update_execution(self):
        """Test actual config update execution."""
        try:
            from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler
            from smartcash.ui.dataset.split.configs.split_defaults import DEFAULT_SPLIT_CONFIG
            
            handler = SplitConfigHandler()
            handler.load_config(DEFAULT_SPLIT_CONFIG)
            
            # Test simple update
            updates = {'data.seed': 999}
            handler.update_config(updates)
            
            assert handler.config['data']['seed'] == 999
            
            # Test nested update
            updates = {'data.split_ratios.train': 0.8}
            handler.update_config(updates)
            
            assert handler.config['data']['split_ratios']['train'] == 0.8
            
        except Exception as e:
            pytest.fail(f"Config update execution test failed: {e}")
    
    def test_module_structure_compliance(self):
        """Test that the module structure follows expected patterns."""
        try:
            # Test that all expected modules exist
            from smartcash.ui.dataset.split import split_initializer
            from smartcash.ui.dataset.split.handlers import config_handler
            from smartcash.ui.dataset.split.configs import split_defaults
            from smartcash.ui.dataset.split.components import split_ui
            
            # Test that main classes exist
            from smartcash.ui.dataset.split.split_initializer import SplitInitializer
            from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler
            
            # Test that configuration exists
            from smartcash.ui.dataset.split.configs.split_defaults import DEFAULT_SPLIT_CONFIG, VALIDATION_RULES
            
            # Test that UI function exists
            from smartcash.ui.dataset.split.components.split_ui import create_split_ui_components
            
            assert True  # If we get here, all imports succeeded
            
        except ImportError as e:
            pytest.fail(f"Module structure compliance test failed: {e}")
    
    def test_default_config_validity(self):
        """Test that the default configuration is valid."""
        try:
            from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler
            from smartcash.ui.dataset.split.configs.split_defaults import DEFAULT_SPLIT_CONFIG
            
            handler = SplitConfigHandler()
            
            # The default config should always be valid
            is_valid = handler.validate_config(DEFAULT_SPLIT_CONFIG)
            assert is_valid is True
            
            # Check that ratios sum to 1.0
            ratios = DEFAULT_SPLIT_CONFIG['data']['split_ratios']
            ratios_sum = ratios['train'] + ratios['val'] + ratios['test']
            assert 0.999 <= ratios_sum <= 1.001
            
        except Exception as e:
            pytest.fail(f"Default config validity test failed: {e}")
    
    def test_error_handling_execution(self):
        """Test actual error handling execution."""
        try:
            from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler
            
            handler = SplitConfigHandler()
            
            # Test with invalid config type
            try:
                handler.load_config("not_a_dict")
                pytest.fail("Should have raised ValueError for invalid config type")
            except ValueError as e:
                assert "Configuration must be a dictionary" in str(e)
            
            # Test with invalid ratios
            invalid_config = {
                'data': {
                    'split_ratios': {
                        'train': 0.5,
                        'val': 0.3,
                        'test': 0.3  # Sum = 1.1
                    },
                    'seed': 42,
                    'shuffle': True,
                    'stratify': False
                },
                'output': {
                    'train_dir': 'data/train',
                    'val_dir': 'data/val',
                    'test_dir': 'data/test',
                    'create_subdirs': True,
                    'overwrite': False,
                    'relative_paths': True,
                    'preserve_dir_structure': True,
                    'use_symlinks': False,
                    'backup': True,
                    'backup_dir': 'data/backup'
                }
            }
            
            try:
                handler.load_config(invalid_config)
                pytest.fail("Should have raised ValueError for invalid ratios")
            except ValueError as e:
                assert "Split ratios must sum to 1.0" in str(e)
                
        except Exception as e:
            pytest.fail(f"Error handling execution test failed: {e}")


class TestSplitModuleRealExecution:
    """Test real execution scenarios without mocking."""
    
    def test_full_workflow_execution(self):
        """Test a complete workflow execution."""
        try:
            from smartcash.ui.dataset.split.split_initializer import SplitInitializer
            from smartcash.ui.dataset.split.configs.split_defaults import DEFAULT_SPLIT_CONFIG
            
            # Create initializer
            initializer = SplitInitializer(config=DEFAULT_SPLIT_CONFIG)
            
            # Verify it was created successfully
            assert initializer is not None
            assert initializer.config == DEFAULT_SPLIT_CONFIG
            assert hasattr(initializer, 'config_handler')
            
            # Test that config handler works
            assert initializer.config_handler.config == DEFAULT_SPLIT_CONFIG
            
            # Test validation
            assert initializer.config_handler.validate_config(DEFAULT_SPLIT_CONFIG) is True
            
        except Exception as e:
            pytest.fail(f"Full workflow execution test failed: {e}")
    
    def test_entry_point_execution(self):
        """Test entry point function execution."""
        try:
            from smartcash.ui.dataset.split.split_initializer import get_split_initializer
            
            # Test getting initializer
            initializer = get_split_initializer()
            assert initializer is not None
            
            # Test with config
            config = {'data': {'seed': 123}}
            initializer_with_config = get_split_initializer(config=config)
            assert initializer_with_config is not None
            assert initializer_with_config.config == config
            
        except Exception as e:
            pytest.fail(f"Entry point execution test failed: {e}")
    
    def test_component_section_imports(self):
        """Test that component sections can be imported."""
        try:
            from smartcash.ui.dataset.split.components.ratio_section import create_ratio_section
            from smartcash.ui.dataset.split.components.path_section import create_path_section
            from smartcash.ui.dataset.split.components.advanced_section import create_advanced_section
            
            assert callable(create_ratio_section)
            assert callable(create_path_section) 
            assert callable(create_advanced_section)
            
        except ImportError as e:
            pytest.fail(f"Component section imports failed: {e}")
    
    def test_ratio_section_execution(self):
        """Test ratio section component execution."""
        try:
            from smartcash.ui.dataset.split.components.ratio_section import create_ratio_section
            from smartcash.ui.dataset.split.configs.split_defaults import DEFAULT_SPLIT_CONFIG
            
            # Create ratio section
            result = create_ratio_section(DEFAULT_SPLIT_CONFIG)
            
            # Verify structure
            assert isinstance(result, dict)
            assert 'ratio_section' in result
            assert 'train_ratio' in result
            assert 'val_ratio' in result
            assert 'test_ratio' in result
            
        except Exception as e:
            pytest.fail(f"Ratio section execution test failed: {e}")


class TestSplitModuleStandardCompliance:
    """Test compliance with SmartCash UI module standards."""
    
    def test_display_initializer_compliance(self):
        """Test that the module follows DisplayInitializer pattern."""
        try:
            from smartcash.ui.dataset.split.split_initializer import SplitInitializer
            from smartcash.ui.core.initializers.display_initializer import DisplayInitializer
            
            # Check inheritance
            assert issubclass(SplitInitializer, DisplayInitializer)
            
            # Check required methods exist
            initializer = SplitInitializer()
            assert hasattr(initializer, '_initialize_impl')
            assert callable(initializer._initialize_impl)
            
        except Exception as e:
            pytest.fail(f"DisplayInitializer compliance test failed: {e}")
    
    def test_entry_point_compliance(self):
        """Test that standard entry points exist."""
        try:
            from smartcash.ui.dataset.split.split_initializer import (
                init_split_ui,
                get_split_initializer
            )
            
            # These are the standard entry points
            assert callable(init_split_ui)
            assert callable(get_split_initializer)
            
        except ImportError as e:
            pytest.fail(f"Entry point compliance test failed: {e}")
    
    def test_config_handler_compliance(self):
        """Test that config handler follows standard pattern."""
        try:
            from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler
            from smartcash.ui.core.handlers.config_handler import ConfigHandler
            
            # Check inheritance
            assert issubclass(SplitConfigHandler, ConfigHandler)
            
            # Check required methods exist
            handler = SplitConfigHandler()
            assert hasattr(handler, 'validate_config')
            assert hasattr(handler, 'load_config')
            assert hasattr(handler, 'update_config')
            assert callable(handler.validate_config)
            assert callable(handler.load_config)
            assert callable(handler.update_config)
            
        except Exception as e:
            pytest.fail(f"Config handler compliance test failed: {e}")
    
    def test_module_structure_compliance(self):
        """Test that module structure follows standards."""
        module_root = Path(__file__).parent.parent.parent.parent.parent / "smartcash" / "ui" / "dataset" / "split"
        
        # Check required directories exist
        required_dirs = ['components', 'configs', 'handlers']
        for dir_name in required_dirs:
            dir_path = module_root / dir_name
            assert dir_path.exists(), f"Required directory missing: {dir_name}"
            assert (dir_path / "__init__.py").exists(), f"__init__.py missing in {dir_name}"
        
        # Check required files exist
        required_files = ['split_initializer.py', '__init__.py']
        for file_name in required_files:
            file_path = module_root / file_name
            assert file_path.exists(), f"Required file missing: {file_name}"