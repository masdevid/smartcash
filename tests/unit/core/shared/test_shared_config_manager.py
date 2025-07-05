"""
Tests for SharedConfigManager class.
"""
import os
import sys
import logging
import pytest
from unittest.mock import Mock, patch, MagicMock, ANY
from datetime import datetime
from typing import Dict, Any, List, Optional
import copy

# Set up debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the project root to the Python path to ensure imports work correctly
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the actual classes we want to test
from smartcash.ui.core.shared.shared_config_manager import (
    SharedConfigManager,
    ConfigDiff,
    ConfigVersioningError
)

# Ensure ConfigDiff is not mocked by any test fixtures
from smartcash.ui.core.shared.shared_config_manager import ConfigDiff as RealConfigDiff
ConfigDiff = RealConfigDiff  # Use the real class, not a mock

# Import the actual get_config_manager function to patch it
from smartcash.common.config.manager import get_config_manager as real_get_config_manager

class TestSharedConfigManager:
    """Test cases for SharedConfigManager."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a test instance of SharedConfigManager with proper mocks."""
        # Reset singleton instances
        SharedConfigManager._instances = {}
        
        # Create a mock config manager
        mock_config = MagicMock()
        mock_config.get.return_value = {}
        mock_config.set.return_value = True
        mock_config.delete.return_value = True
        
        # Patch the get_config_manager function to return our mock
        with patch('smartcash.common.config.manager.get_config_manager', return_value=mock_config):
            # Create a real instance
            manager = SharedConfigManager.get_instance('test_module')
            
            # Reset any existing state
            manager._configs = {}
            manager._config_versions = {}
            manager._validation_rules = {}
            manager._config_templates = {}
            manager._subscribers = {}
            manager._last_updated = {}
            
            # Set up test data
            test_config = {'key1': 'value1', 'key2': 'value2'}
            manager._configs['test_config'] = test_config.copy()
            manager._config_versions['test_config'] = [
                {'key1': 'old_value1'},
                test_config.copy()
            ]
            
            yield manager
            
            # Cleanup
            SharedConfigManager._instances = {}
            
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the SharedConfigManager singleton between tests."""
        # Save the original instances
        original_instances = SharedConfigManager._instances.copy()
        SharedConfigManager._instances = {}
        
        yield
        
        # Restore the original instances
        SharedConfigManager._instances = original_instances
    
    @pytest.fixture
    def test_config(self):
        """Return a test config dictionary."""
        return {'key1': 'value1', 'key2': 'value2'}
        
    @pytest.fixture
    def test_diff(self):
        """Return a test ConfigDiff instance."""
        return ConfigDiff(
            added={'new_key': 'value'},
            removed={'old_key': 'value'},
            changed={'modified_key': {'old': 'old_val', 'new': 'new_val'}}
        )

    def test_get_instance_creates_singleton(self, config_manager):
        """Test that get_instance returns the same instance for the same module."""
        # Reset singleton instances to ensure clean state
        SharedConfigManager._instances = {}
        
        # First call should create a new instance
        instance1 = SharedConfigManager.get_instance('test_module')
        assert isinstance(instance1, SharedConfigManager)
        
        # Second call with same module should return same instance
        instance2 = SharedConfigManager.get_instance('test_module')
        assert instance1 is instance2
        
        # Different module should get different instance
        instance3 = SharedConfigManager.get_instance('another_module')
        assert instance1 is not instance3
        assert isinstance(instance3, SharedConfigManager)

    def test_get_config_returns_copy(self, config_manager, test_config):
        """Test that get_config returns a copy of the config."""
        manager = config_manager
        
        # Add test config
        manager._configs['test_config'] = test_config.copy()
        
        # Get the config
        config = manager.get_config('test_config')
        
        # Should return the expected config
        assert config == test_config
        
        # Should return a copy, not the original
        assert config is not manager._configs['test_config']
        
        # Modify the returned config and ensure original is not affected
        config['key1'] = 'modified'
        assert manager.get_config('test_config') == test_config

    def test_update_config_with_validation(self, config_manager):
        """Test that update_config validates the config if a validator exists."""
        manager = config_manager
        
        # Set up a validation rule that requires 'required_key' to be present
        def validator(config):
            return 'required_key' in config
            
        manager.set_validation_rule('test_config', validator)
        
        # Test valid config
        valid_config = {'required_key': True, 'other': 'value'}
        manager.update_config('test_config', valid_config)
        assert manager.get_config('test_config') == valid_config
        
        # Test invalid config
        with pytest.raises(ValueError, match="Invalid configuration"):
            manager.update_config('test_config', {'missing_required': True})

    def test_register_and_apply_template(self, config_manager):
        """Test template registration and application."""
        manager = config_manager
        
        # Initial config
        initial_config = {'key1': 'value1', 'key2': 'value2'}
        manager._configs['test_config'] = initial_config.copy()
        
        # Register a template
        template = {'key2': 'template_value', 'key3': 'new_value'}
        manager.register_template('test_template', template)
        
        # Apply template with merge
        manager.apply_template('test_config', 'test_template')
        
        # Get the updated config
        config = manager.get_config('test_config')
        
        # Should merge with existing config, with template values taking precedence
        assert config == {
            'key1': 'value1',  # From original
            'key2': 'template_value',  # Overridden by template
            'key3': 'new_value'  # Added by template
        }

    def test_get_config_diff(self, config_manager):
        """Test getting diff between config versions."""
        manager = config_manager
        
        # Set up version history
        manager._config_versions['test_config'] = [
            {'key1': 'value1', 'key2': 'old_value'},
            {'key1': 'value1', 'key2': 'new_value'}
        ]
        
        # Get diff between versions
        diff = manager.get_config_diff('test_config', 0, 1)
        
        # Verify the diff is correct
        assert diff.changed == {'key2': {'old': 'old_value', 'new': 'new_value'}}
        assert diff.added == {}
        assert diff.removed == {}
        assert diff.has_changes is True

    def test_rollback_config(self, config_manager):
        """Test rolling back to a previous config version."""
        manager = config_manager
        
        # Set up version history
        old_config = {'key1': 'old_value'}
        new_config = {'key1': 'new_value'}
        manager._config_versions['test_config'] = [old_config.copy()]
        manager._configs['test_config'] = new_config.copy()
        
        # Rollback to previous version
        assert manager.rollback_config('test_config', 0)
        
        # Verify rollback
        assert manager.get_config('test_config') == old_config
        
        # Version history should be updated
        assert len(manager._config_versions['test_config']) == 1  # Should have one version after rollback

    def test_get_stats(self, config_manager):
        """Test getting statistics about the config manager."""
        manager = config_manager
        
        # Add test data
        manager._configs = {'config1': {}, 'config2': {}}
        manager._config_versions = {'config1': [{}, {}], 'config2': [{}]}
        manager._validation_rules = {'config1': lambda x: True}
        manager._config_templates = {'template1': {}}
        manager._last_updated = {'config1': '2023-01-01T00:00:00'}
        
        # Get stats
        stats = manager.get_stats()
        
        # Verify stats
        assert stats['modules'] == 2  # config1 and config2
        assert stats['total_versions'] == 3  # 2 + 1 versions
        assert stats['validation_rules'] == 1  # One validation rule
        assert stats['templates'] == 1  # One template
        assert 'config1' in stats['last_updated']

class TestConfigDiff:
    """Test cases for ConfigDiff class."""
    
    # This test doesn't need the mock config manager
    def test_config_diff_initialization(self):
        """Test ConfigDiff initialization and properties."""
        logger.debug("Starting test_config_diff_initialization")
        # Test with changes
        added = {'new_key': 'value'}
        removed = {'old_key': 'value'}
        changed = {'modified_key': {'old': 'old_val', 'new': 'new_val'}}
        
        logger.debug(f"Creating ConfigDiff with added={added}, removed={removed}, changed={changed}")
        
        diff = ConfigDiff(added=added, removed=removed, changed=changed)
        
        # Test properties
        assert diff.added == added
        assert diff.removed == removed
        assert diff.changed == changed
        assert diff.has_changes is True
        
        # Test empty diff
        empty_diff = ConfigDiff(added={}, removed={}, changed={})
        assert empty_diff.has_changes is False
        
    # This test doesn't need the mock config manager
    def test_config_diff_equality(self):
        """Test ConfigDiff equality comparison."""
        logger.debug("Starting test_config_diff_equality")
        
        # Create first diff
        diff1_config = {
            'added': {'a': 1},
            'removed': {'b': 2},
            'changed': {'c': {'old': 1, 'new': 2}}
        }
        logger.debug(f"Creating first diff with config: {diff1_config}")
        diff1 = ConfigDiff(**diff1_config)
        
        # Create second diff with same values
        diff2 = ConfigDiff(
            added={'a': 1},
            removed={'b': 2},
            changed={'c': {'old': 1, 'new': 2}}
        )
        
        # Create third diff with different values
        diff3 = ConfigDiff(
            added={'x': 1}, 
            removed={}, 
            changed={}
        )
        
        # Test equality
        assert diff1 == diff2  # Same values should be equal
        assert diff1 != diff3  # Different values should not be equal
        assert diff1 != 'not a ConfigDiff'  # Different types should not be equal
        
        # Test with different values in added
        diff4 = ConfigDiff(
            added={'different': 1},  # Different added
            removed={'b': 2},
            changed={'c': {'old': 1, 'new': 2}}
        )
        assert diff1 != diff4
        
        # Test with different values in removed
        diff5 = ConfigDiff(
            added={'a': 1},
            removed={'different': 2},  # Different removed
            changed={'c': {'old': 1, 'new': 2}}
        )
        assert diff1 != diff5
        
        # Test with different values in changed
        diff6 = ConfigDiff(
            added={'a': 1},
            removed={'b': 2},
            changed={'different': {'old': 1, 'new': 2}}  # Different changed
        )
        assert diff1 != diff6
