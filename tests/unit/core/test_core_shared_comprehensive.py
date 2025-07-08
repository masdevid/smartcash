"""
Comprehensive tests for Core Shared Components module.

This module provides complete test coverage for the core shared infrastructure,
including SharedConfigManager functionality, error handling, and integration tests.
"""

import pytest
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
from datetime import datetime, timedelta

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from smartcash.ui.core.shared.shared_config_manager import (
    SharedConfigManager, 
    ConfigDiff, 
    ConfigError, 
    ConfigVersioningError,
    get_shared_config_manager,
    broadcast_config_update,
    subscribe_to_config
)


class TestSharedConfigManager:
    """Test suite for SharedConfigManager functionality."""
    
    @pytest.fixture
    def test_manager(self):
        """Create a SharedConfigManager instance for testing."""
        return SharedConfigManager("test_module")
    
    @pytest.fixture
    def fresh_manager(self):
        """Create a fresh manager instance that won't interfere with singleton."""
        # Clear singleton instances for testing
        original_instances = SharedConfigManager._instances.copy()
        SharedConfigManager._instances.clear()
        
        manager = SharedConfigManager("fresh_test")
        
        yield manager
        
        # Restore original instances
        SharedConfigManager._instances = original_instances
    
    def test_singleton_behavior(self):
        """Test that SharedConfigManager follows singleton pattern per module."""
        manager1 = SharedConfigManager.get_instance("test_singleton")
        manager2 = SharedConfigManager.get_instance("test_singleton")
        manager3 = SharedConfigManager.get_instance("different_module")
        
        assert manager1 is manager2
        assert manager1 is not manager3
        assert manager1.parent_module == "test_singleton"
        assert manager3.parent_module == "different_module"
    
    def test_basic_config_operations(self, fresh_manager):
        """Test basic configuration operations."""
        test_config = {"key1": "value1", "key2": "value2"}
        
        # Test setting config
        fresh_manager.update_config("test_module", test_config)
        
        # Test getting config
        retrieved_config = fresh_manager.get_config("test_module")
        assert retrieved_config == test_config
        assert retrieved_config is not test_config  # Should be a copy
        
        # Test updating config
        updated_config = {"key1": "updated_value", "key3": "new_value"}
        fresh_manager.update_config("test_module", updated_config)
        
        final_config = fresh_manager.get_config("test_module")
        assert final_config == updated_config
    
    def test_config_with_none_values(self, fresh_manager):
        """Test configuration with None values."""
        test_config = {"key1": None, "key2": "value", "key3": None}
        
        fresh_manager.update_config("test_none", test_config)
        retrieved_config = fresh_manager.get_config("test_none")
        
        assert retrieved_config == test_config
        assert retrieved_config["key1"] is None
        assert retrieved_config["key2"] == "value"
        assert retrieved_config["key3"] is None
    
    def test_get_nonexistent_config(self, fresh_manager):
        """Test getting configuration that doesn't exist."""
        result = fresh_manager.get_config("nonexistent_module")
        assert result is None
    
    def test_get_all_configs(self, fresh_manager):
        """Test getting all configurations."""
        config1 = {"module1": "config1"}
        config2 = {"module2": "config2"}
        
        fresh_manager.update_config("module1", config1)
        fresh_manager.update_config("module2", config2)
        
        all_configs = fresh_manager.get_all_configs()
        
        assert len(all_configs) == 2
        assert all_configs["module1"] == config1
        assert all_configs["module2"] == config2
    
    def test_clear_module_config(self, fresh_manager):
        """Test clearing specific module configuration."""
        test_config = {"key": "value"}
        
        fresh_manager.update_config("test_clear", test_config)
        assert fresh_manager.get_config("test_clear") == test_config
        
        fresh_manager.clear_module_config("test_clear")
        assert fresh_manager.get_config("test_clear") is None


class TestConfigVersioning:
    """Test configuration versioning functionality."""
    
    @pytest.fixture
    def versioned_manager(self):
        """Create a manager with versioning enabled."""
        # Clear singleton instances for testing
        original_instances = SharedConfigManager._instances.copy()
        SharedConfigManager._instances.clear()
        
        manager = SharedConfigManager("versioned_test")
        
        yield manager
        
        # Restore original instances
        SharedConfigManager._instances = original_instances
    
    def test_version_history_creation(self, versioned_manager):
        """Test that version history is created correctly."""
        config1 = {"version": 1}
        config2 = {"version": 2}
        config3 = {"version": 3}
        
        # First update
        versioned_manager.update_config("test_versioning", config1)
        assert versioned_manager.get_version_count("test_versioning") == 0  # No history yet
        
        # Second update should create history
        versioned_manager.update_config("test_versioning", config2)
        assert versioned_manager.get_version_count("test_versioning") == 1
        
        # Third update
        versioned_manager.update_config("test_versioning", config3)
        assert versioned_manager.get_version_count("test_versioning") == 2
    
    def test_config_diff(self, versioned_manager):
        """Test configuration diff functionality."""
        config1 = {"key1": "value1", "key2": "value2"}
        config2 = {"key1": "updated_value", "key3": "new_value"}
        
        # Set up version history
        versioned_manager.update_config("test_diff", config1)
        versioned_manager.update_config("test_diff", config2)
        
        # Get diff
        diff = versioned_manager.get_config_diff("test_diff", -2, -1)
        
        assert diff.added == {"key3": "new_value"}
        assert diff.removed == {"key2": "value2"}
        assert diff.changed == {"key1": {"old": "value1", "new": "updated_value"}}
        assert diff.has_changes
    
    def test_config_diff_no_changes(self, versioned_manager):
        """Test diff when there are no changes."""
        config = {"key": "value"}
        
        versioned_manager.update_config("test_no_diff", config)
        versioned_manager.update_config("test_no_diff", config.copy())
        
        diff = versioned_manager.get_config_diff("test_no_diff", -2, -1)
        
        assert not diff.added
        assert not diff.removed
        assert not diff.changed
        assert not diff.has_changes
    
    def test_rollback_config(self, versioned_manager):
        """Test configuration rollback functionality."""
        config1 = {"version": 1}
        config2 = {"version": 2}
        config3 = {"version": 3}
        
        # Build version history
        versioned_manager.update_config("test_rollback", config1)
        versioned_manager.update_config("test_rollback", config2)
        versioned_manager.update_config("test_rollback", config3)
        
        # Current should be config3
        assert versioned_manager.get_config("test_rollback") == config3
        
        # Rollback one step
        success = versioned_manager.rollback_config("test_rollback", 1)
        assert success
        assert versioned_manager.get_config("test_rollback") == config2
        
        # Rollback another step
        success = versioned_manager.rollback_config("test_rollback", 1)
        assert success
        assert versioned_manager.get_config("test_rollback") == config1
    
    def test_rollback_insufficient_history(self, versioned_manager):
        """Test rollback when there's insufficient history."""
        config = {"version": 1}
        
        versioned_manager.update_config("test_insufficient", config)
        
        # Try to rollback without sufficient history
        success = versioned_manager.rollback_config("test_insufficient", 2)
        assert not success
        
        # Config should remain unchanged
        assert versioned_manager.get_config("test_insufficient") == config
    
    def test_clear_version_history(self, versioned_manager):
        """Test clearing version history."""
        config1 = {"version": 1}
        config2 = {"version": 2}
        
        # Build version history
        versioned_manager.update_config("test_clear_history", config1)
        versioned_manager.update_config("test_clear_history", config2)
        
        assert versioned_manager.get_version_count("test_clear_history") == 1
        
        # Clear specific module history
        versioned_manager.clear_version_history("test_clear_history")
        assert versioned_manager.get_version_count("test_clear_history") == 0
        
        # Add another module
        versioned_manager.update_config("other_module", config1)
        versioned_manager.update_config("other_module", config2)
        
        # Clear all history
        versioned_manager.clear_version_history()
        assert versioned_manager.get_version_count("other_module") == 0


class TestConfigValidation:
    """Test configuration validation functionality."""
    
    @pytest.fixture
    def validation_manager(self):
        """Create a manager for validation testing."""
        # Clear singleton instances for testing
        original_instances = SharedConfigManager._instances.copy()
        SharedConfigManager._instances.clear()
        
        manager = SharedConfigManager("validation_test")
        
        yield manager
        
        # Restore original instances
        SharedConfigManager._instances = original_instances
    
    def test_set_validation_rule(self, validation_manager):
        """Test setting validation rules."""
        def validator(config):
            return "required_field" in config
        
        validation_manager.set_validation_rule("test_validation", validator)
        
        # Valid config should work
        valid_config = {"required_field": "value", "other": "data"}
        validation_manager.update_config("test_validation", valid_config)
        
        # Invalid config should fail
        invalid_config = {"other": "data"}
        with pytest.raises(ValueError, match="Config validation failed"):
            validation_manager.update_config("test_validation", invalid_config)
    
    def test_complex_validation_rule(self, validation_manager):
        """Test more complex validation rules."""
        def complex_validator(config):
            # Must have both 'name' and 'type' fields
            # 'type' must be one of specific values
            if "name" not in config or "type" not in config:
                return False
            return config["type"] in ["A", "B", "C"]
        
        validation_manager.set_validation_rule("complex_validation", complex_validator)
        
        # Valid configs
        valid_configs = [
            {"name": "test", "type": "A"},
            {"name": "test2", "type": "B", "extra": "field"},
            {"name": "test3", "type": "C"}
        ]
        
        for config in valid_configs:
            validation_manager.update_config("complex_validation", config)
        
        # Invalid configs
        invalid_configs = [
            {"name": "test"},  # Missing type
            {"type": "A"},     # Missing name
            {"name": "test", "type": "D"},  # Invalid type
            {}  # Missing both
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                validation_manager.update_config("complex_validation", config)
    
    def test_validation_with_none_values(self, validation_manager):
        """Test validation with None values in config."""
        def none_aware_validator(config):
            # Allow None values but require specific structure
            return "status" in config and (config["status"] is None or isinstance(config["status"], str))
        
        validation_manager.set_validation_rule("none_validation", none_aware_validator)
        
        # Valid configs with None
        valid_configs = [
            {"status": None},
            {"status": "active"},
            {"status": None, "other": "value"}
        ]
        
        for config in valid_configs:
            validation_manager.update_config("none_validation", config)
        
        # Invalid config
        invalid_config = {"other": "value"}  # Missing status
        with pytest.raises(ValueError):
            validation_manager.update_config("none_validation", invalid_config)


class TestConfigTemplates:
    """Test configuration template functionality."""
    
    @pytest.fixture
    def template_manager(self):
        """Create a manager for template testing."""
        # Clear singleton instances for testing
        original_instances = SharedConfigManager._instances.copy()
        SharedConfigManager._instances.clear()
        
        manager = SharedConfigManager("template_test")
        
        yield manager
        
        # Restore original instances
        SharedConfigManager._instances = original_instances
    
    def test_register_and_apply_template(self, template_manager):
        """Test registering and applying templates."""
        # Register a template
        default_template = {
            "setting1": "default_value1",
            "setting2": "default_value2",
            "nested": {"key": "value"}
        }
        
        template_manager.register_template("default", default_template)
        
        # Apply template
        template_manager.apply_template("test_module", "default")
        
        # Check that config matches template
        config = template_manager.get_config("test_module")
        assert config == default_template
        assert config is not default_template  # Should be a copy
    
    def test_apply_nonexistent_template(self, template_manager):
        """Test applying a template that doesn't exist."""
        with pytest.raises(ValueError, match="Template not found"):
            template_manager.apply_template("test_module", "nonexistent")
    
    def test_merge_template_with_existing_config(self, template_manager):
        """Test merging template with existing configuration."""
        # Set up existing config
        existing_config = {
            "existing_key": "existing_value",
            "shared_key": "original_value"
        }
        template_manager.update_config("merge_test", existing_config)
        
        # Register template
        template = {
            "template_key": "template_value",
            "shared_key": "template_value"
        }
        template_manager.register_template("merge_template", template)
        
        # Apply template with merge
        template_manager.apply_template("merge_test", "merge_template", merge=True)
        
        # Check merged result
        final_config = template_manager.get_config("merge_test")
        expected = {
            "template_key": "template_value",
            "shared_key": "original_value",  # Existing should override template
            "existing_key": "existing_value"
        }
        
        assert final_config == expected
    
    def test_replace_template_without_merge(self, template_manager):
        """Test replacing config with template without merge."""
        # Set up existing config
        existing_config = {"existing_key": "existing_value"}
        template_manager.update_config("replace_test", existing_config)
        
        # Register template
        template = {"template_key": "template_value"}
        template_manager.register_template("replace_template", template)
        
        # Apply template without merge
        template_manager.apply_template("replace_test", "replace_template", merge=False)
        
        # Check that config was completely replaced
        final_config = template_manager.get_config("replace_test")
        assert final_config == template
        assert "existing_key" not in final_config


class TestSubscriberNotifications:
    """Test subscriber notification functionality."""
    
    @pytest.fixture
    def notification_manager(self):
        """Create a manager for notification testing."""
        # Clear singleton instances for testing
        original_instances = SharedConfigManager._instances.copy()
        SharedConfigManager._instances.clear()
        
        manager = SharedConfigManager("notification_test")
        
        yield manager
        
        # Restore original instances
        SharedConfigManager._instances = original_instances
    
    def test_subscribe_and_notify(self, notification_manager):
        """Test basic subscription and notification."""
        notifications = []
        
        def callback(config):
            notifications.append(config.copy())
        
        # Subscribe
        unsubscribe = notification_manager.subscribe("test_notify", callback)
        
        # Update config
        test_config = {"key": "value"}
        notification_manager.update_config("test_notify", test_config)
        
        # Check notification
        assert len(notifications) == 1
        assert notifications[0] == test_config
        
        # Test unsubscribe
        unsubscribe()
        
        # Update again
        notification_manager.update_config("test_notify", {"key": "new_value"})
        
        # Should still have only one notification
        assert len(notifications) == 1
    
    def test_multiple_subscribers(self, notification_manager):
        """Test multiple subscribers to the same module."""
        notifications_1 = []
        notifications_2 = []
        
        def callback_1(config):
            notifications_1.append(config.copy())
        
        def callback_2(config):
            notifications_2.append(config.copy())
        
        # Subscribe both
        notification_manager.subscribe("multi_test", callback_1)
        notification_manager.subscribe("multi_test", callback_2)
        
        # Update config
        test_config = {"key": "value"}
        notification_manager.update_config("multi_test", test_config)
        
        # Both should receive notification
        assert len(notifications_1) == 1
        assert len(notifications_2) == 1
        assert notifications_1[0] == test_config
        assert notifications_2[0] == test_config
    
    def test_subscriber_isolation(self, notification_manager):
        """Test that subscribers only receive notifications for their module."""
        notifications_a = []
        notifications_b = []
        
        def callback_a(config):
            notifications_a.append(config)
        
        def callback_b(config):
            notifications_b.append(config)
        
        # Subscribe to different modules
        notification_manager.subscribe("module_a", callback_a)
        notification_manager.subscribe("module_b", callback_b)
        
        # Update module A
        notification_manager.update_config("module_a", {"a": "value"})
        
        # Only callback A should receive notification
        assert len(notifications_a) == 1
        assert len(notifications_b) == 0
        
        # Update module B
        notification_manager.update_config("module_b", {"b": "value"})
        
        # Now callback B should also have notification
        assert len(notifications_a) == 1
        assert len(notifications_b) == 1


class TestThreadSafety:
    """Test thread safety of SharedConfigManager."""
    
    @pytest.fixture
    def thread_manager(self):
        """Create a manager for thread safety testing."""
        # Clear singleton instances for testing
        original_instances = SharedConfigManager._instances.copy()
        SharedConfigManager._instances.clear()
        
        manager = SharedConfigManager("thread_test")
        
        yield manager
        
        # Restore original instances
        SharedConfigManager._instances = original_instances
    
    def test_concurrent_config_updates(self, thread_manager):
        """Test concurrent configuration updates."""
        results = []
        errors = []
        
        def update_config(thread_id):
            try:
                for i in range(5):
                    config = {"thread": thread_id, "iteration": i, "timestamp": time.time()}
                    thread_manager.update_config(f"thread_{thread_id}", config)
                    time.sleep(0.001)  # Small delay
                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_config, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        
        # Verify final state
        for i in range(5):
            final_config = thread_manager.get_config(f"thread_{i}")
            assert final_config is not None
            assert final_config["thread"] == i
            assert final_config["iteration"] == 4  # Last iteration
    
    def test_concurrent_subscriptions(self, thread_manager):
        """Test concurrent subscription operations."""
        all_notifications = []
        subscription_errors = []
        
        def subscriber_thread(thread_id):
            try:
                notifications = []
                
                def callback(config):
                    notifications.append((thread_id, config.copy()))
                
                # Subscribe
                unsubscribe = thread_manager.subscribe("concurrent_test", callback)
                
                # Wait a bit for potential updates
                time.sleep(0.1)
                
                # Unsubscribe
                unsubscribe()
                
                all_notifications.extend(notifications)
            except Exception as e:
                subscription_errors.append(f"Subscriber {thread_id} error: {e}")
        
        def updater_thread():
            try:
                for i in range(3):
                    config = {"update": i, "timestamp": time.time()}
                    thread_manager.update_config("concurrent_test", config)
                    time.sleep(0.05)
            except Exception as e:
                subscription_errors.append(f"Updater error: {e}")
        
        # Create subscriber threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=subscriber_thread, args=(i,))
            threads.append(thread)
        
        # Create updater thread
        updater = threading.Thread(target=updater_thread)
        threads.append(updater)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
        
        # Check for errors
        assert len(subscription_errors) == 0, f"Subscription errors: {subscription_errors}"
        
        # Should have received some notifications
        assert len(all_notifications) > 0


class TestManagerStatistics:
    """Test manager statistics and monitoring functionality."""
    
    @pytest.fixture
    def stats_manager(self):
        """Create a manager for statistics testing."""
        # Clear singleton instances for testing
        original_instances = SharedConfigManager._instances.copy()
        SharedConfigManager._instances.clear()
        
        manager = SharedConfigManager("stats_test")
        
        yield manager
        
        # Restore original instances
        SharedConfigManager._instances = original_instances
    
    def test_basic_stats(self, stats_manager):
        """Test basic statistics collection."""
        # Initially empty
        stats = stats_manager.get_stats()
        assert stats["modules"] == 0
        assert stats["total_versions"] == 0
        assert stats["validation_rules"] == 0
        assert stats["templates"] == 0
        
        # Add some data
        stats_manager.update_config("module1", {"key": "value"})
        stats_manager.update_config("module2", {"key": "value"})
        stats_manager.set_validation_rule("module1", lambda x: True)
        stats_manager.register_template("template1", {"default": "value"})
        
        # Check updated stats
        stats = stats_manager.get_stats()
        assert stats["modules"] == 2
        assert stats["validation_rules"] == 1
        assert stats["templates"] == 1
    
    def test_last_updated_tracking(self, stats_manager):
        """Test last updated time tracking."""
        # Update config
        before_time = datetime.now()
        stats_manager.update_config("time_test", {"key": "value"})
        after_time = datetime.now()
        
        # Check last updated time
        last_updated = stats_manager.get_last_updated("time_test")
        assert last_updated is not None
        assert before_time <= last_updated <= after_time
        
        # Test for non-existent module
        assert stats_manager.get_last_updated("nonexistent") is None


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_shared_config_manager(self):
        """Test get_shared_config_manager convenience function."""
        manager1 = get_shared_config_manager("convenience_test")
        manager2 = get_shared_config_manager("convenience_test")
        
        assert manager1 is manager2
        assert isinstance(manager1, SharedConfigManager)
        assert manager1.parent_module == "convenience_test"
    
    def test_broadcast_config_update(self):
        """Test broadcast_config_update convenience function."""
        notifications = []
        
        def callback(config):
            notifications.append(config)
        
        # Subscribe using convenience function
        unsubscribe = subscribe_to_config("broadcast_test", "test_module", callback)
        
        # Broadcast update
        test_config = {"broadcasted": True}
        broadcast_config_update("broadcast_test", "test_module", test_config, persist=False)
        
        # Check notification
        assert len(notifications) == 1
        assert notifications[0] == test_config
        
        # Cleanup
        unsubscribe()


class TestErrorHandling:
    """Test error handling in SharedConfigManager."""
    
    @pytest.fixture
    def error_manager(self):
        """Create a manager for error testing."""
        # Clear singleton instances for testing
        original_instances = SharedConfigManager._instances.copy()
        SharedConfigManager._instances.clear()
        
        manager = SharedConfigManager("error_test")
        
        yield manager
        
        # Restore original instances
        SharedConfigManager._instances = original_instances
    
    def test_config_versioning_errors(self, error_manager):
        """Test errors in config versioning."""
        # Test diff with no version history
        with pytest.raises(ConfigVersioningError, match="No version history"):
            error_manager.get_config_diff("nonexistent_module")
        
        # Test diff with insufficient versions
        error_manager.update_config("single_version", {"key": "value"})
        diff = error_manager.get_config_diff("single_version")
        
        # Should return empty diff
        assert not diff.has_changes
    
    def test_validation_errors(self, error_manager):
        """Test validation error handling."""
        def failing_validator(config):
            raise ValueError("Validator failed")
        
        error_manager.set_validation_rule("failing_validation", failing_validator)
        
        with pytest.raises(ValueError, match="Validator failed"):
            error_manager.update_config("failing_validation", {"key": "value"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])