"""
Comprehensive tests for ProgressTracker component with 100% coverage target.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import time
import threading
from typing import Dict, List

from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
from smartcash.ui.components.progress_tracker.types import ProgressConfig, ProgressLevel
from smartcash.ui.components.progress_tracker.progress_config import ProgressBarConfig, get_level_configs, get_default_weights, get_container_height
from smartcash.ui.components.progress_tracker.callback_manager import CallbackManager
from smartcash.ui.components.progress_tracker.tqdm_manager import TqdmManager
from smartcash.ui.components.progress_tracker.factory import (
    create_single_progress_tracker, create_dual_progress_tracker,
    create_triple_progress_tracker, create_flexible_tracker
)
from smartcash.ui.components.progress_tracker.progress_tracker import (
    create_progress_tracker, update_progress, complete_progress, error_progress
)


class TestProgressLevel:
    """Test ProgressLevel enum."""
    
    def test_progress_level_values(self):
        """Test ProgressLevel enum values."""
        assert ProgressLevel.SINGLE.value == 1
        assert ProgressLevel.DUAL.value == 2
        assert ProgressLevel.TRIPLE.value == 3


class TestProgressConfig:
    """Test ProgressConfig class."""
    
    def test_progress_config_defaults(self):
        """Test ProgressConfig with default values."""
        config = ProgressConfig()
        assert config.level == ProgressLevel.SINGLE
        assert config.operation == "Process"
        assert config.steps == ["Progress"]
        assert config.auto_advance is True
        assert config.auto_hide is False
        assert config.auto_hide_delay == 3600.0
        assert config.animation_speed == 0.1
        assert config.width_adjustment == 0
    
    def test_progress_config_custom_values(self):
        """Test ProgressConfig with custom values."""
        config = ProgressConfig(
            level=ProgressLevel.TRIPLE,
            operation="Custom Operation",
            steps=["Step1", "Step2", "Step3"],
            auto_advance=False,
            auto_hide=True,
            auto_hide_delay=1800.0,
            animation_speed=0.2,
            width_adjustment=10
        )
        assert config.level == ProgressLevel.TRIPLE
        assert config.operation == "Custom Operation"
        assert config.steps == ["Step1", "Step2", "Step3"]
        assert config.auto_advance is False
        assert config.auto_hide is True
        assert config.auto_hide_delay == 1800.0
        assert config.animation_speed == 0.2
        assert config.width_adjustment == 10
    
    def test_progress_config_dual_validation(self):
        """Test ProgressConfig validation for DUAL level."""
        config = ProgressConfig(level=ProgressLevel.DUAL)
        assert config.steps == ["Overall", "Current"]
    
    def test_progress_config_triple_validation(self):
        """Test ProgressConfig validation for TRIPLE level."""
        config = ProgressConfig(level=ProgressLevel.TRIPLE)
        assert config.steps == ["Overall", "Current", "Details"]
    
    def test_progress_config_invalid_level_raises_error(self):
        """Test ProgressConfig raises error for invalid level without steps."""
        with pytest.raises(ValueError, match="Steps harus diisi"):
            ProgressConfig(level=ProgressLevel.DUAL, steps=[])
    
    def test_get_step_weights_empty_steps(self):
        """Test get_step_weights with empty steps."""
        config = ProgressConfig(steps=[])
        weights = config.get_step_weights()
        assert weights == {}
    
    def test_get_step_weights_default(self):
        """Test get_step_weights with default weights."""
        config = ProgressConfig(steps=["Step1", "Step2"])
        weights = config.get_step_weights()
        assert weights == {"Step1": 1, "Step2": 1}
    
    def test_get_step_weights_custom(self):
        """Test get_step_weights with custom weights."""
        config = ProgressConfig(
            steps=["Step1", "Step2"],
            step_weights={"Step1": 2, "Step2": 3}
        )
        weights = config.get_step_weights()
        assert weights == {"Step1": 2, "Step2": 3}
    
    def test_get_step_weights_partial(self):
        """Test get_step_weights with partial weights."""
        config = ProgressConfig(
            steps=["Step1", "Step2", "Step3"],
            step_weights={"Step1": 2}
        )
        weights = config.get_step_weights()
        assert weights == {"Step1": 2, "Step2": 1, "Step3": 1}
    
    def test_get_container_height(self):
        """Test get_container_height for different levels."""
        single_config = ProgressConfig(level=ProgressLevel.SINGLE)
        assert single_config.get_container_height() == '120px'
        
        dual_config = ProgressConfig(level=ProgressLevel.DUAL)
        assert dual_config.get_container_height() == '160px'
        
        triple_config = ProgressConfig(level=ProgressLevel.TRIPLE)
        assert triple_config.get_container_height() == '200px'
    
    def test_get_level_configs(self):
        """Test get_level_configs method."""
        config = ProgressConfig(level=ProgressLevel.DUAL)
        level_configs = config.get_level_configs()
        assert len(level_configs) == 2
        assert all(isinstance(cfg, dict) for cfg in level_configs)


class TestProgressBarConfig:
    """Test ProgressBarConfig class."""
    
    def test_progress_bar_config_creation(self):
        """Test ProgressBarConfig creation."""
        config = ProgressBarConfig(
            name="test",
            description="Test Bar",
            emoji="🔄",
            color="#28a745",
            position=0,
            visible=True
        )
        assert config.name == "test"
        assert config.description == "Test Bar"
        assert config.emoji == "🔄"
        assert config.color == "#28a745"
        assert config.position == 0
        assert config.visible is True
    
    def test_progress_bar_config_defaults(self):
        """Test ProgressBarConfig with default values."""
        config = ProgressBarConfig(
            name="test",
            description="Test Bar",
            emoji="🔄",
            color="#28a745",
            position=0
        )
        assert config.visible is True
    
    def test_get_tqdm_color(self):
        """Test get_tqdm_color method."""
        config = ProgressBarConfig("test", "Test", "🔄", "#28a745", 0)
        assert config.get_tqdm_color() == "green"
        
        config = ProgressBarConfig("test", "Test", "🔄", "#007bff", 0)
        assert config.get_tqdm_color() == "blue"
        
        config = ProgressBarConfig("test", "Test", "🔄", "#unknown", 0)
        assert config.get_tqdm_color() == "blue"  # Default


class TestCallbackManager:
    """Test CallbackManager class."""
    
    def test_callback_manager_init(self):
        """Test CallbackManager initialization."""
        manager = CallbackManager()
        assert manager.callbacks == {}
        assert manager.one_time_callbacks == set()
    
    def test_register_callback(self):
        """Test register callback."""
        manager = CallbackManager()
        callback = Mock()
        callback_id = manager.register("test_event", callback)
        assert isinstance(callback_id, str)
        assert "test_event" in manager.callbacks
        assert len(manager.callbacks["test_event"]) == 1
    
    def test_register_one_time_callback(self):
        """Test register one-time callback."""
        manager = CallbackManager()
        callback = Mock()
        callback_id = manager.register("test_event", callback, one_time=True)
        assert callback_id in manager.one_time_callbacks
    
    def test_trigger_callback(self):
        """Test trigger callback."""
        manager = CallbackManager()
        callback = Mock()
        manager.register("test_event", callback)
        manager.trigger("test_event", "arg1", kwarg1="value1")
        callback.assert_called_once_with("arg1", kwarg1="value1")
    
    def test_trigger_one_time_callback(self):
        """Test trigger one-time callback removes it."""
        manager = CallbackManager()
        callback = Mock()
        callback_id = manager.register("test_event", callback, one_time=True)
        manager.trigger("test_event")
        callback.assert_called_once()
        assert callback_id not in manager.one_time_callbacks
        assert len(manager.callbacks["test_event"]) == 0
    
    def test_trigger_callback_with_exception(self):
        """Test trigger callback handles exceptions."""
        manager = CallbackManager()
        callback = Mock(side_effect=Exception("Test error"))
        callback_id = manager.register("test_event", callback)
        
        with patch('builtins.print') as mock_print:
            manager.trigger("test_event")
            mock_print.assert_called_once()
            assert "Callback error" in mock_print.call_args[0][0]
        
        # Callback should be removed after exception
        assert len(manager.callbacks["test_event"]) == 0
    
    def test_trigger_nonexistent_event(self):
        """Test trigger nonexistent event."""
        manager = CallbackManager()
        manager.trigger("nonexistent_event")  # Should not raise
    
    def test_unregister_callback(self):
        """Test unregister callback."""
        manager = CallbackManager()
        callback = Mock()
        callback_id = manager.register("test_event", callback)
        manager.unregister(callback_id)
        assert len(manager.callbacks["test_event"]) == 0
    
    def test_clear_event(self):
        """Test clear event callbacks."""
        manager = CallbackManager()
        callback1 = Mock()
        callback2 = Mock()
        manager.register("test_event", callback1)
        manager.register("test_event", callback2)
        manager.clear_event("test_event")
        assert len(manager.callbacks["test_event"]) == 0
    
    def test_clear_all(self):
        """Test clear all callbacks."""
        manager = CallbackManager()
        callback = Mock()
        manager.register("test_event", callback)
        manager.clear_all()
        assert manager.callbacks == {}
        assert manager.one_time_callbacks == set()
    
    def test_get_event_count(self):
        """Test get event count."""
        manager = CallbackManager()
        callback1 = Mock()
        callback2 = Mock()
        manager.register("test_event", callback1)
        manager.register("test_event", callback2)
        assert manager.get_event_count("test_event") == 2
        assert manager.get_event_count("nonexistent") == 0
    
    def test_has_callbacks(self):
        """Test has callbacks."""
        manager = CallbackManager()
        callback = Mock()
        assert not manager.has_callbacks("test_event")
        manager.register("test_event", callback)
        assert manager.has_callbacks("test_event")


class TestTqdmManager:
    """Test TqdmManager class."""
    
    def test_tqdm_manager_init(self):
        """Test TqdmManager initialization."""
        ui_manager = Mock()
        manager = TqdmManager(ui_manager)
        assert manager.ui_manager == ui_manager
        assert manager.tqdm_bars == {}
        assert manager.progress_values == {}
        assert manager.progress_messages == {}
    
    @patch('smartcash.ui.components.progress_tracker.tqdm_manager.tqdm')
    @patch('smartcash.ui.components.progress_tracker.tqdm_manager.clear_output')
    def test_initialize_bars(self, mock_clear_output, mock_tqdm):
        """Test initialize_bars method."""
        ui_manager = Mock()
        ui_manager._ui_components = {
            'overall_output': Mock(),
            'current_output': Mock()
        }
        manager = TqdmManager(ui_manager)
        
        bar_configs = [
            ProgressBarConfig("overall", "Overall", "📊", "#28a745", 0),
            ProgressBarConfig("current", "Current", "⚡", "#28a745", 1)
        ]
        
        manager.initialize_bars(bar_configs)
        
        assert mock_tqdm.call_count == 2
        assert mock_clear_output.call_count == 2
    
    def test_initialize_bars_invisible(self):
        """Test initialize_bars skips invisible bars."""
        ui_manager = Mock()
        ui_manager._ui_components = {'overall_output': Mock()}
        manager = TqdmManager(ui_manager)
        
        bar_configs = [
            ProgressBarConfig("overall", "Overall", "📊", "#28a745", 0, visible=False)
        ]
        
        with patch('smartcash.ui.components.progress_tracker.tqdm_manager.tqdm') as mock_tqdm:
            manager.initialize_bars(bar_configs)
            mock_tqdm.assert_not_called()
    
    def test_update_bar(self):
        """Test update_bar method."""
        ui_manager = Mock()
        manager = TqdmManager(ui_manager)
        
        mock_bar = Mock()
        manager.tqdm_bars["test"] = mock_bar
        
        manager.update_bar("test", 50, "Test message")
        
        assert mock_bar.n == 50
        mock_bar.refresh.assert_called_once()
        mock_bar.set_description.assert_called_once()
        assert manager.progress_values["test"] == 50
        assert manager.progress_messages["test"] == "Test message"
    
    def test_update_bar_clamps_progress(self):
        """Test update_bar clamps progress to valid range."""
        ui_manager = Mock()
        manager = TqdmManager(ui_manager)
        
        mock_bar = Mock()
        manager.tqdm_bars["test"] = mock_bar
        
        manager.update_bar("test", 150, "Test message")
        assert mock_bar.n == 100
        
        manager.update_bar("test", -10, "Test message")
        assert mock_bar.n == 0
    
    def test_update_bar_nonexistent(self):
        """Test update_bar with nonexistent bar."""
        ui_manager = Mock()
        manager = TqdmManager(ui_manager)
        
        # Should not raise exception
        manager.update_bar("nonexistent", 50, "Test message")
    
    def test_set_all_complete(self):
        """Test set_all_complete method."""
        ui_manager = Mock()
        manager = TqdmManager(ui_manager)
        
        mock_bar1 = Mock()
        mock_bar2 = Mock()
        manager.tqdm_bars["bar1"] = mock_bar1
        manager.tqdm_bars["bar2"] = mock_bar2
        
        manager.set_all_complete("Completed!")
        
        assert mock_bar1.n == 100
        assert mock_bar2.n == 100
        mock_bar1.refresh.assert_called_once()
        mock_bar2.refresh.assert_called_once()
        mock_bar1.set_description.assert_called_once()
        mock_bar2.set_description.assert_called_once()
    
    def test_set_all_error(self):
        """Test set_all_error method."""
        ui_manager = Mock()
        manager = TqdmManager(ui_manager)
        
        mock_bar = Mock()
        manager.tqdm_bars["test"] = mock_bar
        manager.progress_values["test"] = 75
        
        manager.set_all_error("Error occurred!")
        
        assert mock_bar.n == 75
        mock_bar.refresh.assert_called_once()
        mock_bar.set_description.assert_called_once()
    
    def test_close_all_bars(self):
        """Test close_all_bars method."""
        ui_manager = Mock()
        ui_manager._ui_components = {'overall_output': Mock()}
        manager = TqdmManager(ui_manager)
        
        mock_bar = Mock()
        manager.tqdm_bars["overall"] = mock_bar
        
        with patch('smartcash.ui.components.progress_tracker.tqdm_manager.clear_output') as mock_clear:
            manager.close_all_bars()
            mock_bar.close.assert_called_once()
            mock_clear.assert_called_once()
        
        assert manager.tqdm_bars == {}
    
    def test_close_all_bars_exception(self):
        """Test close_all_bars handles exceptions."""
        ui_manager = Mock()
        manager = TqdmManager(ui_manager)
        
        mock_bar = Mock()
        mock_bar.close.side_effect = Exception("Close error")
        manager.tqdm_bars["test"] = mock_bar
        
        # Should not raise exception
        manager.close_all_bars()
        assert manager.tqdm_bars == {}
    
    def test_reset(self):
        """Test reset method."""
        ui_manager = Mock()
        manager = TqdmManager(ui_manager)
        
        manager.progress_values["test"] = 50
        manager.progress_messages["test"] = "Test"
        
        with patch.object(manager, 'close_all_bars') as mock_close:
            manager.reset()
            mock_close.assert_called_once()
        
        assert manager.progress_values == {}
        assert manager.progress_messages == {}
    
    def test_get_progress_value(self):
        """Test get_progress_value method."""
        ui_manager = Mock()
        manager = TqdmManager(ui_manager)
        
        manager.progress_values["test"] = 75
        assert manager.get_progress_value("test") == 75
        assert manager.get_progress_value("nonexistent") == 0
    
    def test_get_progress_message(self):
        """Test get_progress_message method."""
        ui_manager = Mock()
        manager = TqdmManager(ui_manager)
        
        manager.progress_messages["test"] = "Test message"
        assert manager.get_progress_message("test") == "Test message"
        assert manager.get_progress_message("nonexistent") == ""
    
    def test_clean_message(self):
        """Test _clean_message static method."""
        # Test with emoji removal
        result = TqdmManager._clean_message("📊 Test message")
        assert result == "Test message"
        
        # Test with progress indicators
        result = TqdmManager._clean_message("Test message [50%]")
        assert result == "Test message"
        
        result = TqdmManager._clean_message("Test message (1/10)")
        assert result == "Test message"
        
        # Test with multiple spaces
        result = TqdmManager._clean_message("Test   message")
        assert result == "Test message"
        
        # Test with empty string
        result = TqdmManager._clean_message("")
        assert result == ""
        
        # Test with None
        result = TqdmManager._clean_message(None)
        assert result is None
    
    def test_has_emoji(self):
        """Test _has_emoji static method."""
        assert TqdmManager._has_emoji("📊 Test")
        assert TqdmManager._has_emoji("🔄 Processing")
        assert not TqdmManager._has_emoji("Test message")
        assert not TqdmManager._has_emoji("")
    
    def test_truncate_message(self):
        """Test _truncate_message static method."""
        result = TqdmManager._truncate_message("Short message", 20)
        assert result == "Short message"
        
        result = TqdmManager._truncate_message("This is a very long message", 10)
        assert result == "This is..."
        assert len(result) == 10


class TestProgressTracker:
    """Test ProgressTracker class."""
    
    def test_progress_tracker_init_default(self):
        """Test ProgressTracker initialization with default config."""
        tracker = ProgressTracker()
        assert tracker.component_name == "progress_tracker"
        assert isinstance(tracker._config, ProgressConfig)
        assert isinstance(tracker.callback_manager, CallbackManager)
        assert tracker.tqdm_manager is None
        assert tracker._current_step_index == 0
        assert tracker._is_complete is False
        assert tracker._is_error is False
    
    def test_progress_tracker_init_custom_config(self):
        """Test ProgressTracker initialization with custom config."""
        config = ProgressConfig(level=ProgressLevel.DUAL, operation="Custom Op")
        tracker = ProgressTracker("custom_tracker", config)
        assert tracker.component_name == "custom_tracker"
        assert tracker._config == config
        assert tracker._config.operation == "Custom Op"
    
    @patch('smartcash.ui.components.progress_tracker.progress_tracker.widgets')
    def test_create_ui_components(self, mock_widgets):
        """Test _create_ui_components method."""
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HTML.return_value = Mock()
        mock_widgets.Output.return_value = Mock()
        mock_widgets.Layout.return_value = Mock()
        
        tracker = ProgressTracker()
        tracker._create_ui_components()
        
        assert 'container' in tracker._ui_components
        assert 'header' in tracker._ui_components
        assert 'status' in tracker._ui_components
        assert 'overall_output' in tracker._ui_components
        assert 'step_output' in tracker._ui_components
        assert 'current_output' in tracker._ui_components
    
    def test_progress_bar_property(self):
        """Test progress_bar property."""
        tracker = ProgressTracker()
        
        # Test with no tqdm_manager
        assert tracker.progress_bar is None
        
        # Test with tqdm_manager but no bars
        tracker.tqdm_manager = Mock()
        tracker.tqdm_manager.bars = {}
        assert tracker.progress_bar is None
        
        # Test with tqdm_manager and bars
        mock_bar = Mock()
        tracker.tqdm_manager.bars = {"test": mock_bar}
        assert tracker.progress_bar == mock_bar
    
    def test_status_label_property(self):
        """Test status_label property."""
        tracker = ProgressTracker()
        tracker._ui_components = {'status': Mock()}
        assert tracker.status_label is not None
    
    def test_update_container_height(self):
        """Test _update_container_height method."""
        tracker = ProgressTracker()
        
        # Test with no ui_components
        tracker._update_container_height()  # Should not raise
        
        # Test with ui_components
        mock_container = Mock()
        mock_container.layout = Mock()
        tracker._ui_components = {'container': mock_container}
        
        tracker._update_container_height()
        
        # Should update height based on level
        assert mock_container.layout.height == '90px'  # 60 + 30 for SINGLE
        assert mock_container.layout.min_height == '90px'
    
    def test_register_default_callbacks(self):
        """Test _register_default_callbacks method."""
        tracker = ProgressTracker()
        
        # Check that callbacks are registered
        assert tracker.callback_manager.has_callbacks('complete')
        assert tracker.callback_manager.has_callbacks('progress_update')
    
    def test_sync_progress_state(self):
        """Test _sync_progress_state method."""
        tracker = ProgressTracker()
        
        # Test with no tqdm_manager
        tracker._sync_progress_state("test", 50, "message")  # Should not raise
        
        # Test with tqdm_manager
        mock_manager = Mock()
        tracker.tqdm_manager = mock_manager
        
        tracker._sync_progress_state("test", 50, "message")
        mock_manager.update_bar.assert_called_once_with("test", 50, "message")
    
    def test_delayed_hide(self):
        """Test _delayed_hide method."""
        config = ProgressConfig(auto_hide=True, auto_hide_delay=0.1)
        tracker = ProgressTracker(config=config)
        
        with patch.object(tracker, 'hide') as mock_hide:
            with patch('threading.Thread') as mock_thread:
                tracker._delayed_hide(0.01)
                mock_thread.assert_called_once()
                
                # Call the thread function
                thread_func = mock_thread.call_args[1]['target']
                thread_func()
                mock_hide.assert_called_once()
    
    def test_delayed_hide_disabled(self):
        """Test _delayed_hide when auto_hide is disabled."""
        config = ProgressConfig(auto_hide=False)
        tracker = ProgressTracker(config=config)
        
        with patch.object(tracker, 'hide') as mock_hide:
            tracker._delayed_hide()
            mock_hide.assert_not_called()
    
    @patch('smartcash.ui.components.progress_tracker.progress_tracker.widgets')
    def test_show(self, mock_widgets):
        """Test show method."""
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HTML.return_value = Mock()
        mock_widgets.Output.return_value = Mock()
        mock_widgets.Layout.return_value = Mock()
        
        tracker = ProgressTracker()
        tracker.show("Test Operation", ["Step1", "Step2"])
        
        assert tracker._config.operation == "Test Operation"
        assert tracker._config.steps == ["Step1", "Step2"]
        assert tracker._initialized is True
    
    @patch('smartcash.ui.components.progress_tracker.progress_tracker.widgets')
    def test_hide(self, mock_widgets):
        """Test hide method."""
        mock_widgets.VBox.return_value = Mock()
        mock_widgets.HTML.return_value = Mock()
        mock_widgets.Output.return_value = Mock()
        mock_widgets.Layout.return_value = Mock()
        
        tracker = ProgressTracker()
        tracker.initialize()
        
        mock_container = Mock()
        mock_container.layout = Mock()
        tracker._ui_components['container'] = mock_container
        
        tracker.hide()
        
        assert mock_container.layout.display == 'none'
        assert mock_container.layout.visibility == 'hidden'
    
    def test_update_status(self):
        """Test update_status method."""
        tracker = ProgressTracker()
        
        # Test with uninitialized tracker
        tracker.update_status("Test message")
        assert tracker._initialized is True
    
    def test_set_progress(self):
        """Test set_progress method."""
        tracker = ProgressTracker()
        
        # Test with no tqdm_manager
        tracker.set_progress(50, "test", "message")  # Should not raise
        
        # Test with tqdm_manager
        mock_manager = Mock()
        tracker.tqdm_manager = mock_manager
        
        tracker.set_progress(50, "test", "message")
        mock_manager.update_bar.assert_called_once_with("test", 50, "message")
    
    def test_complete(self):
        """Test complete method."""
        tracker = ProgressTracker()
        
        mock_manager = Mock()
        tracker.tqdm_manager = mock_manager
        
        with patch.object(tracker.callback_manager, 'trigger') as mock_trigger:
            tracker.complete("Done!")
            
            mock_manager.set_all_complete.assert_called_once()
            mock_trigger.assert_called_once_with('complete')
            assert tracker._is_complete is True
    
    def test_error(self):
        """Test error method."""
        tracker = ProgressTracker()
        
        mock_manager = Mock()
        tracker.tqdm_manager = mock_manager
        
        with patch.object(tracker.callback_manager, 'trigger') as mock_trigger:
            tracker.error("Error occurred!")
            
            mock_manager.set_all_error.assert_called_once_with("Error occurred!")
            mock_trigger.assert_called_once_with('error', "Error occurred!")
            assert tracker._is_error is True
    
    def test_reset(self):
        """Test reset method."""
        tracker = ProgressTracker()
        
        mock_manager = Mock()
        tracker.tqdm_manager = mock_manager
        
        tracker._is_complete = True
        tracker._is_error = True
        tracker._current_step_index = 5
        
        with patch.object(tracker.callback_manager, 'trigger') as mock_trigger:
            tracker.reset()
            
            mock_manager.initialize_bars.assert_called_once()
            mock_trigger.assert_called_once_with('reset')
            assert tracker._is_complete is False
            assert tracker._is_error is False
            assert tracker._current_step_index == 0
    
    def test_callback_registration_methods(self):
        """Test callback registration methods."""
        tracker = ProgressTracker()
        
        # Test callback registration
        progress_callback = Mock()
        step_callback = Mock()
        complete_callback = Mock()
        error_callback = Mock()
        reset_callback = Mock()
        
        progress_id = tracker.on_progress_update(progress_callback)
        step_id = tracker.on_step_complete(step_callback)
        complete_id = tracker.on_complete(complete_callback)
        error_id = tracker.on_error(error_callback)
        reset_id = tracker.on_reset(reset_callback)
        
        assert isinstance(progress_id, str)
        assert isinstance(step_id, str)
        assert isinstance(complete_id, str)
        assert isinstance(error_id, str)
        assert isinstance(reset_id, str)
        
        # Test callback removal
        tracker.remove_callback(progress_id)
        assert not tracker.callback_manager.has_callbacks('progress_update')
    
    def test_backward_compatibility_properties(self):
        """Test backward compatibility properties."""
        tracker = ProgressTracker()
        
        # Test container property
        tracker._ui_components = {'container': Mock()}
        assert tracker.container is not None
        
        # Test status_widget property
        tracker._ui_components = {'status': Mock()}
        assert tracker.status_widget is not None
        
        # Test step_info_widget property (always None)
        assert tracker.step_info_widget is None
        
        # Test progress_bars property
        assert tracker.progress_bars == {'main': None}
        
        # Test with tqdm_manager
        mock_manager = Mock()
        mock_manager.tqdm_bars = {'test': Mock()}
        tracker.tqdm_manager = mock_manager
        tracker._config.level = ProgressLevel.DUAL
        
        assert tracker.progress_bars == {'test': mock_manager.tqdm_bars['test']}


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_single_progress_tracker(self):
        """Test create_single_progress_tracker."""
        tracker = create_single_progress_tracker("Test Op", auto_hide=True)
        assert isinstance(tracker, ProgressTracker)
        assert tracker._config.level == ProgressLevel.SINGLE
        assert tracker._config.operation == "Test Op"
        assert tracker._config.auto_hide is True
    
    def test_create_dual_progress_tracker(self):
        """Test create_dual_progress_tracker."""
        tracker = create_dual_progress_tracker("Test Op", auto_hide=True)
        assert isinstance(tracker, ProgressTracker)
        assert tracker._config.level == ProgressLevel.DUAL
        assert tracker._config.operation == "Test Op"
        assert tracker._config.auto_hide is True
    
    def test_create_triple_progress_tracker(self):
        """Test create_triple_progress_tracker."""
        tracker = create_triple_progress_tracker(
            "Test Op",
            steps=["S1", "S2", "S3"],
            step_weights={"S1": 1, "S2": 2, "S3": 3},
            auto_hide=True
        )
        assert isinstance(tracker, ProgressTracker)
        assert tracker._config.level == ProgressLevel.TRIPLE
        assert tracker._config.operation == "Test Op"
        assert tracker._config.steps == ["S1", "S2", "S3"]
        assert tracker._config.step_weights == {"S1": 1, "S2": 2, "S3": 3}
        assert tracker._config.auto_hide is True
    
    def test_create_triple_progress_tracker_defaults(self):
        """Test create_triple_progress_tracker with defaults."""
        tracker = create_triple_progress_tracker()
        assert tracker._config.steps == ["Initialization", "Processing", "Completion"]
        assert tracker._config.step_weights == {}
    
    def test_create_flexible_tracker(self):
        """Test create_flexible_tracker."""
        config = ProgressConfig(level=ProgressLevel.DUAL, operation="Flexible")
        tracker = create_flexible_tracker(config)
        assert isinstance(tracker, ProgressTracker)
        assert tracker._config == config


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_level_configs(self):
        """Test get_level_configs function."""
        single_configs = get_level_configs(ProgressLevel.SINGLE)
        assert len(single_configs) == 1
        assert single_configs[0].name == "primary"
        
        dual_configs = get_level_configs(ProgressLevel.DUAL)
        assert len(dual_configs) == 2
        assert dual_configs[0].name == "overall"
        assert dual_configs[1].name == "current"
        
        triple_configs = get_level_configs(ProgressLevel.TRIPLE)
        assert len(triple_configs) == 3
        assert triple_configs[0].name == "overall"
        assert triple_configs[1].name == "step"
        assert triple_configs[2].name == "current"
    
    def test_get_default_weights(self):
        """Test get_default_weights function."""
        weights = get_default_weights(["A", "B", "C"])
        assert weights == {"A": 34, "B": 33, "C": 33}
        
        weights = get_default_weights(["A", "B"])
        assert weights == {"A": 50, "B": 50}
        
        weights = get_default_weights([])
        assert weights == {}
    
    def test_get_container_height(self):
        """Test get_container_height function."""
        assert get_container_height(ProgressLevel.SINGLE) == '120px'
        assert get_container_height(ProgressLevel.DUAL) == '160px'
        assert get_container_height(ProgressLevel.TRIPLE) == '200px'


class TestLegacyFunctions:
    """Test legacy compatibility functions."""
    
    def test_create_progress_tracker(self):
        """Test create_progress_tracker legacy function."""
        config = ProgressConfig(level=ProgressLevel.DUAL)
        tracker = create_progress_tracker(config)
        assert isinstance(tracker, ProgressTracker)
        assert tracker.component_name == "legacy_progress_tracker"
        assert tracker._config == config
    
    def test_create_progress_tracker_default(self):
        """Test create_progress_tracker with default config."""
        tracker = create_progress_tracker()
        assert isinstance(tracker, ProgressTracker)
        assert isinstance(tracker._config, ProgressConfig)
    
    def test_update_progress_with_tracker(self):
        """Test update_progress with valid tracker."""
        tracker = Mock()
        tracker.set_progress = Mock()
        
        update_progress(tracker, 50, "Test message", "test_level")
        tracker.set_progress.assert_called_once_with(50, "test_level", "Test message")
    
    def test_update_progress_with_legacy_tracker(self):
        """Test update_progress with legacy tracker."""
        tracker = Mock()
        tracker.update_bar = Mock()
        tracker.tqdm_bars = {}
        delattr(tracker, 'set_progress')  # Remove set_progress method
        
        update_progress(tracker, 50, "Test message", "test_level")
        tracker.update_bar.assert_called_once_with("test_level", 50, "Test message")
    
    def test_complete_progress_with_tracker(self):
        """Test complete_progress with valid tracker."""
        tracker = Mock()
        tracker.complete = Mock()
        
        complete_progress(tracker, "Done!")
        tracker.complete.assert_called_once_with("Done!")
    
    def test_complete_progress_with_legacy_tracker(self):
        """Test complete_progress with legacy tracker."""
        tracker = Mock()
        tracker.set_all_complete = Mock()
        tracker._config = Mock()
        tracker._config.get_level_configs = Mock(return_value=[])
        delattr(tracker, 'complete')  # Remove complete method
        
        complete_progress(tracker, "Done!")
        tracker.set_all_complete.assert_called_once_with("Done!", [])
    
    def test_error_progress_with_tracker(self):
        """Test error_progress with valid tracker."""
        tracker = Mock()
        tracker.error = Mock()
        
        error_progress(tracker, "Error!")
        tracker.error.assert_called_once_with("Error!")
    
    def test_error_progress_with_legacy_tracker(self):
        """Test error_progress with legacy tracker."""
        tracker = Mock()
        tracker.set_all_error = Mock()
        delattr(tracker, 'error')  # Remove error method
        
        error_progress(tracker, "Error!")
        tracker.set_all_error.assert_called_once_with("Error!")


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_progress_tracker_no_tqdm_manager(self):
        """Test ProgressTracker operations with no tqdm_manager."""
        tracker = ProgressTracker()
        
        # Should not raise exceptions
        tracker.set_progress(50)
        tracker.complete()
        tracker.error()
        tracker.reset()
    
    def test_tqdm_manager_no_ui_components(self):
        """Test TqdmManager with missing UI components."""
        ui_manager = Mock()
        ui_manager._ui_components = {}
        manager = TqdmManager(ui_manager)
        
        bar_configs = [
            ProgressBarConfig("overall", "Overall", "📊", "#28a745", 0)
        ]
        
        # Should not raise exceptions
        manager.initialize_bars(bar_configs)
        assert manager.tqdm_bars == {}
    
    def test_callback_manager_double_removal(self):
        """Test CallbackManager handles double removal."""
        manager = CallbackManager()
        callback = Mock()
        callback_id = manager.register("test", callback)
        
        manager.unregister(callback_id)
        manager.unregister(callback_id)  # Should not raise
    
    def test_progress_config_post_init_edge_cases(self):
        """Test ProgressConfig post_init edge cases."""
        # Test with TRIPLE level but no steps should raise
        with pytest.raises(ValueError):
            ProgressConfig(level=ProgressLevel.TRIPLE, steps=[])
        
        # Test with custom steps
        config = ProgressConfig(
            level=ProgressLevel.TRIPLE,
            steps=["Custom1", "Custom2"]
        )
        assert config.steps == ["Custom1", "Custom2"]
        
        # Test step_weights initialization
        config = ProgressConfig(steps=["A", "B"])
        assert config.step_weights == {"A": 1, "B": 1}