"""
Tests for EvaluationUIHandler
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio

from smartcash.ui.model.evaluate.handlers.evaluation_ui_handler import EvaluationUIHandler
from smartcash.ui.model.evaluate.constants import (
    DEFAULT_CONFIG, AVAILABLE_SCENARIOS, AVAILABLE_MODELS, 
    DEFAULT_ENABLED_METRICS, AVAILABLE_METRICS
)


class TestEvaluationUIHandler:
    """Test suite for EvaluationUIHandler."""
    
    @pytest.fixture
    def handler(self):
        """Create EvaluationUIHandler instance."""
        return EvaluationUIHandler()
    
    @pytest.fixture
    def mock_ui_components(self):
        """Create mock UI components."""
        components = {}
        
        # Scenario checkboxes
        components['position_variation_checkbox'] = Mock()
        components['position_variation_checkbox'].value = True
        components['lighting_variation_checkbox'] = Mock()
        components['lighting_variation_checkbox'].value = True
        
        # Model checkboxes
        components['cspdarknet_checkbox'] = Mock()
        components['cspdarknet_checkbox'].value = True
        components['efficientnet_checkbox'] = Mock()
        components['efficientnet_checkbox'].value = True
        
        # Settings sliders
        components['confidence_threshold_slider'] = Mock()
        components['confidence_threshold_slider'].value = 0.25
        components['iou_threshold_slider'] = Mock()
        components['iou_threshold_slider'].value = 0.45
        components['num_variations_slider'] = Mock()
        components['num_variations_slider'].value = 5.0
        
        # Metric checkboxes
        for metric in AVAILABLE_METRICS:
            checkbox_name = f'{metric}_metric_checkbox'
            components[checkbox_name] = Mock()
            components[checkbox_name].value = metric in DEFAULT_ENABLED_METRICS
        
        # Action buttons
        components['run_scenario_btn'] = Mock()
        components['run_comprehensive_btn'] = Mock()
        components['load_checkpoint_btn'] = Mock()
        components['list_checkpoints_btn'] = Mock()
        components['stop_evaluation_btn'] = Mock()
        components['save_config_btn'] = Mock()
        components['reset_config_btn'] = Mock()
        
        # UI display components
        components['evaluation_summary'] = Mock()
        components['evaluation_summary'].value = ""
        components['progress_tracker'] = Mock()
        
        return components
    
    def test_initialization(self, handler):
        """Test handler initialization."""
        assert handler.module_name == 'evaluate'
        assert handler.parent_module == 'model'
        assert handler.current_config == DEFAULT_CONFIG
        assert handler.evaluation_active is False
        assert handler.current_results == {}
        assert handler.selected_scenarios == set(AVAILABLE_SCENARIOS)
        assert handler.selected_models == set(AVAILABLE_MODELS)
        assert handler.selected_metrics == set(DEFAULT_ENABLED_METRICS)
    
    def test_extract_config_from_ui(self, handler, mock_ui_components):
        """Test configuration extraction from UI components."""
        handler._ui_components = mock_ui_components
        
        # Test with various checkbox states
        mock_ui_components['position_variation_checkbox'].value = True
        mock_ui_components['lighting_variation_checkbox'].value = False
        mock_ui_components['cspdarknet_checkbox'].value = False
        mock_ui_components['efficientnet_checkbox'].value = True
        
        config = handler.extract_config_from_ui()
        
        # Check scenario selections
        assert config['evaluation']['scenarios']['position_variation']['enabled'] is True
        assert config['evaluation']['scenarios']['lighting_variation']['enabled'] is False
        
        # Check model selections
        assert 'position_variation' in handler.selected_scenarios
        assert 'lighting_variation' not in handler.selected_scenarios
        assert 'cspdarknet' not in handler.selected_models
        assert 'efficientnet_b4' in handler.selected_models
        
        # Check inference settings
        assert config['inference']['confidence_threshold'] == 0.25
        assert config['inference']['iou_threshold'] == 0.45
        
        # Check augmentation settings
        for scenario_config in config['evaluation']['scenarios'].values():
            if 'augmentation_config' in scenario_config:
                assert scenario_config['augmentation_config']['num_variations'] == 5
    
    def test_update_ui_from_config(self, handler, mock_ui_components):
        """Test UI update from configuration."""
        handler._ui_components = mock_ui_components
        
        # Custom config
        config = {
            'evaluation': {
                'scenarios': {
                    'position_variation': {'enabled': False},
                    'lighting_variation': {'enabled': True}
                },
                'selected_metrics': ['map', 'precision']
            },
            'inference': {
                'confidence_threshold': 0.5,
                'iou_threshold': 0.6
            }
        }
        
        handler.update_ui_from_config(config)
        
        # Verify UI components were updated
        assert mock_ui_components['position_variation_checkbox'].value is False
        assert mock_ui_components['lighting_variation_checkbox'].value is True
        assert mock_ui_components['confidence_threshold_slider'].value == 0.5
        assert mock_ui_components['iou_threshold_slider'].value == 0.6
        
        # Verify metric checkboxes
        assert mock_ui_components['map_metric_checkbox'].value is True
        assert mock_ui_components['precision_metric_checkbox'].value is True
        assert mock_ui_components['recall_metric_checkbox'].value is False
    
    def test_setup_with_ui_components(self, handler, mock_ui_components):
        """Test handler setup with UI components."""
        handler.setup(mock_ui_components)
        
        assert handler._ui_components == mock_ui_components
        
        # Verify event handlers were set up (mock observe calls)
        for checkbox in ['position_variation_checkbox', 'lighting_variation_checkbox']:
            mock_ui_components[checkbox].observe.assert_called()
        
        for button in ['run_scenario_btn', 'run_comprehensive_btn']:
            mock_ui_components[button].on_click.assert_called()
    
    def test_scenario_change_handling(self, handler, mock_ui_components):
        """Test scenario selection change handling."""
        handler._ui_components = mock_ui_components
        
        # Simulate scenario change
        handler._on_scenario_change('position_variation_checkbox', None)
        
        # Should trigger config extraction and summary update
        # (Actual implementation depends on checkbox values)
        pass
    
    def test_model_change_handling(self, handler, mock_ui_components):
        """Test model selection change handling."""
        handler._ui_components = mock_ui_components
        
        # Simulate model change
        handler._on_model_change('cspdarknet_checkbox', None)
        
        # Should trigger config extraction and summary update
        pass
    
    def test_handle_run_scenario_success(self, handler, mock_ui_components):
        """Test single scenario evaluation handling."""
        handler._ui_components = mock_ui_components
        handler.selected_scenarios = {'position_variation'}
        handler.selected_models = {'cspdarknet'}
        
        with patch.object(handler, '_run_async_operation') as mock_async:
            handler._handle_run_scenario()
            
            mock_async.assert_called_once()
            args, kwargs = mock_async.call_args
            config = args[1]  # Second argument is config
            assert config['scenario'] == 'position_variation'
            assert config['model'] == 'cspdarknet'
    
    def test_handle_run_scenario_no_selection(self, handler, mock_ui_components):
        """Test scenario evaluation with no selections."""
        handler._ui_components = mock_ui_components
        handler.selected_scenarios = set()  # No scenarios selected
        handler.selected_models = {'cspdarknet'}
        
        with patch.object(handler, 'track_status') as mock_track:
            handler._handle_run_scenario()
            
            mock_track.assert_called_with(
                "❌ Please select at least one scenario and one model", "error"
            )
    
    def test_handle_run_scenario_already_active(self, handler, mock_ui_components):
        """Test scenario evaluation when already active."""
        handler._ui_components = mock_ui_components
        handler.evaluation_active = True
        
        with patch.object(handler, 'track_status') as mock_track:
            handler._handle_run_scenario()
            
            mock_track.assert_called_with(
                "⚠️ Evaluation is already in progress", "warning"
            )
    
    def test_handle_run_comprehensive(self, handler, mock_ui_components):
        """Test comprehensive evaluation handling."""
        handler._ui_components = mock_ui_components
        handler.selected_scenarios = {'position_variation', 'lighting_variation'}
        handler.selected_models = {'cspdarknet', 'efficientnet_b4'}
        
        with patch.object(handler, '_run_async_operation') as mock_async:
            handler._handle_run_comprehensive()
            
            mock_async.assert_called_once()
            args, kwargs = mock_async.call_args
            config = args[1]  # Second argument is config
            assert set(config['scenarios']) == handler.selected_scenarios
            assert set(config['models']) == handler.selected_models
    
    def test_handle_load_checkpoint(self, handler, mock_ui_components):
        """Test checkpoint loading handling."""
        handler._ui_components = mock_ui_components
        handler.selected_models = {'cspdarknet'}
        
        with patch.object(handler, '_run_async_operation') as mock_async:
            handler._handle_load_checkpoint()
            
            mock_async.assert_called_once()
            args, kwargs = mock_async.call_args
            config = args[1]  # Second argument is config
            assert config['action'] == 'load'
            assert config['model'] == 'cspdarknet'
    
    def test_handle_list_checkpoints(self, handler, mock_ui_components):
        """Test checkpoint listing handling."""
        handler._ui_components = mock_ui_components
        
        with patch.object(handler, '_run_async_operation') as mock_async:
            handler._handle_list_checkpoints()
            
            mock_async.assert_called_once()
            args, kwargs = mock_async.call_args
            config = args[1]  # Second argument is config
            assert config['action'] == 'list'
    
    def test_handle_stop_evaluation(self, handler, mock_ui_components):
        """Test evaluation stopping."""
        handler._ui_components = mock_ui_components
        handler.evaluation_active = True
        
        with patch.object(handler, 'track_status') as mock_track:
            handler._handle_stop_evaluation()
            
            assert handler.evaluation_active is False
            mock_track.assert_called_with("🛑 Evaluation stopped by user", "info")
    
    def test_handle_stop_evaluation_not_active(self, handler, mock_ui_components):
        """Test stopping when no evaluation is active."""
        handler._ui_components = mock_ui_components
        handler.evaluation_active = False
        
        with patch.object(handler, 'track_status') as mock_track:
            handler._handle_stop_evaluation()
            
            mock_track.assert_called_with(
                "⚠️ No evaluation is currently running", "warning"
            )
    
    def test_handle_save_config(self, handler, mock_ui_components):
        """Test configuration saving."""
        handler._ui_components = mock_ui_components
        
        with patch.object(handler, 'extract_config_from_ui') as mock_extract:
            with patch.object(handler, 'track_status') as mock_track:
                mock_extract.return_value = {'test': 'config'}
                
                handler._handle_save_config()
                
                mock_track.assert_called_with("💾 Configuration saved", "success")
    
    def test_handle_reset_config(self, handler, mock_ui_components):
        """Test configuration reset."""
        handler._ui_components = mock_ui_components
        
        # Modify current state
        handler.current_config = {'modified': 'config'}
        handler.selected_scenarios = {'position_variation'}
        handler.selected_models = {'cspdarknet'}
        
        with patch.object(handler, 'update_ui_from_config') as mock_update:
            with patch.object(handler, 'track_status') as mock_track:
                handler._handle_reset_config()
                
                # Verify reset to defaults
                assert handler.current_config == DEFAULT_CONFIG
                assert handler.selected_scenarios == set(AVAILABLE_SCENARIOS)
                assert handler.selected_models == set(AVAILABLE_MODELS)
                assert handler.selected_metrics == set(DEFAULT_ENABLED_METRICS)
                
                mock_update.assert_called_once_with(DEFAULT_CONFIG)
                mock_track.assert_called_with("🔄 Configuration reset to defaults", "info")
    
    def test_create_progress_callback(self, handler, mock_ui_components):
        """Test progress callback creation."""
        handler._ui_components = mock_ui_components
        mock_ui_components['progress_tracker'].update = Mock()
        
        callback = handler._create_progress_callback()
        callback(50, "Test progress")
        
        mock_ui_components['progress_tracker'].update.assert_called_once_with(50, "Test progress")
    
    def test_create_log_callback(self, handler, mock_ui_components):
        """Test log callback creation."""
        handler._ui_components = mock_ui_components
        
        with patch.object(handler, 'track_status') as mock_track:
            callback = handler._create_log_callback()
            callback("Test message", "INFO")
            
            mock_track.assert_called_once_with("Test message", "info")
    
    def test_handle_operation_result_success(self, handler, mock_ui_components):
        """Test successful operation result handling."""
        handler._ui_components = mock_ui_components
        
        result = {
            'success': True,
            'result': {
                'results': {
                    'test_scenario': {'metrics': {'map': 0.85}}
                }
            }
        }
        
        with patch.object(handler, 'track_status') as mock_track:
            with patch.object(handler, '_update_evaluation_summary') as mock_update:
                handler._handle_operation_result("Test Operation", result)
                
                mock_track.assert_called_with("✅ Test Operation completed successfully", "success")
                assert 'test_scenario' in handler.current_results
    
    def test_handle_operation_result_failure(self, handler, mock_ui_components):
        """Test failed operation result handling."""
        handler._ui_components = mock_ui_components
        
        result = {
            'success': False,
            'error': 'Test error message'
        }
        
        with patch.object(handler, 'track_status') as mock_track:
            handler._handle_operation_result("Test Operation", result)
            
            mock_track.assert_called_with("❌ Test Operation failed: Test error message", "error")
    
    def test_handle_operation_result_checkpoints(self, handler, mock_ui_components):
        """Test operation result with checkpoints."""
        handler._ui_components = mock_ui_components
        
        result = {
            'success': True,
            'result': {
                'checkpoints': [
                    {'path': 'checkpoint1.pt'},
                    {'path': 'checkpoint2.pt'}
                ]
            }
        }
        
        with patch.object(handler, 'track_status') as mock_track:
            handler._handle_operation_result("List Checkpoints", result)
            
            # Should track both success and checkpoint count
            calls = mock_track.call_args_list
            success_call = [call for call in calls if "completed successfully" in call[0][0]]
            checkpoint_call = [call for call in calls if "Found 2 checkpoints" in call[0][0]]
            
            assert len(success_call) == 1
            assert len(checkpoint_call) == 1
    
    def test_sync_config_with_ui(self, handler, mock_ui_components):
        """Test configuration sync with UI."""
        handler._ui_components = mock_ui_components
        
        with patch.object(handler, 'extract_config_from_ui') as mock_extract:
            mock_extract.return_value = {'test': 'extracted_config'}
            
            handler.sync_config_with_ui()
            
            mock_extract.assert_called_once()
            assert handler.current_config['test'] == 'extracted_config'
    
    def test_sync_ui_with_config(self, handler, mock_ui_components):
        """Test UI sync with configuration."""
        handler._ui_components = mock_ui_components
        
        with patch.object(handler, 'update_ui_from_config') as mock_update:
            handler.sync_ui_with_config()
            
            mock_update.assert_called_once_with(handler.current_config)
    
    def test_initialize_method(self, handler):
        """Test handler initialization method."""
        handler.evaluation_service = None  # Reset to test re-initialization
        
        handler.initialize()
        
        assert handler.evaluation_service is not None
        assert handler.evaluation_active is False
        assert handler.current_config == DEFAULT_CONFIG
    
    def test_get_evaluation_status(self, handler):
        """Test evaluation status retrieval."""
        # Setup test state
        handler.evaluation_active = True
        handler.selected_scenarios = {'position_variation'}
        handler.selected_models = {'cspdarknet'}
        handler.selected_metrics = {'map', 'precision'}
        handler.current_results = {'test': 'result'}
        
        with patch.object(handler.evaluation_service, 'get_current_status') as mock_status:
            mock_status.return_value = {'service': 'status'}
            
            status = handler.get_evaluation_status()
            
            assert status['evaluation_active'] is True
            assert status['selected_scenarios'] == ['position_variation']
            assert status['selected_models'] == ['cspdarknet']
            assert status['selected_metrics'] == ['map', 'precision']
            assert status['num_results'] == 1
            assert status['service_status'] == {'service': 'status'}
    
    def test_update_evaluation_summary(self, handler, mock_ui_components):
        """Test evaluation summary update."""
        handler._ui_components = mock_ui_components
        handler.selected_scenarios = {'position_variation', 'lighting_variation'}
        handler.selected_models = {'cspdarknet', 'efficientnet_b4'}
        handler.selected_metrics = {'map', 'precision', 'recall'}
        handler.evaluation_active = False
        
        config = DEFAULT_CONFIG.copy()
        
        handler._update_evaluation_summary(config)
        
        # Verify summary was updated
        summary_html = mock_ui_components['evaluation_summary'].value
        assert "🎯" in summary_html  # Icon should be present
        assert "4" in summary_html   # Total tests (2 scenarios x 2 models)
        assert "3" in summary_html   # Number of metrics
        assert "⏸️ Ready" in summary_html  # Status when not active
    
    def test_run_async_operation_threading(self, handler, mock_ui_components):
        """Test async operation runs in separate thread."""
        handler._ui_components = mock_ui_components
        
        mock_operation = Mock()
        mock_operation.execute = AsyncMock(return_value={'success': True})
        
        config = {'test': 'config'}
        
        with patch('threading.Thread') as mock_thread_class:
            mock_thread = Mock()
            mock_thread_class.return_value = mock_thread
            
            handler._run_async_operation(mock_operation, config, "Test Operation")
            
            # Verify thread was created and started
            mock_thread_class.assert_called_once()
            mock_thread.start.assert_called_once()
            
            # Verify evaluation is marked as active
            assert handler.evaluation_active is True
    
    def test_metrics_change_handling(self, handler, mock_ui_components):
        """Test metric selection change handling."""
        handler._ui_components = mock_ui_components
        
        # Simulate metrics change
        handler._on_metrics_change('map_metric_checkbox', None)
        
        # Should trigger config extraction and summary update
        pass
    
    def test_settings_change_handling(self, handler, mock_ui_components):
        """Test settings change handling."""
        handler._ui_components = mock_ui_components
        
        # Simulate settings change
        handler._on_settings_change('confidence_threshold_slider', None)
        
        # Should trigger config extraction
        pass


class TestEvaluationUIHandlerIntegration:
    """Integration tests for EvaluationUIHandler."""
    
    @pytest.fixture
    def handler_with_real_service(self):
        """Create handler with real evaluation service."""
        handler = EvaluationUIHandler()
        # Force simulation mode for testing
        handler.evaluation_service._backend_available = False
        return handler
    
    def test_end_to_end_scenario_evaluation(self, handler_with_real_service):
        """Test end-to-end scenario evaluation flow."""
        handler = handler_with_real_service
        
        # Setup selections
        handler.selected_scenarios = {'position_variation'}
        handler.selected_models = {'cspdarknet'}
        handler.selected_metrics = {'map', 'precision'}
        
        # Mock UI components
        mock_components = {'progress_tracker': Mock()}
        handler._ui_components = mock_components
        
        # Track status calls
        status_calls = []
        def track_status(message, level):
            status_calls.append((message, level))
        
        handler.track_status = track_status
        
        # Run scenario evaluation
        handler._handle_run_scenario()
        
        # Give some time for async operation (in real test would use proper async testing)
        import time
        time.sleep(0.1)
        
        # Verify operation was initiated
        assert len(status_calls) > 0
        assert any("Starting" in call[0] for call in status_calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])