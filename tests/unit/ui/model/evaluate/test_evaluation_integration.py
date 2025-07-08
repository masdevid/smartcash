"""
Integration tests for Evaluation module
"""
import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add smartcash to path for testing
smartcash_path = Path(__file__).parent.parent.parent.parent.parent.parent / "smartcash"
sys.path.insert(0, str(smartcash_path))


class TestEvaluationModuleIntegration:
    """Integration tests for the complete evaluation module."""
    
    def test_module_imports(self):
        """Test that all evaluation module components can be imported."""
        try:
            from smartcash.ui.model.evaluate import constants
            from smartcash.ui.model.evaluate.services import evaluation_service
            from smartcash.ui.model.evaluate.handlers import evaluation_ui_handler
            from smartcash.ui.model.evaluate.operations import (
                scenario_evaluation_operation,
                comprehensive_evaluation_operation,
                checkpoint_operation
            )
            from smartcash.ui.model.evaluate.components import evaluation_ui
            from smartcash.ui.model.evaluate import evaluation_initializer
            
            assert constants.DEFAULT_CONFIG is not None
            assert evaluation_service.EvaluationService is not None
            assert evaluation_ui_handler.EvaluationUIHandler is not None
            
        except ImportError as e:
            pytest.fail(f"Failed to import evaluation module components: {e}")
    
    def test_constants_structure(self):
        """Test that constants are properly structured."""
        from smartcash.ui.model.evaluate.constants import (
            DEFAULT_CONFIG, EvaluationOperation, EvaluationPhase,
            TestScenario, BackboneModel, AVAILABLE_SCENARIOS,
            AVAILABLE_MODELS, AVAILABLE_METRICS, DEFAULT_ENABLED_METRICS
        )
        
        # Test enums
        assert EvaluationOperation.TEST_SCENARIO.value == "test_scenario"
        assert EvaluationPhase.IDLE.value == "idle"
        assert TestScenario.POSITION_VARIATION.value == "position_variation"
        assert BackboneModel.CSPDARKNET.value == "cspdarknet"
        
        # Test config structure
        assert 'evaluation' in DEFAULT_CONFIG
        assert 'inference' in DEFAULT_CONFIG
        assert 'scenarios' in DEFAULT_CONFIG['evaluation']
        assert 'metrics' in DEFAULT_CONFIG['evaluation']
        
        # Test lists
        assert 'position_variation' in AVAILABLE_SCENARIOS
        assert 'lighting_variation' in AVAILABLE_SCENARIOS
        assert 'cspdarknet' in AVAILABLE_MODELS
        assert 'efficientnet_b4' in AVAILABLE_MODELS
        assert 'map' in AVAILABLE_METRICS
        assert 'precision' in AVAILABLE_METRICS
        assert len(DEFAULT_ENABLED_METRICS) > 0
    
    def test_service_initialization(self):
        """Test evaluation service can be initialized."""
        from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService
        from smartcash.ui.model.evaluate.constants import DEFAULT_CONFIG, EvaluationPhase
        
        service = EvaluationService()
        
        assert service.config == DEFAULT_CONFIG
        assert service.current_phase == EvaluationPhase.IDLE
        assert service.evaluation_results == {}
        assert service.active_scenarios == set()
        assert service.loaded_checkpoints == {}
    
    def test_handler_initialization(self):
        """Test UI handler can be initialized."""
        from smartcash.ui.model.evaluate.handlers.evaluation_ui_handler import EvaluationUIHandler
        from smartcash.ui.model.evaluate.constants import (
            DEFAULT_CONFIG, AVAILABLE_SCENARIOS, AVAILABLE_MODELS, DEFAULT_ENABLED_METRICS
        )
        
        handler = EvaluationUIHandler()
        
        assert handler.module_name == 'evaluate'
        assert handler.parent_module == 'model'
        assert handler.current_config == DEFAULT_CONFIG
        assert handler.evaluation_active is False
        assert handler.selected_scenarios == set(AVAILABLE_SCENARIOS)
        assert handler.selected_models == set(AVAILABLE_MODELS)
        assert handler.selected_metrics == set(DEFAULT_ENABLED_METRICS)
    
    def test_operations_initialization(self):
        """Test that all operations can be initialized."""
        from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService
        from smartcash.ui.model.evaluate.operations.scenario_evaluation_operation import ScenarioEvaluationOperation
        from smartcash.ui.model.evaluate.operations.comprehensive_evaluation_operation import ComprehensiveEvaluationOperation
        from smartcash.ui.model.evaluate.operations.checkpoint_operation import CheckpointOperation
        
        service = EvaluationService()
        
        scenario_op = ScenarioEvaluationOperation(service)
        comprehensive_op = ComprehensiveEvaluationOperation(service)
        checkpoint_op = CheckpointOperation(service)
        
        assert scenario_op.service == service
        assert comprehensive_op.service == service
        assert checkpoint_op.service == service
    
    @pytest.mark.asyncio
    async def test_service_scenario_evaluation_flow(self):
        """Test complete scenario evaluation flow."""
        from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService
        
        service = EvaluationService()
        service._backend_available = False  # Force simulation mode
        
        # Setup callbacks
        progress_calls = []
        log_calls = []
        metrics_calls = []
        
        def progress_callback(progress, message):
            progress_calls.append((progress, message))
        
        def log_callback(message, level):
            log_calls.append((message, level))
            
        def metrics_callback(metrics):
            metrics_calls.append(metrics)
        
        service.set_callbacks(
            progress_callback=progress_callback,
            log_callback=log_callback,
            metrics_callback=metrics_callback
        )
        
        # Run scenario evaluation
        result = await service.run_scenario_evaluation(
            scenario="position_variation",
            model="cspdarknet",
            selected_metrics=["map", "precision", "recall"]
        )
        
        # Verify results
        assert result["success"] is True
        assert result["scenario"] == "position_variation"
        assert result["model"] == "cspdarknet"
        assert "results" in result
        assert "timestamp" in result
        
        # Verify callbacks were called
        assert len(progress_calls) > 0
        assert len(log_calls) > 0
        assert len(metrics_calls) > 0
        
        # Verify progress reached 100%
        max_progress = max(call[0] for call in progress_calls)
        assert max_progress == 100
        
        # Verify metrics were reported
        final_metrics = metrics_calls[-1]
        assert "map" in final_metrics
        assert "precision" in final_metrics
        assert "recall" in final_metrics
    
    @pytest.mark.asyncio
    async def test_service_comprehensive_evaluation_flow(self):
        """Test complete comprehensive evaluation flow."""
        from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService
        
        service = EvaluationService()
        service._backend_available = False  # Force simulation mode
        
        # Run comprehensive evaluation with limited scope for faster testing
        result = await service.run_comprehensive_evaluation(
            scenarios=["position_variation"],
            models=["cspdarknet", "efficientnet_b4"],
            selected_metrics=["map", "precision"]
        )
        
        # Verify results
        assert result["success"] is True
        assert result["total_tests"] == 2  # 1 scenario x 2 models
        assert result["completed_tests"] == 2
        assert len(result["errors"]) == 0
        assert "results" in result
        assert "summary" in result
        
        # Verify all test combinations are present
        assert "position_variation_cspdarknet" in result["results"]
        assert "position_variation_efficientnet_b4" in result["results"]
        
        # Verify summary contains expected data
        summary = result["summary"]
        assert summary["success"] is True
        assert "overall_metrics" in summary
        assert "best_model" in summary
    
    def test_handler_service_integration(self):
        """Test handler integrates properly with service."""
        from smartcash.ui.model.evaluate.handlers.evaluation_ui_handler import EvaluationUIHandler
        
        handler = EvaluationUIHandler()
        
        # Verify handler has operations with service
        assert handler.evaluation_service is not None
        assert handler.scenario_operation is not None
        assert handler.comprehensive_operation is not None
        assert handler.checkpoint_operation is not None
        
        # Verify operations reference the same service
        assert handler.scenario_operation.service == handler.evaluation_service
        assert handler.comprehensive_operation.service == handler.evaluation_service
        assert handler.checkpoint_operation.service == handler.evaluation_service
    
    def test_ui_config_extraction_integration(self):
        """Test UI configuration extraction works with real components."""
        from smartcash.ui.model.evaluate.handlers.evaluation_ui_handler import EvaluationUIHandler
        
        handler = EvaluationUIHandler()
        
        # Create minimal mock UI components
        mock_components = {
            'position_variation_checkbox': Mock(value=True),
            'lighting_variation_checkbox': Mock(value=False),
            'cspdarknet_checkbox': Mock(value=True),
            'efficientnet_checkbox': Mock(value=False),
            'confidence_threshold_slider': Mock(value=0.3),
            'iou_threshold_slider': Mock(value=0.5),
            'num_variations_slider': Mock(value=3.0),
        }
        
        # Add metric checkboxes
        from smartcash.ui.model.evaluate.constants import AVAILABLE_METRICS
        for metric in AVAILABLE_METRICS:
            mock_components[f'{metric}_metric_checkbox'] = Mock(value=metric == 'map')
        
        handler._ui_components = mock_components
        
        # Extract config
        config = handler.extract_config_from_ui()
        
        # Verify config structure and values
        assert config['evaluation']['scenarios']['position_variation']['enabled'] is True
        assert config['evaluation']['scenarios']['lighting_variation']['enabled'] is False
        assert config['inference']['confidence_threshold'] == 0.3
        assert config['inference']['iou_threshold'] == 0.5
        
        # Verify selections were updated
        assert 'position_variation' in handler.selected_scenarios
        assert 'lighting_variation' not in handler.selected_scenarios
        assert 'cspdarknet' in handler.selected_models
        assert 'efficientnet_b4' not in handler.selected_models
        assert 'map' in handler.selected_metrics
        assert len(handler.selected_metrics) == 1  # Only map selected
    
    @pytest.mark.asyncio
    async def test_operation_execution_integration(self):
        """Test operations execute properly through handler."""
        from smartcash.ui.model.evaluate.handlers.evaluation_ui_handler import EvaluationUIHandler
        
        handler = EvaluationUIHandler()
        handler.evaluation_service._backend_available = False  # Force simulation mode
        
        # Setup for scenario evaluation
        handler.selected_scenarios = {'position_variation'}
        handler.selected_models = {'cspdarknet'}
        handler.selected_metrics = {'map', 'precision'}
        
        # Mock UI components
        handler._ui_components = {
            'progress_tracker': Mock(),
        }
        
        # Create callbacks
        progress_calls = []
        log_calls = []
        
        def mock_progress(progress, message):
            progress_calls.append((progress, message))
        
        def mock_log(message, level):
            log_calls.append((message, level))
        
        # Execute scenario operation directly
        config = {
            "scenario": "position_variation",
            "model": "cspdarknet",
            "selected_metrics": ["map", "precision"]
        }
        
        result = await handler.scenario_operation.execute(
            config=config,
            progress_callback=mock_progress,
            log_callback=mock_log
        )
        
        # Verify operation completed successfully
        assert result["success"] is True
        assert "result" in result
        
        # Verify callbacks were called (through service)
        assert len(progress_calls) > 0
        assert len(log_calls) > 0
    
    def test_module_file_structure(self):
        """Test that module follows expected file structure."""
        module_path = Path(__file__).parent.parent.parent.parent.parent.parent / "smartcash" / "ui" / "model" / "evaluate"
        
        # Check required files exist
        required_files = [
            "__init__.py",
            "constants.py",
            "evaluation_initializer.py",
            "services/__init__.py",
            "services/evaluation_service.py",
            "handlers/__init__.py",
            "handlers/evaluation_ui_handler.py",
            "operations/__init__.py",
            "operations/scenario_evaluation_operation.py",
            "operations/comprehensive_evaluation_operation.py",
            "operations/checkpoint_operation.py",
            "components/__init__.py",
            "components/evaluation_ui.py",
            "configs/__init__.py",
            "configs/evaluation_defaults.py"
        ]
        
        for file_path in required_files:
            full_path = module_path / file_path
            assert full_path.exists(), f"Required file missing: {file_path}"
    
    def test_error_handling_integration(self):
        """Test error handling works across module components."""
        from smartcash.ui.model.evaluate.handlers.evaluation_ui_handler import EvaluationUIHandler
        
        handler = EvaluationUIHandler()
        
        # Test handling when no scenarios/models selected
        handler.selected_scenarios = set()
        handler.selected_models = set()
        
        status_calls = []
        handler.track_status = lambda msg, level: status_calls.append((msg, level))
        
        # Try to run scenario evaluation
        handler._handle_run_scenario()
        
        # Should get error status
        assert len(status_calls) > 0
        error_call = status_calls[-1]
        assert error_call[1] == "error"
        assert "select at least one" in error_call[0].lower()
    
    def test_configuration_persistence_integration(self):
        """Test configuration can be saved and loaded through handler."""
        from smartcash.ui.model.evaluate.handlers.evaluation_ui_handler import EvaluationUIHandler
        from smartcash.ui.model.evaluate.constants import DEFAULT_CONFIG
        
        handler = EvaluationUIHandler()
        
        # Modify configuration
        test_config = DEFAULT_CONFIG.copy()
        test_config['inference']['confidence_threshold'] = 0.8
        test_config['evaluation']['selected_metrics'] = ['map', 'precision']
        
        # Update handler config
        handler.current_config = test_config
        handler.selected_metrics = {'map', 'precision'}
        
        # Create mock UI components that reflect the config
        mock_components = {
            'confidence_threshold_slider': Mock(value=0.8),
            'map_metric_checkbox': Mock(value=True),
            'precision_metric_checkbox': Mock(value=True),
            'recall_metric_checkbox': Mock(value=False),
            'f1_score_metric_checkbox': Mock(value=False),
            'accuracy_metric_checkbox': Mock(value=False),
            'inference_time_metric_checkbox': Mock(value=False),
            'evaluation_summary': Mock()
        }
        handler._ui_components = mock_components
        
        # Update UI from config
        handler.update_ui_from_config(test_config)
        
        # Verify UI was updated
        assert mock_components['confidence_threshold_slider'].value == 0.8
        assert mock_components['map_metric_checkbox'].value is True
        assert mock_components['precision_metric_checkbox'].value is True
        assert mock_components['recall_metric_checkbox'].value is False
    
    def test_backend_availability_detection(self):
        """Test backend availability detection works properly."""
        from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService
        
        service = EvaluationService()
        
        # Check backend availability (will depend on actual environment)
        backend_available = service._check_backend_availability()
        
        # Should return boolean
        assert isinstance(backend_available, bool)
        
        # Check that cached result is used
        second_check = service._check_backend_availability()
        assert second_check == backend_available
        
        # Verify service status includes backend availability
        status = service.get_current_status()
        assert 'backend_available' in status
        assert status['backend_available'] == backend_available


class TestEvaluationModuleWorkflow:
    """Test complete evaluation workflow scenarios."""
    
    @pytest.mark.asyncio
    async def test_single_scenario_workflow(self):
        """Test complete single scenario evaluation workflow."""
        from smartcash.ui.model.evaluate.handlers.evaluation_ui_handler import EvaluationUIHandler
        
        # Initialize handler
        handler = EvaluationUIHandler()
        handler.evaluation_service._backend_available = False
        
        # Setup UI selections
        handler.selected_scenarios = {'position_variation'}
        handler.selected_models = {'cspdarknet'}
        handler.selected_metrics = {'map', 'precision', 'recall'}
        
        # Mock status tracking
        status_history = []
        handler.track_status = lambda msg, level: status_history.append((msg, level))
        
        # Execute scenario evaluation operation directly
        config = {
            "scenario": "position_variation",
            "model": "cspdarknet", 
            "selected_metrics": ["map", "precision", "recall"]
        }
        
        result = await handler.scenario_operation.execute(config=config)
        
        # Verify workflow completed successfully
        assert result["success"] is True
        
        # Verify result structure
        evaluation_result = result["result"]
        assert evaluation_result["scenario"] == "position_variation"
        assert evaluation_result["model"] == "cspdarknet"
        assert "results" in evaluation_result
        assert "timestamp" in evaluation_result
        
        # Verify metrics are present
        metrics = evaluation_result["results"]["metrics"]
        assert "map" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        
        # Verify performance grade was assigned
        assert "grade" in evaluation_result["results"]
        assert evaluation_result["results"]["grade"] in ["A+", "A", "B", "C", "D"]
    
    @pytest.mark.asyncio
    async def test_comprehensive_evaluation_workflow(self):
        """Test complete comprehensive evaluation workflow."""
        from smartcash.ui.model.evaluate.handlers.evaluation_ui_handler import EvaluationUIHandler
        
        # Initialize handler
        handler = EvaluationUIHandler()
        handler.evaluation_service._backend_available = False
        
        # Setup for comprehensive evaluation (limited scope for testing)
        handler.selected_scenarios = {'position_variation', 'lighting_variation'}
        handler.selected_models = {'cspdarknet'}
        handler.selected_metrics = {'map', 'precision'}
        
        # Execute comprehensive evaluation
        config = {
            "scenarios": ["position_variation", "lighting_variation"],
            "models": ["cspdarknet"],
            "selected_metrics": ["map", "precision"]
        }
        
        result = await handler.comprehensive_operation.execute(config=config)
        
        # Verify workflow completed successfully
        assert result["success"] is True
        
        # Verify result structure
        comprehensive_result = result["result"]
        assert comprehensive_result["total_tests"] == 2
        assert comprehensive_result["completed_tests"] == 2
        assert len(comprehensive_result["errors"]) == 0
        
        # Verify all test combinations are present
        results = comprehensive_result["results"]
        assert len(results) == 2
        assert any("position_variation" in key for key in results.keys())
        assert any("lighting_variation" in key for key in results.keys())
        
        # Verify summary was generated
        summary = comprehensive_result["summary"]
        assert summary["success"] is True
        assert "overall_metrics" in summary
        assert "best_scenario" in summary
        assert "best_model" in summary
    
    @pytest.mark.asyncio
    async def test_checkpoint_management_workflow(self):
        """Test checkpoint management workflow."""
        from smartcash.ui.model.evaluate.handlers.evaluation_ui_handler import EvaluationUIHandler
        
        # Initialize handler
        handler = EvaluationUIHandler()
        
        # Test checkpoint listing
        with patch('smartcash.ui.model.evaluate.operations.checkpoint_operation.CheckpointSelector') as mock_selector_class:
            mock_selector = Mock()
            mock_selector_class.return_value = mock_selector
            
            mock_checkpoints = [
                {"path": "checkpoint1.pt", "epoch": 50, "map": 0.80},
                {"path": "checkpoint2.pt", "epoch": 100, "map": 0.85}
            ]
            mock_selector.find_checkpoints.return_value = mock_checkpoints
            
            # List checkpoints
            list_config = {"action": "list"}
            list_result = await handler.checkpoint_operation.execute(config=list_config)
            
            assert list_result["success"] is True
            assert list_result["result"]["num_checkpoints"] == 2
            assert list_result["result"]["checkpoints"] == mock_checkpoints
            
            # Load specific checkpoint
            mock_checkpoint_info = {
                "path": "checkpoint2.pt",
                "model": "cspdarknet",
                "epoch": 100,
                "map": 0.85
            }
            mock_selector.analyze_checkpoint.return_value = mock_checkpoint_info
            
            load_config = {
                "action": "load",
                "model": "cspdarknet",
                "checkpoint_path": "checkpoint2.pt"
            }
            load_result = await handler.checkpoint_operation.execute(config=load_config)
            
            assert load_result["success"] is True
            assert load_result["result"]["checkpoint_info"] == mock_checkpoint_info
    
    def test_error_recovery_workflow(self):
        """Test error recovery in evaluation workflow."""
        from smartcash.ui.model.evaluate.handlers.evaluation_ui_handler import EvaluationUIHandler
        
        handler = EvaluationUIHandler()
        
        # Test handling of missing selections
        handler.selected_scenarios = set()
        handler.selected_models = set()
        
        status_messages = []
        handler.track_status = lambda msg, level: status_messages.append((msg, level))
        
        # Try to run evaluation without selections
        handler._handle_run_scenario()
        
        # Should get error message
        error_messages = [msg for msg, level in status_messages if level == "error"]
        assert len(error_messages) > 0
        assert any("select at least one" in msg.lower() for msg in error_messages)
        
        # Test handling of evaluation already running
        handler.evaluation_active = True
        handler.selected_scenarios = {'position_variation'}
        handler.selected_models = {'cspdarknet'}
        
        status_messages.clear()
        handler._handle_run_scenario()
        
        # Should get warning message
        warning_messages = [msg for msg, level in status_messages if level == "warning"]
        assert len(warning_messages) > 0
        assert any("already in progress" in msg.lower() for msg in warning_messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])