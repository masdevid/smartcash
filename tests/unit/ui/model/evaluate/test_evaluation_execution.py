"""
Execution tests for evaluation module - validates that the module executes without errors
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add smartcash to path for testing
smartcash_path = Path(__file__).parent.parent.parent.parent.parent.parent / "smartcash"
sys.path.insert(0, str(smartcash_path))


class TestEvaluationModuleExecution:
    """Test that evaluation module can be executed without errors."""
    
    def test_evaluation_initializer_execution(self):
        """Test that evaluation initializer can be executed."""
        with patch('IPython.display.display') as mock_display:
            with patch('ipywidgets.VBox') as mock_vbox:
                with patch('ipywidgets.HTML') as mock_html:
                    mock_vbox.return_value = Mock()
                    mock_html.return_value = Mock()
                    
                    try:
                        from smartcash.ui.model.evaluate.evaluation_initializer import init_evaluation_ui
                        
                        # Execute the initializer
                        result = init_evaluation_ui()
                        
                        # Should not raise an exception and should display something
                        mock_display.assert_called()
                        
                    except Exception as e:
                        pytest.fail(f"Evaluation initializer execution failed: {e}")
    
    def test_evaluation_service_instantiation(self):
        """Test that evaluation service can be instantiated."""
        try:
            from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService
            
            service = EvaluationService()
            
            # Basic functionality test
            status = service.get_current_status()
            assert isinstance(status, dict)
            assert 'phase' in status
            assert 'backend_available' in status
            
        except Exception as e:
            pytest.fail(f"Evaluation service instantiation failed: {e}")
    
    def test_evaluation_ui_handler_instantiation(self):
        """Test that evaluation UI handler can be instantiated."""
        try:
            from smartcash.ui.model.evaluate.handlers.evaluation_ui_handler import EvaluationUIHandler
            
            handler = EvaluationUIHandler()
            
            # Test initialization
            handler.initialize()
            
            # Test status retrieval
            status = handler.get_evaluation_status()
            assert isinstance(status, dict)
            assert 'evaluation_active' in status
            assert 'selected_scenarios' in status
            assert 'selected_models' in status
            
        except Exception as e:
            pytest.fail(f"Evaluation UI handler instantiation failed: {e}")
    
    def test_evaluation_operations_instantiation(self):
        """Test that evaluation operations can be instantiated."""
        try:
            from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService
            from smartcash.ui.model.evaluate.operations.scenario_evaluation_operation import ScenarioEvaluationOperation
            from smartcash.ui.model.evaluate.operations.comprehensive_evaluation_operation import ComprehensiveEvaluationOperation
            from smartcash.ui.model.evaluate.operations.checkpoint_operation import CheckpointOperation
            
            service = EvaluationService()
            
            # Test operation instantiation
            scenario_op = ScenarioEvaluationOperation(service)
            comprehensive_op = ComprehensiveEvaluationOperation(service)
            checkpoint_op = CheckpointOperation(service)
            
            # Verify operations are properly linked to service
            assert scenario_op.service == service
            assert comprehensive_op.service == service
            assert checkpoint_op.service == service
            
        except Exception as e:
            pytest.fail(f"Evaluation operations instantiation failed: {e}")
    
    def test_evaluation_ui_components_creation(self):
        """Test that evaluation UI components can be created."""
        with patch('ipywidgets.VBox') as mock_vbox:
            with patch('ipywidgets.HBox') as mock_hbox:
                with patch('ipywidgets.HTML') as mock_html:
                    with patch('ipywidgets.Checkbox') as mock_checkbox:
                        with patch('ipywidgets.FloatSlider') as mock_slider:
                            with patch('ipywidgets.Button') as mock_button:
                                with patch('ipywidgets.Tab') as mock_tab:
                                    # Mock all widget returns
                                    mock_vbox.return_value = Mock()
                                    mock_hbox.return_value = Mock()
                                    mock_html.return_value = Mock()
                                    mock_checkbox.return_value = Mock()
                                    mock_slider.return_value = Mock()
                                    mock_button.return_value = Mock()
                                    mock_tab.return_value = Mock()
                                    
                                    try:
                                        from smartcash.ui.model.evaluate.components.evaluation_ui import create_evaluation_ui
                                        
                                        # Create UI components
                                        ui_components = create_evaluation_ui()
                                        
                                        # Should return a dictionary of components
                                        assert isinstance(ui_components, dict)
                                        assert len(ui_components) > 0
                                        
                                        # Check for key components
                                        expected_components = [
                                            'main_container',
                                            'evaluation_summary',
                                            'run_scenario_btn',
                                            'run_comprehensive_btn'
                                        ]
                                        
                                        for component in expected_components:
                                            assert component in ui_components, f"Missing component: {component}"
                                        
                                    except Exception as e:
                                        pytest.fail(f"Evaluation UI components creation failed: {e}")
    
    @pytest.mark.asyncio
    async def test_evaluation_service_simulation_execution(self):
        """Test that evaluation service can execute operations in simulation mode."""
        try:
            from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService
            
            service = EvaluationService()
            service._backend_available = False  # Force simulation mode
            
            # Test scenario evaluation
            result = await service.run_scenario_evaluation(
                scenario="position_variation",
                model="cspdarknet",
                selected_metrics=["map", "precision"]
            )
            
            assert result["success"] is True
            assert result["scenario"] == "position_variation"
            assert result["model"] == "cspdarknet"
            assert "results" in result
            
            # Test comprehensive evaluation
            comp_result = await service.run_comprehensive_evaluation(
                scenarios=["position_variation"],
                models=["cspdarknet"],
                selected_metrics=["map"]
            )
            
            assert comp_result["success"] is True
            assert comp_result["total_tests"] == 1
            assert "results" in comp_result
            assert "summary" in comp_result
            
        except Exception as e:
            pytest.fail(f"Evaluation service simulation execution failed: {e}")
    
    @pytest.mark.asyncio
    async def test_evaluation_operations_execution(self):
        """Test that evaluation operations can execute without errors."""
        try:
            from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService
            from smartcash.ui.model.evaluate.operations.scenario_evaluation_operation import ScenarioEvaluationOperation
            from smartcash.ui.model.evaluate.operations.comprehensive_evaluation_operation import ComprehensiveEvaluationOperation
            
            service = EvaluationService()
            service._backend_available = False  # Force simulation mode
            
            # Test scenario operation
            scenario_op = ScenarioEvaluationOperation(service)
            scenario_config = {
                "scenario": "position_variation",
                "model": "cspdarknet",
                "selected_metrics": ["map"]
            }
            
            scenario_result = await scenario_op.execute(config=scenario_config)
            
            assert scenario_result["success"] is True
            assert "result" in scenario_result
            
            # Test comprehensive operation
            comprehensive_op = ComprehensiveEvaluationOperation(service)
            comprehensive_config = {
                "scenarios": ["position_variation"],
                "models": ["cspdarknet"],
                "selected_metrics": ["map"]
            }
            
            comprehensive_result = await comprehensive_op.execute(config=comprehensive_config)
            
            assert comprehensive_result["success"] is True
            assert "result" in comprehensive_result
            
        except Exception as e:
            pytest.fail(f"Evaluation operations execution failed: {e}")
    
    def test_evaluation_handler_ui_integration(self):
        """Test that evaluation handler can integrate with UI components."""
        with patch('ipywidgets.VBox') as mock_vbox:
            with patch('ipywidgets.HTML') as mock_html:
                with patch('ipywidgets.Checkbox') as mock_checkbox:
                    with patch('ipywidgets.FloatSlider') as mock_slider:
                        with patch('ipywidgets.Button') as mock_button:
                            # Mock widget returns
                            mock_widget = Mock()
                            mock_widget.value = True
                            mock_widget.observe = Mock()
                            mock_widget.on_click = Mock()
                            
                            mock_vbox.return_value = mock_widget
                            mock_html.return_value = mock_widget
                            mock_checkbox.return_value = mock_widget
                            mock_slider.return_value = mock_widget
                            mock_button.return_value = mock_widget
                            
                            try:
                                from smartcash.ui.model.evaluate.handlers.evaluation_ui_handler import EvaluationUIHandler
                                from smartcash.ui.model.evaluate.components.evaluation_ui import create_evaluation_ui
                                
                                # Create handler and UI components
                                handler = EvaluationUIHandler()
                                ui_components = create_evaluation_ui()
                                
                                # Setup handler with UI components
                                handler.setup(ui_components)
                                
                                # Test configuration extraction and updating
                                config = handler.extract_config_from_ui()
                                assert isinstance(config, dict)
                                
                                handler.update_ui_from_config(config)
                                
                                # Test status methods
                                status = handler.get_evaluation_status()
                                assert isinstance(status, dict)
                                
                            except Exception as e:
                                pytest.fail(f"Evaluation handler UI integration failed: {e}")
    
    def test_evaluation_constants_accessibility(self):
        """Test that all evaluation constants are accessible."""
        try:
            from smartcash.ui.model.evaluate.constants import (
                DEFAULT_CONFIG,
                EvaluationOperation,
                EvaluationPhase,
                TestScenario,
                BackboneModel,
                AVAILABLE_SCENARIOS,
                AVAILABLE_MODELS,
                AVAILABLE_METRICS,
                DEFAULT_ENABLED_METRICS,
                SCENARIO_CONFIGS,
                MODEL_CONFIGS,
                METRIC_CONFIGS,
                OPERATION_MESSAGES,
                UI_CONFIG
            )
            
            # Test that all constants are properly defined
            assert DEFAULT_CONFIG is not None
            assert isinstance(DEFAULT_CONFIG, dict)
            
            # Test enums
            assert EvaluationOperation.TEST_SCENARIO.value == "test_scenario"
            assert EvaluationPhase.IDLE.value == "idle"
            assert TestScenario.POSITION_VARIATION.value == "position_variation"
            assert BackboneModel.CSPDARKNET.value == "cspdarknet"
            
            # Test lists
            assert isinstance(AVAILABLE_SCENARIOS, list)
            assert isinstance(AVAILABLE_MODELS, list)
            assert isinstance(AVAILABLE_METRICS, list)
            assert isinstance(DEFAULT_ENABLED_METRICS, list)
            
            # Test configs
            assert isinstance(SCENARIO_CONFIGS, dict)
            assert isinstance(MODEL_CONFIGS, dict)
            assert isinstance(METRIC_CONFIGS, dict)
            assert isinstance(OPERATION_MESSAGES, dict)
            assert isinstance(UI_CONFIG, dict)
            
        except Exception as e:
            pytest.fail(f"Evaluation constants accessibility failed: {e}")
    
    def test_evaluation_error_handling(self):
        """Test that evaluation module handles errors gracefully."""
        try:
            from smartcash.ui.model.evaluate.handlers.evaluation_ui_handler import EvaluationUIHandler
            
            handler = EvaluationUIHandler()
            
            # Test error handling when no selections made
            handler.selected_scenarios = set()
            handler.selected_models = set()
            
            status_calls = []
            handler.track_status = lambda msg, level: status_calls.append((msg, level))
            
            # Should handle error gracefully
            handler._handle_run_scenario()
            
            # Should have logged error status
            error_statuses = [call for call in status_calls if call[1] == "error"]
            assert len(error_statuses) > 0
            
        except Exception as e:
            pytest.fail(f"Evaluation error handling test failed: {e}")
    
    def test_evaluation_config_validation(self):
        """Test that evaluation configuration validation works."""
        try:
            from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService
            from smartcash.ui.model.evaluate.constants import DEFAULT_CONFIG
            
            # Test with default config
            service1 = EvaluationService()
            assert service1.config == DEFAULT_CONFIG
            
            # Test with custom config
            custom_config = {
                "evaluation": {
                    "data": {
                        "test_dir": "custom/test/dir"
                    }
                }
            }
            
            service2 = EvaluationService(config=custom_config)
            assert service2.config == custom_config
            assert service2.config != DEFAULT_CONFIG
            
        except Exception as e:
            pytest.fail(f"Evaluation config validation failed: {e}")
    
    def test_evaluation_module_imports_comprehensive(self):
        """Comprehensive test of all module imports."""
        try:
            # Test main module components
            from smartcash.ui.model.evaluate import constants
            from smartcash.ui.model.evaluate import evaluation_initializer
            
            # Test services
            from smartcash.ui.model.evaluate.services import evaluation_service
            
            # Test handlers
            from smartcash.ui.model.evaluate.handlers import evaluation_ui_handler
            
            # Test operations
            from smartcash.ui.model.evaluate.operations import scenario_evaluation_operation
            from smartcash.ui.model.evaluate.operations import comprehensive_evaluation_operation
            from smartcash.ui.model.evaluate.operations import checkpoint_operation
            
            # Test components
            from smartcash.ui.model.evaluate.components import evaluation_ui
            
            # Test configs
            from smartcash.ui.model.evaluate.configs import evaluation_defaults
            
            # Verify key classes are accessible
            assert hasattr(evaluation_service, 'EvaluationService')
            assert hasattr(evaluation_ui_handler, 'EvaluationUIHandler')
            assert hasattr(scenario_evaluation_operation, 'ScenarioEvaluationOperation')
            assert hasattr(comprehensive_evaluation_operation, 'ComprehensiveEvaluationOperation')
            assert hasattr(checkpoint_operation, 'CheckpointOperation')
            assert hasattr(evaluation_ui, 'create_evaluation_ui')
            assert hasattr(evaluation_initializer, 'init_evaluation_ui')
            
        except Exception as e:
            pytest.fail(f"Comprehensive module imports failed: {e}")


class TestEvaluationModuleStress:
    """Stress tests for evaluation module."""
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_evaluations(self):
        """Test multiple evaluations can run concurrently in simulation mode."""
        try:
            from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService
            import asyncio
            
            # Create multiple services
            services = [EvaluationService() for _ in range(3)]
            for service in services:
                service._backend_available = False  # Force simulation mode
            
            # Run concurrent evaluations
            tasks = []
            for i, service in enumerate(services):
                task = service.run_scenario_evaluation(
                    scenario="position_variation",
                    model="cspdarknet", 
                    selected_metrics=["map"]
                )
                tasks.append(task)
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks)
            
            # Verify all succeeded
            for result in results:
                assert result["success"] is True
                assert "results" in result
                
        except Exception as e:
            pytest.fail(f"Multiple concurrent evaluations test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_comprehensive_evaluation_large_matrix(self):
        """Test comprehensive evaluation with full test matrix."""
        try:
            from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService
            
            service = EvaluationService()
            service._backend_available = False  # Force simulation mode
            
            # Full test matrix: 2 scenarios x 2 models = 4 tests
            result = await service.run_comprehensive_evaluation(
                scenarios=["position_variation", "lighting_variation"],
                models=["cspdarknet", "efficientnet_b4"],
                selected_metrics=["map", "precision", "recall", "f1_score"]
            )
            
            assert result["success"] is True
            assert result["total_tests"] == 4
            assert result["completed_tests"] == 4
            assert len(result["errors"]) == 0
            
            # Verify all combinations are present
            results = result["results"]
            assert len(results) == 4
            
            expected_keys = [
                "position_variation_cspdarknet",
                "position_variation_efficientnet_b4", 
                "lighting_variation_cspdarknet",
                "lighting_variation_efficientnet_b4"
            ]
            
            for key in expected_keys:
                assert key in results
                assert results[key]["success"] is True
                
        except Exception as e:
            pytest.fail(f"Large matrix comprehensive evaluation test failed: {e}")
    
    def test_handler_setup_teardown_multiple_times(self):
        """Test handler can be set up and torn down multiple times."""
        try:
            from smartcash.ui.model.evaluate.handlers.evaluation_ui_handler import EvaluationUIHandler
            
            # Create and setup handler multiple times
            for i in range(5):
                handler = EvaluationUIHandler()
                
                # Mock UI components
                mock_components = {
                    'position_variation_checkbox': Mock(value=True),
                    'cspdarknet_checkbox': Mock(value=True),
                    'evaluation_summary': Mock(),
                }
                
                # Add mock observe and on_click methods
                for component in mock_components.values():
                    component.observe = Mock()
                    component.on_click = Mock()
                
                # Setup and initialize
                handler.setup(mock_components)
                handler.initialize()
                
                # Test basic functionality
                status = handler.get_evaluation_status()
                assert isinstance(status, dict)
                
                # Clear references
                handler._ui_components = {}
                handler.evaluation_service = None
                
        except Exception as e:
            pytest.fail(f"Multiple handler setup/teardown test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])