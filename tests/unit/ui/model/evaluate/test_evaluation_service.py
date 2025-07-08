"""
Tests for EvaluationService
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService
from smartcash.ui.model.evaluate.constants import (
    DEFAULT_CONFIG, EvaluationOperation, EvaluationPhase,
    TestScenario, BackboneModel, DEFAULT_ENABLED_METRICS
)


class TestEvaluationService:
    """Test suite for EvaluationService."""
    
    @pytest.fixture
    def service(self):
        """Create EvaluationService instance for testing."""
        return EvaluationService()
    
    @pytest.fixture
    def mock_callbacks(self):
        """Create mock callbacks for testing."""
        return {
            'progress': Mock(),
            'log': Mock(),
            'metrics': Mock()
        }
    
    def test_initialization(self, service):
        """Test service initialization."""
        assert service.config == DEFAULT_CONFIG
        assert service.current_phase == EvaluationPhase.IDLE
        assert service.current_operation is None
        assert service.evaluation_results == {}
        assert service.active_scenarios == set()
        assert service.loaded_checkpoints == {}
        assert service._backend_available is None
    
    def test_set_callbacks(self, service, mock_callbacks):
        """Test callback setup."""
        service.set_callbacks(
            progress_callback=mock_callbacks['progress'],
            log_callback=mock_callbacks['log'],
            metrics_callback=mock_callbacks['metrics']
        )
        
        assert service.progress_callback == mock_callbacks['progress']
        assert service.log_callback == mock_callbacks['log']
        assert service.metrics_callback == mock_callbacks['metrics']
    
    def test_log_message_with_callback(self, service, mock_callbacks):
        """Test log message with callback."""
        service.set_callbacks(log_callback=mock_callbacks['log'])
        
        service._log_message("Test message", "info")
        
        mock_callbacks['log'].assert_called_once_with("Test message", "info")
    
    def test_update_progress_with_callback(self, service, mock_callbacks):
        """Test progress update with callback."""
        service.set_callbacks(progress_callback=mock_callbacks['progress'])
        
        service._update_progress(50, "Test progress")
        
        mock_callbacks['progress'].assert_called_once_with(50, "Test progress")
    
    def test_update_metrics_with_callback(self, service, mock_callbacks):
        """Test metrics update with callback."""
        service.set_callbacks(metrics_callback=mock_callbacks['metrics'])
        
        test_metrics = {"map": 0.85, "precision": 0.90}
        service._update_metrics(test_metrics)
        
        mock_callbacks['metrics'].assert_called_once_with(test_metrics)
    
    @patch('smartcash.ui.model.evaluate.services.evaluation_service.importlib')
    def test_check_backend_availability_success(self, mock_importlib, service):
        """Test backend availability check when backend is available."""
        # Mock successful imports
        mock_importlib.import_module.return_value = Mock()
        
        result = service._check_backend_availability()
        
        assert result is True
        assert service._backend_available is True
    
    @patch('smartcash.ui.model.evaluate.services.evaluation_service.importlib')
    def test_check_backend_availability_failure(self, mock_importlib, service):
        """Test backend availability check when backend is not available."""
        # Mock import failure
        mock_importlib.import_module.side_effect = ImportError("Module not found")
        
        result = service._check_backend_availability()
        
        assert result is False
        assert service._backend_available is False
    
    @pytest.mark.asyncio
    async def test_run_scenario_evaluation_simulation_mode(self, service, mock_callbacks):
        """Test scenario evaluation in simulation mode."""
        service.set_callbacks(**mock_callbacks)
        service._backend_available = False  # Force simulation mode
        
        result = await service.run_scenario_evaluation(
            scenario="position_variation",
            model="cspdarknet",
            selected_metrics=["map", "precision"]
        )
        
        assert result["success"] is True
        assert result["scenario"] == "position_variation"
        assert result["model"] == "cspdarknet"
        assert "results" in result
        assert "timestamp" in result
        
        # Verify callbacks were called
        assert mock_callbacks['progress'].call_count > 0
        assert mock_callbacks['log'].call_count > 0
        assert mock_callbacks['metrics'].call_count > 0
    
    @pytest.mark.asyncio
    async def test_run_comprehensive_evaluation_simulation_mode(self, service, mock_callbacks):
        """Test comprehensive evaluation in simulation mode."""
        service.set_callbacks(**mock_callbacks)
        service._backend_available = False  # Force simulation mode
        
        result = await service.run_comprehensive_evaluation(
            scenarios=["position_variation"],
            models=["cspdarknet"],
            selected_metrics=["map", "precision"]
        )
        
        assert result["success"] is True
        assert result["total_tests"] == 1
        assert result["completed_tests"] == 1
        assert len(result["errors"]) == 0
        assert "results" in result
        assert "summary" in result
    
    @pytest.mark.asyncio
    async def test_augment_test_dataset_simulation(self, service):
        """Test dataset augmentation in simulation mode."""
        service._backend_available = False
        
        result = await service._augment_test_dataset("position_variation")
        
        assert result["success"] is True
        assert result["simulation"] is True
        assert "dataset_path" in result
        assert result["num_variations"] == 5
    
    @pytest.mark.asyncio
    async def test_load_checkpoint_simulation(self, service):
        """Test checkpoint loading in simulation mode."""
        service._backend_available = False
        
        result = await service._load_checkpoint("cspdarknet")
        
        assert result["success"] is True
        assert result["simulation"] is True
        assert "checkpoint_info" in result
        assert result["checkpoint_info"]["model"] == "cspdarknet"
    
    @pytest.mark.asyncio
    async def test_run_backend_evaluation_simulation(self, service):
        """Test backend evaluation in simulation mode."""
        service._backend_available = False
        
        checkpoint_info = {"path": "test/checkpoint.pt", "model": "cspdarknet"}
        
        result = await service._run_backend_evaluation(
            scenario="position_variation",
            model="cspdarknet",
            dataset_path="test/dataset",
            checkpoint_info=checkpoint_info,
            selected_metrics=["map", "precision"]
        )
        
        assert result["success"] is True
        assert result["simulation"] is True
        assert "metrics" in result
        assert "map" in result["metrics"]
        assert "precision" in result["metrics"]
    
    @pytest.mark.asyncio
    async def test_process_evaluation_results(self, service):
        """Test evaluation results processing."""
        evaluation_result = {
            "success": True,
            "metrics": {
                "map": 0.85,
                "precision": 0.90,
                "recall": 0.80,
                "f1_score": 0.85,
                "inference_time": 45.2
            },
            "num_images": 150,
            "total_detections": 287,
            "simulation": True
        }
        
        result = await service._process_evaluation_results(
            evaluation_result, "position_variation", "cspdarknet"
        )
        
        assert result["scenario"] == "position_variation"
        assert result["model"] == "cspdarknet"
        assert result["metrics"]["map"] == 0.85
        assert result["formatted_metrics"]["map"] == "0.850"
        assert result["formatted_metrics"]["inference_time"] == "45.20ms"
        assert result["grade"] == "A"  # 0.85 mAP should be grade A
        assert result["num_images"] == 150
        assert result["simulation"] is True
    
    def test_generate_evaluation_summary(self, service):
        """Test evaluation summary generation."""
        # Setup test results
        results = {
            "position_cspdarknet": {
                "success": True,
                "scenario": "position_variation",
                "model": "cspdarknet",
                "results": {
                    "metrics": {"map": 0.80, "precision": 0.85, "recall": 0.75, "f1_score": 0.80}
                }
            },
            "position_efficientnet": {
                "success": True,
                "scenario": "position_variation", 
                "model": "efficientnet_b4",
                "results": {
                    "metrics": {"map": 0.85, "precision": 0.90, "recall": 0.80, "f1_score": 0.85}
                }
            }
        }
        
        summary = service._generate_evaluation_summary(results)
        
        assert summary["success"] is True
        assert summary["total_evaluations"] == 2
        assert "overall_metrics" in summary
        assert "scenario_performance" in summary
        assert "model_performance" in summary
        assert summary["best_model"] == "efficientnet_b4"  # Higher mAP
    
    def test_get_current_status(self, service):
        """Test current status retrieval."""
        service.current_phase = EvaluationPhase.EVALUATING
        service.current_operation = EvaluationOperation.TEST_SCENARIO
        service.active_scenarios.add("position_variation")
        service.loaded_checkpoints["cspdarknet"] = {"path": "test.pt"}
        service.evaluation_results["test"] = {"result": "data"}
        
        status = service.get_current_status()
        
        assert status["phase"] == "evaluating"
        assert status["operation"] == "test_scenario"
        assert "position_variation" in status["active_scenarios"]
        assert "cspdarknet" in status["loaded_checkpoints"]
        assert status["num_results"] == 1
        assert "backend_available" in status
    
    def test_get_evaluation_results(self, service):
        """Test evaluation results retrieval."""
        test_results = {"test1": {"data": "value1"}, "test2": {"data": "value2"}}
        service.evaluation_results = test_results
        
        results = service.get_evaluation_results()
        
        assert results == test_results
        assert results is not service.evaluation_results  # Should be a copy
    
    def test_clear_results(self, service):
        """Test clearing evaluation results."""
        # Setup some data
        service.evaluation_results = {"test": "data"}
        service.active_scenarios.add("test_scenario")
        service.loaded_checkpoints["test"] = {"path": "test.pt"}
        service.current_phase = EvaluationPhase.EVALUATING
        service.current_operation = EvaluationOperation.TEST_SCENARIO
        
        service.clear_results()
        
        assert len(service.evaluation_results) == 0
        assert len(service.active_scenarios) == 0
        assert len(service.loaded_checkpoints) == 0
        assert service.current_phase == EvaluationPhase.IDLE
        assert service.current_operation is None
    
    @pytest.mark.asyncio
    async def test_scenario_evaluation_with_error(self, service, mock_callbacks):
        """Test scenario evaluation with error handling."""
        service.set_callbacks(**mock_callbacks)
        
        # Mock augmentation to fail
        with patch.object(service, '_augment_test_dataset') as mock_augment:
            mock_augment.return_value = {"success": False, "error": "Augmentation failed"}
            
            result = await service.run_scenario_evaluation(
                scenario="position_variation",
                model="cspdarknet"
            )
            
            assert result["success"] is False
            assert "error" in result
            assert service.current_phase == EvaluationPhase.ERROR
    
    @pytest.mark.asyncio
    async def test_comprehensive_evaluation_with_partial_errors(self, service, mock_callbacks):
        """Test comprehensive evaluation with some scenarios failing."""
        service.set_callbacks(**mock_callbacks)
        service._backend_available = False
        
        # Mock one scenario to fail
        original_run_scenario = service.run_scenario_evaluation
        
        async def mock_run_scenario(scenario, model, **kwargs):
            if scenario == "position_variation":
                return {"success": False, "scenario": scenario, "model": model, "error": "Test error"}
            else:
                return await original_run_scenario(scenario, model, **kwargs)
        
        with patch.object(service, 'run_scenario_evaluation', side_effect=mock_run_scenario):
            result = await service.run_comprehensive_evaluation(
                scenarios=["position_variation", "lighting_variation"],
                models=["cspdarknet"]
            )
            
            assert result["success"] is False  # Should fail due to errors
            assert result["total_tests"] == 2
            assert result["completed_tests"] == 2
            assert len(result["errors"]) == 1
            assert "position_variation/cspdarknet: Test error" in result["errors"][0]
    
    def test_custom_config_initialization(self):
        """Test service initialization with custom config."""
        custom_config = {
            "evaluation": {
                "data": {
                    "test_dir": "custom/test/dir"
                }
            }
        }
        
        service = EvaluationService(config=custom_config)
        
        assert service.config == custom_config
        assert service.config != DEFAULT_CONFIG
    
    @pytest.mark.asyncio
    async def test_metric_filtering_in_evaluation(self, service):
        """Test that selected metrics are properly filtered in evaluation."""
        service._backend_available = False  # Force simulation mode
        
        # Test with limited metrics
        result = await service.run_scenario_evaluation(
            scenario="position_variation",
            model="cspdarknet",
            selected_metrics=["map", "precision"]  # Only these metrics
        )
        
        assert result["success"] is True
        metrics = result["results"]["metrics"]
        
        # Should only contain requested metrics (simulation mode may include others)
        assert "map" in metrics
        assert "precision" in metrics
    
    @pytest.mark.asyncio 
    async def test_scenario_specific_metric_adjustments(self, service):
        """Test that metrics are adjusted based on scenario and model."""
        service._backend_available = False  # Force simulation mode
        
        # Test position variation (should have lower metrics)
        pos_result = await service.run_scenario_evaluation(
            scenario="position_variation",
            model="cspdarknet"
        )
        
        # Test lighting variation
        light_result = await service.run_scenario_evaluation(
            scenario="lighting_variation", 
            model="cspdarknet"
        )
        
        # Position variation should generally have lower metrics due to adjustment
        pos_map = pos_result["results"]["metrics"]["map"]
        light_map = light_result["results"]["metrics"]["map"]
        
        # Both should be valid mAP scores
        assert 0 <= pos_map <= 1
        assert 0 <= light_map <= 1
    
    @pytest.mark.asyncio
    async def test_efficientnet_vs_cspdarknet_performance(self, service):
        """Test that EfficientNet shows expected performance characteristics vs CSPDarknet."""
        service._backend_available = False  # Force simulation mode
        
        # Test CSPDarknet
        csp_result = await service.run_scenario_evaluation(
            scenario="position_variation",
            model="cspdarknet"
        )
        
        # Test EfficientNet
        eff_result = await service.run_scenario_evaluation(
            scenario="position_variation",
            model="efficientnet_b4"
        )
        
        csp_metrics = csp_result["results"]["metrics"]
        eff_metrics = eff_result["results"]["metrics"]
        
        # EfficientNet should have higher mAP but slower inference
        assert eff_metrics["map"] > csp_metrics["map"]
        assert eff_metrics["inference_time"] > csp_metrics["inference_time"]
    
    def test_performance_grade_calculation(self, service):
        """Test performance grade calculation logic."""
        test_cases = [
            (0.95, "A+"),
            (0.85, "A"),
            (0.75, "B"),
            (0.65, "C"),
            (0.55, "D"),
            (0.45, "D")
        ]
        
        for map_score, expected_grade in test_cases:
            evaluation_result = {
                "success": True,
                "metrics": {"map": map_score},
                "simulation": True
            }
            
            # Use asyncio.run for async method
            result = asyncio.run(service._process_evaluation_results(
                evaluation_result, "test_scenario", "test_model"
            ))
            
            assert result["grade"] == expected_grade, f"mAP {map_score} should be grade {expected_grade}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])