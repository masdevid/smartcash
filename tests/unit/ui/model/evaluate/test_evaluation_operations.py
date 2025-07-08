"""
Tests for Evaluation Operations
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from smartcash.ui.model.evaluate.operations.scenario_evaluation_operation import ScenarioEvaluationOperation
from smartcash.ui.model.evaluate.operations.comprehensive_evaluation_operation import ComprehensiveEvaluationOperation
from smartcash.ui.model.evaluate.operations.checkpoint_operation import CheckpointOperation
from smartcash.ui.model.evaluate.services.evaluation_service import EvaluationService


class TestScenarioEvaluationOperation:
    """Test suite for ScenarioEvaluationOperation."""
    
    @pytest.fixture
    def mock_service(self):
        """Create mock evaluation service."""
        service = Mock(spec=EvaluationService)
        service.run_scenario_evaluation = AsyncMock()
        return service
    
    @pytest.fixture
    def operation(self, mock_service):
        """Create ScenarioEvaluationOperation instance."""
        return ScenarioEvaluationOperation(mock_service)
    
    @pytest.mark.asyncio
    async def test_execute_success(self, operation, mock_service):
        """Test successful scenario evaluation execution."""
        # Setup mock service response
        mock_result = {
            "success": True,
            "scenario": "position_variation",
            "model": "cspdarknet",
            "results": {
                "metrics": {"map": 0.85, "precision": 0.90},
                "grade": "A"
            }
        }
        mock_service.run_scenario_evaluation.return_value = mock_result
        
        # Setup test config
        config = {
            "scenario": "position_variation",
            "model": "cspdarknet",
            "selected_metrics": ["map", "precision"]
        }
        
        # Execute operation
        result = await operation.execute(
            config=config,
            progress_callback=Mock(),
            log_callback=Mock()
        )
        
        # Verify result
        assert result["success"] is True
        assert result["result"] == mock_result
        
        # Verify service was called correctly
        mock_service.run_scenario_evaluation.assert_called_once_with(
            scenario="position_variation",
            model="cspdarknet",
            checkpoint_path=None,
            selected_metrics=["map", "precision"]
        )
    
    @pytest.mark.asyncio
    async def test_execute_with_checkpoint(self, operation, mock_service):
        """Test scenario evaluation with specific checkpoint."""
        mock_result = {"success": True}
        mock_service.run_scenario_evaluation.return_value = mock_result
        
        config = {
            "scenario": "lighting_variation",
            "model": "efficientnet_b4",
            "checkpoint_path": "custom/checkpoint.pt"
        }
        
        result = await operation.execute(config=config)
        
        mock_service.run_scenario_evaluation.assert_called_once_with(
            scenario="lighting_variation",
            model="efficientnet_b4",
            checkpoint_path="custom/checkpoint.pt",
            selected_metrics=None
        )
    
    @pytest.mark.asyncio
    async def test_execute_failure(self, operation, mock_service):
        """Test scenario evaluation with service failure."""
        # Mock service to raise exception
        mock_service.run_scenario_evaluation.side_effect = Exception("Service error")
        
        config = {
            "scenario": "position_variation",
            "model": "cspdarknet"
        }
        
        result = await operation.execute(config=config)
        
        assert result["success"] is False
        assert "error" in result
        assert "Service error" in result["error"]
    
    @pytest.mark.asyncio
    async def test_execute_missing_required_config(self, operation, mock_service):
        """Test execution with missing required configuration."""
        # Missing scenario
        config = {"model": "cspdarknet"}
        
        result = await operation.execute(config=config)
        
        assert result["success"] is False
        assert "error" in result
        assert "scenario" in result["error"].lower()
        
        # Missing model
        config = {"scenario": "position_variation"}
        
        result = await operation.execute(config=config)
        
        assert result["success"] is False
        assert "error" in result
        assert "model" in result["error"].lower()


class TestComprehensiveEvaluationOperation:
    """Test suite for ComprehensiveEvaluationOperation."""
    
    @pytest.fixture
    def mock_service(self):
        """Create mock evaluation service."""
        service = Mock(spec=EvaluationService)
        service.run_comprehensive_evaluation = AsyncMock()
        return service
    
    @pytest.fixture
    def operation(self, mock_service):
        """Create ComprehensiveEvaluationOperation instance."""
        return ComprehensiveEvaluationOperation(mock_service)
    
    @pytest.mark.asyncio
    async def test_execute_success(self, operation, mock_service):
        """Test successful comprehensive evaluation execution."""
        mock_result = {
            "success": True,
            "total_tests": 4,
            "completed_tests": 4,
            "errors": [],
            "results": {
                "position_cspdarknet": {"success": True},
                "position_efficientnet": {"success": True},
                "lighting_cspdarknet": {"success": True},
                "lighting_efficientnet": {"success": True}
            },
            "summary": {
                "overall_metrics": {"map": 0.82},
                "best_model": "efficientnet_b4"
            }
        }
        mock_service.run_comprehensive_evaluation.return_value = mock_result
        
        config = {
            "scenarios": ["position_variation", "lighting_variation"],
            "models": ["cspdarknet", "efficientnet_b4"],
            "selected_metrics": ["map", "precision", "recall"]
        }
        
        result = await operation.execute(
            config=config,
            progress_callback=Mock(),
            log_callback=Mock()
        )
        
        assert result["success"] is True
        assert result["result"] == mock_result
        
        mock_service.run_comprehensive_evaluation.assert_called_once_with(
            scenarios=["position_variation", "lighting_variation"],
            models=["cspdarknet", "efficientnet_b4"],
            selected_metrics=["map", "precision", "recall"]
        )
    
    @pytest.mark.asyncio
    async def test_execute_with_defaults(self, operation, mock_service):
        """Test execution with default scenarios and models."""
        mock_result = {"success": True}
        mock_service.run_comprehensive_evaluation.return_value = mock_result
        
        config = {}  # Empty config, should use defaults
        
        result = await operation.execute(config=config)
        
        mock_service.run_comprehensive_evaluation.assert_called_once_with(
            scenarios=None,  # Will use service defaults
            models=None,     # Will use service defaults
            selected_metrics=None
        )
    
    @pytest.mark.asyncio
    async def test_execute_partial_success(self, operation, mock_service):
        """Test comprehensive evaluation with partial success."""
        mock_result = {
            "success": False,  # Some tests failed
            "total_tests": 2,
            "completed_tests": 2,
            "errors": ["position_variation/cspdarknet: Test error"],
            "results": {
                "position_cspdarknet": {"success": False, "error": "Test error"},
                "position_efficientnet": {"success": True}
            }
        }
        mock_service.run_comprehensive_evaluation.return_value = mock_result
        
        config = {
            "scenarios": ["position_variation"],
            "models": ["cspdarknet", "efficientnet_b4"]
        }
        
        result = await operation.execute(config=config)
        
        # Operation succeeds even if some evaluations fail
        assert result["success"] is True
        assert result["result"]["success"] is False  # But service indicates partial failure
        assert len(result["result"]["errors"]) == 1
    
    @pytest.mark.asyncio
    async def test_execute_service_exception(self, operation, mock_service):
        """Test comprehensive evaluation with service exception."""
        mock_service.run_comprehensive_evaluation.side_effect = Exception("Comprehensive evaluation failed")
        
        config = {
            "scenarios": ["position_variation"],
            "models": ["cspdarknet"]
        }
        
        result = await operation.execute(config=config)
        
        assert result["success"] is False
        assert "error" in result
        assert "Comprehensive evaluation failed" in result["error"]


class TestCheckpointOperation:
    """Test suite for CheckpointOperation."""
    
    @pytest.fixture
    def mock_service(self):
        """Create mock evaluation service."""
        service = Mock(spec=EvaluationService)
        return service
    
    @pytest.fixture
    def operation(self, mock_service):
        """Create CheckpointOperation instance."""
        return CheckpointOperation(mock_service)
    
    @pytest.mark.asyncio
    async def test_execute_list_action(self, operation, mock_service):
        """Test checkpoint listing operation."""
        with patch('smartcash.ui.model.evaluate.operations.checkpoint_operation.CheckpointSelector') as mock_selector_class:
            mock_selector = Mock()
            mock_selector_class.return_value = mock_selector
            
            mock_checkpoints = [
                {"path": "checkpoint1.pt", "epoch": 50, "map": 0.80},
                {"path": "checkpoint2.pt", "epoch": 100, "map": 0.85}
            ]
            mock_selector.find_checkpoints.return_value = mock_checkpoints
            
            config = {"action": "list"}
            
            result = await operation.execute(config=config)
            
            assert result["success"] is True
            assert "result" in result
            assert result["result"]["checkpoints"] == mock_checkpoints
            assert result["result"]["num_checkpoints"] == 2
    
    @pytest.mark.asyncio
    async def test_execute_load_action(self, operation, mock_service):
        """Test checkpoint loading operation."""
        with patch('smartcash.ui.model.evaluate.operations.checkpoint_operation.CheckpointSelector') as mock_selector_class:
            mock_selector = Mock()
            mock_selector_class.return_value = mock_selector
            
            mock_checkpoint_info = {
                "path": "best_checkpoint.pt",
                "model": "cspdarknet",
                "epoch": 100,
                "map": 0.85
            }
            
            # Test with specific checkpoint path
            mock_selector.analyze_checkpoint.return_value = mock_checkpoint_info
            
            config = {
                "action": "load",
                "model": "cspdarknet",
                "checkpoint_path": "specific_checkpoint.pt"
            }
            
            result = await operation.execute(config=config)
            
            assert result["success"] is True
            assert result["result"]["checkpoint_info"] == mock_checkpoint_info
            mock_selector.analyze_checkpoint.assert_called_once_with("specific_checkpoint.pt")
    
    @pytest.mark.asyncio
    async def test_execute_load_auto_select(self, operation, mock_service):
        """Test checkpoint loading with auto-selection."""
        with patch('smartcash.ui.model.evaluate.operations.checkpoint_operation.CheckpointSelector') as mock_selector_class:
            mock_selector = Mock()
            mock_selector_class.return_value = mock_selector
            
            mock_checkpoints = [
                {"path": "checkpoint1.pt", "map": 0.80},
                {"path": "checkpoint2.pt", "map": 0.85}
            ]
            mock_best_checkpoint = {"path": "checkpoint2.pt", "map": 0.85}
            
            mock_selector.find_checkpoints.return_value = mock_checkpoints
            mock_selector.select_best_checkpoint.return_value = mock_best_checkpoint
            
            config = {
                "action": "load",
                "model": "cspdarknet"
                # No specific checkpoint_path - should auto-select
            }
            
            result = await operation.execute(config=config)
            
            assert result["success"] is True
            assert result["result"]["checkpoint_info"] == mock_best_checkpoint
            mock_selector.find_checkpoints.assert_called_once()
            mock_selector.select_best_checkpoint.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_analyze_action(self, operation, mock_service):
        """Test checkpoint analysis operation."""
        with patch('smartcash.ui.model.evaluate.operations.checkpoint_operation.CheckpointSelector') as mock_selector_class:
            mock_selector = Mock()
            mock_selector_class.return_value = mock_selector
            
            mock_analysis = {
                "path": "checkpoint.pt",
                "model_info": {"backbone": "cspdarknet"},
                "training_info": {"epoch": 100, "best_map": 0.85},
                "file_info": {"size": "50MB", "created": "2024-01-01"}
            }
            mock_selector.analyze_checkpoint.return_value = mock_analysis
            
            config = {
                "action": "analyze",
                "checkpoint_path": "checkpoint.pt"
            }
            
            result = await operation.execute(config=config)
            
            assert result["success"] is True
            assert result["result"]["analysis"] == mock_analysis
    
    @pytest.mark.asyncio
    async def test_execute_select_best_action(self, operation, mock_service):
        """Test select best checkpoint operation."""
        with patch('smartcash.ui.model.evaluate.operations.checkpoint_operation.CheckpointSelector') as mock_selector_class:
            mock_selector = Mock()
            mock_selector_class.return_value = mock_selector
            
            mock_checkpoints = [
                {"path": "checkpoint1.pt", "map": 0.80},
                {"path": "checkpoint2.pt", "map": 0.85},
                {"path": "checkpoint3.pt", "map": 0.82}
            ]
            mock_best = {"path": "checkpoint2.pt", "map": 0.85}
            
            mock_selector.find_checkpoints.return_value = mock_checkpoints
            mock_selector.select_best_checkpoint.return_value = mock_best
            
            config = {
                "action": "select_best",
                "model": "cspdarknet",
                "metric": "map",
                "mode": "max"
            }
            
            result = await operation.execute(config=config)
            
            assert result["success"] is True
            assert result["result"]["best_checkpoint"] == mock_best
            assert result["result"]["all_checkpoints"] == mock_checkpoints
            
            mock_selector.select_best_checkpoint.assert_called_once_with(
                mock_checkpoints, metric="map", mode="max"
            )
    
    @pytest.mark.asyncio
    async def test_execute_invalid_action(self, operation, mock_service):
        """Test checkpoint operation with invalid action."""
        config = {"action": "invalid_action"}
        
        result = await operation.execute(config=config)
        
        assert result["success"] is False
        assert "error" in result
        assert "Unknown checkpoint action" in result["error"]
    
    @pytest.mark.asyncio
    async def test_execute_missing_action(self, operation, mock_service):
        """Test checkpoint operation with missing action."""
        config = {}  # No action specified
        
        result = await operation.execute(config=config)
        
        assert result["success"] is False
        assert "error" in result
        assert "action" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_execute_backend_import_error(self, operation, mock_service):
        """Test checkpoint operation when backend is not available."""
        with patch('smartcash.ui.model.evaluate.operations.checkpoint_operation.CheckpointSelector') as mock_selector_class:
            mock_selector_class.side_effect = ImportError("Backend not available")
            
            config = {"action": "list"}
            
            result = await operation.execute(config=config)
            
            assert result["success"] is False
            assert "error" in result
            assert "Backend not available" in result["error"]
    
    @pytest.mark.asyncio
    async def test_execute_no_checkpoints_found(self, operation, mock_service):
        """Test checkpoint operation when no checkpoints are found."""
        with patch('smartcash.ui.model.evaluate.operations.checkpoint_operation.CheckpointSelector') as mock_selector_class:
            mock_selector = Mock()
            mock_selector_class.return_value = mock_selector
            mock_selector.find_checkpoints.return_value = []  # No checkpoints
            
            config = {
                "action": "load",
                "model": "cspdarknet"
            }
            
            result = await operation.execute(config=config)
            
            assert result["success"] is False
            assert "error" in result
            assert "No checkpoints found" in result["error"]


class TestOperationIntegration:
    """Integration tests for operation interactions."""
    
    @pytest.fixture
    def real_service(self):
        """Create real evaluation service for integration testing."""
        service = EvaluationService()
        service._backend_available = False  # Force simulation mode
        return service
    
    @pytest.mark.asyncio
    async def test_scenario_to_comprehensive_workflow(self, real_service):
        """Test workflow from scenario to comprehensive evaluation."""
        # First run a single scenario
        scenario_op = ScenarioEvaluationOperation(real_service)
        scenario_config = {
            "scenario": "position_variation",
            "model": "cspdarknet",
            "selected_metrics": ["map", "precision"]
        }
        
        scenario_result = await scenario_op.execute(config=scenario_config)
        assert scenario_result["success"] is True
        
        # Then run comprehensive evaluation
        comprehensive_op = ComprehensiveEvaluationOperation(real_service)
        comprehensive_config = {
            "scenarios": ["position_variation", "lighting_variation"],
            "models": ["cspdarknet", "efficientnet_b4"],
            "selected_metrics": ["map", "precision", "recall"]
        }
        
        comprehensive_result = await comprehensive_op.execute(config=comprehensive_config)
        assert comprehensive_result["success"] is True
        
        # Verify comprehensive results include multiple evaluations
        results = comprehensive_result["result"]["results"]
        assert len(results) == 4  # 2 scenarios x 2 models
        
        # Check that service stores results
        service_results = real_service.get_evaluation_results()
        assert len(service_results) > 0
    
    @pytest.mark.asyncio
    async def test_operation_callback_integration(self, real_service):
        """Test that operations properly use callbacks."""
        progress_calls = []
        log_calls = []
        
        def progress_callback(progress, message):
            progress_calls.append((progress, message))
        
        def log_callback(message, level):
            log_calls.append((message, level))
        
        scenario_op = ScenarioEvaluationOperation(real_service)
        config = {
            "scenario": "position_variation",
            "model": "cspdarknet"
        }
        
        result = await scenario_op.execute(
            config=config,
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        
        assert result["success"] is True
        
        # Verify callbacks were called
        assert len(progress_calls) > 0
        assert len(log_calls) > 0
        
        # Verify progress increases
        progress_values = [call[0] for call in progress_calls]
        assert max(progress_values) == 100  # Should reach 100%
        assert min(progress_values) >= 0   # Should start from 0 or positive


if __name__ == "__main__":
    pytest.main([__file__, "-v"])