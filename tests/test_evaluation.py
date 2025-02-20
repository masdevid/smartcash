"""Test suite for evaluation handlers and metrics."""

import pytest
import torch
import numpy as np
from pathlib import Path

from smartcash.utils.metrics import MetricsCalculator
from smartcash.handlers.evaluation_handler import EvaluationHandler
from smartcash.handlers.base_evaluation_handler import BaseEvaluationHandler
from smartcash.handlers.research_scenario_handler import ResearchScenarioHandler

class TestEvaluation:
    @pytest.fixture
    def config_path(self):
        """Fixture for config path."""
        return Path(__file__).parent.parent / 'configs' / 'base_config.yaml'
    
    @pytest.fixture
    def sample_predictions(self):
        """Fixture for sample model predictions."""
        predictions = torch.tensor([
            # [x, y, width, height, confidence, class]
            [0.5, 0.5, 0.2, 0.2, 0.9, 1],  # class 1, high confidence
            [0.3, 0.7, 0.1, 0.1, 0.7, 2],  # class 2, medium confidence
            [0.8, 0.2, 0.15, 0.15, 0.6, 1]  # class 1, low confidence
        ])
        return predictions
    
    @pytest.fixture
    def sample_targets(self):
        """Fixture for ground truth targets."""
        targets = torch.tensor([
            [0.5, 0.5, 0.2, 0.2, 1],  # class 1
            [0.3, 0.7, 0.1, 0.1, 2]   # class 2
        ])
        return targets
    
    @pytest.fixture
    def config(self, config_path):
        """Fixture for evaluation config."""
        return {
            'output_dir': 'tests/output',
            'checkpoints_dir': 'tests/checkpoints',
            'data_dir': 'tests/data',
            'metrics': {
                'mAP': True,
                'precision': True,
                'recall': True,
                'f1_score': True,
                'confusion_matrix': True
            }
        }
    
    def test_metrics_calculator(self, sample_predictions, sample_targets):
        """Test metrics calculator functionality."""
        metrics_calc = MetricsCalculator()
        
        # Update with predictions and targets
        metrics_calc.update(
            predictions=sample_predictions.unsqueeze(0),  # Add batch dimension
            targets=sample_targets.unsqueeze(0)
        )
        
        # Calculate final metrics
        final_metrics = metrics_calc.compute()
        
        # Check key metrics
        expected_metrics = [
            'precision', 'recall', 'accuracy', 
            'f1', 'mAP', 'inference_time'
        ]
        
        for metric in expected_metrics:
            assert metric in final_metrics, f"Metric {metric} not found"
            assert isinstance(final_metrics[metric], float), f"Metric {metric} should be float"
        
        # Check metric ranges
        assert 0 <= final_metrics['precision'] <= 1, "Precision should be between 0 and 1"
        assert 0 <= final_metrics['recall'] <= 1, "Recall should be between 0 and 1"
        assert 0 <= final_metrics['accuracy'] <= 1, "Accuracy should be between 0 and 1"
        assert 0 <= final_metrics['f1'] <= 1, "F1-score should be between 0 and 1"
    
    def test_base_evaluation_handler(self, config, sample_predictions, sample_targets):
        """Test base evaluation handler functionality."""
        handler = BaseEvaluationHandler(config=config)
        
        # Create mock test loader
        class MockDataLoader:
            def __iter__(self):
                return iter([(sample_predictions.unsqueeze(0), sample_targets.unsqueeze(0))])
        
        # Test evaluation
        results = handler.evaluate_model(
            model_path='tests/checkpoints/mock_model.pth',
            dataset='test'
        )
        
        assert isinstance(results, dict), "Results should be a dictionary"
        assert 'Akurasi' in results, "Results should contain accuracy"
        assert 'Precision' in results, "Results should contain precision"
        assert 'Recall' in results, "Results should contain recall"
        assert 'F1-Score' in results, "Results should contain F1-score"
        assert 'mAP' in results, "Results should contain mAP"
    
    def test_research_scenario_handler(self, config):
        """Test research scenario handler functionality."""
        handler = ResearchScenarioHandler(config=config)
        
        # Test single scenario evaluation
        scenario_results = handler.evaluate_scenario(
            scenario_name="Test Scenario",
            model_path='tests/checkpoints/mock_model.pth',
            test_data_path='tests/data/test_scenario'
        )
        
        assert isinstance(scenario_results, dict), "Scenario results should be a dictionary"
        assert 'Akurasi' in scenario_results, "Results should contain accuracy"
        assert 'Precision' in scenario_results, "Results should contain precision"
        assert 'Waktu Inferensi' in scenario_results, "Results should contain inference time"
    
    def test_evaluation_handler_regular(self, config):
        """Test main evaluation handler with regular evaluation."""
        handler = EvaluationHandler(config=config)
        
        # Test regular evaluation
        results = handler.evaluate(eval_type='regular')
        
        assert isinstance(results, dict), "Regular evaluation results should be a dictionary"
        assert len(results) > 0, "Should have results for at least one model"
        
        for model_name, metrics in results.items():
            assert isinstance(metrics, dict), f"Metrics for {model_name} should be a dictionary"
            assert 'Akurasi' in metrics, f"Results for {model_name} should contain accuracy"
    
    def test_evaluation_handler_research(self, config):
        """Test main evaluation handler with research evaluation."""
        handler = EvaluationHandler(config=config)
        
        # Test research evaluation
        results = handler.evaluate(eval_type='research')
        
        assert isinstance(results, dict), "Research evaluation results should be a dictionary"
        assert 'research_results' in results, "Should contain research_results key"
        assert len(results['research_results']) > 0, "Should have at least one scenario result"
    
    def test_invalid_evaluation_type(self, config):
        """Test handling of invalid evaluation type."""
        handler = EvaluationHandler(config=config)
        
        with pytest.raises(ValueError, match="Unknown evaluation type"):
            handler.evaluate(eval_type='invalid')