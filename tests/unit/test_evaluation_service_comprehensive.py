import unittest
from unittest.mock import MagicMock, patch, call
from smartcash.model.evaluation.evaluation_service import EvaluationService
from smartcash.model.evaluation.utils.inference_timer import InferenceTimer

class TestEvaluationServiceComprehensive(unittest.TestCase):

    @patch('smartcash.model.evaluation.evaluation_service.InferenceService')
    @patch('smartcash.model.evaluation.evaluation_service.ScenarioManager')
    @patch('smartcash.model.evaluation.evaluation_service.CheckpointSelector')
    @patch('smartcash.model.evaluation.evaluation_service.ResultsAggregator')
    @patch('smartcash.model.evaluation.evaluation_service.EvaluationChartGenerator')
    @patch('smartcash.model.evaluation.evaluators.scenario_evaluator.create_scenario_evaluator')
    @patch('smartcash.model.evaluation.utils.evaluation_progress_bridge.EvaluationProgressBridge')
    @patch('smartcash.model.evaluation.processors.data_loader.EvaluationDataLoader')
    def setUp(self, MockInferenceService, MockScenarioManager, MockCheckpointSelector, MockResultsAggregator, MockChartGenerator, MockCreateScenarioEvaluator, MockProgressBridge, MockEvaluationDataLoader):
        self.mock_inference_service = MockInferenceService.return_value
        self.mock_scenario_manager = MockScenarioManager.return_value
        self.mock_checkpoint_selector = MockCheckpointSelector.return_value
        self.mock_results_aggregator = MockResultsAggregator.return_value
        self.mock_chart_generator = MockChartGenerator.return_value
        self.mock_create_scenario_evaluator = MockCreateScenarioEvaluator
        self.mock_progress_bridge = MockProgressBridge.return_value
        self.mock_evaluation_data_loader = MockEvaluationDataLoader.return_value
        self.mock_evaluation_data_loader.load_scenario_data.return_value = {
            'images': [MagicMock()],
            'labels': [MagicMock()]
        }

        self.config = {
            'evaluation': {
                'export': {'include_visualizations': True},
                'data': {'charts_dir': 'data/evaluation/charts'}
            }
        }
        self.service = EvaluationService(self.config)
        
        # Mock internal components initialized in _initialize_components
        self.service.scenario_manager = self.mock_scenario_manager
        self.service.checkpoint_selector = self.mock_checkpoint_selector
        self.service.results_aggregator = self.mock_results_aggregator
        self.service.chart_generator = self.mock_chart_generator
        self.service.progress_bridge = self.mock_progress_bridge
        
        # Mock _load_checkpoint to return a valid checkpoint_info
        self.service._load_checkpoint = MagicMock(return_value={
            'path': 'dummy_checkpoint.pt',
            'model_loaded': True,
            'optimized_run': False,
            'display_name': 'Dummy Checkpoint',
            'backbone': 'yolov5s'
        })

    def test_run_evaluation_success(self):
        # Arrange
        scenarios = ['position_variation', 'lighting_variation']
        checkpoints = ['checkpoint1.pt', 'checkpoint2.pt']

        self.mock_checkpoint_selector.list_available_checkpoints.return_value = [
            {'path': 'checkpoint1.pt'},
            {'path': 'checkpoint2.pt'}
        ]
        
        # Mock the scenario evaluator to return some metrics
        self.mock_create_scenario_evaluator.return_value.evaluate_scenario.side_effect = [
            {'metrics': {'mAP': 0.85}, 'additional_data': {}},
            {'metrics': {'mAP': 0.88}, 'additional_data': {}}
        ]

        # Act
        result = self.service.run_evaluation(scenarios=scenarios, checkpoints=checkpoints)

        # Assert
        self.assertTrue(result['status'] == 'success')
        self.assertEqual(result['scenarios_evaluated'], 2)
        self.assertEqual(result['checkpoints_evaluated'], 2)
        
        # Verify calls to load_checkpoint
        expected_load_calls = [
            call('checkpoint1.pt'),
            call('checkpoint2.pt')
        ]
        self.service._load_checkpoint.assert_has_calls(expected_load_calls, any_order=True)

        # Verify calls to evaluate_scenario
        expected_eval_calls = [
            call(scenario_name='position_variation', checkpoint_info=self.service._load_checkpoint.return_value, inference_service=self.mock_inference_service, progress_callback=self.mock_progress_bridge.update_metrics),
            call(scenario_name='lighting_variation', checkpoint_info=self.service._load_checkpoint.return_value, inference_service=self.mock_inference_service, progress_callback=self.mock_progress_bridge.update_metrics)
        ]
        # Note: The order of scenario_name might vary depending on internal iteration, so we check individual calls
        self.mock_create_scenario_evaluator.return_value.evaluate_scenario.assert_has_calls(expected_eval_calls, any_order=True)

        # Verify aggregator and chart generator calls
        self.mock_results_aggregator.add_scenario_results.assert_called()
        self.mock_results_aggregator.export_results.assert_called_once()
        self.mock_chart_generator.generate_all_charts.assert_called_once()
        self.mock_progress_bridge.start_evaluation.assert_called_once()
        self.mock_progress_bridge.complete_evaluation.assert_called_once()

    def test_run_evaluation_error_handling(self):
        # Arrange
        scenarios = ['position_variation']
        checkpoints = ['checkpoint1.pt']

        self.mock_checkpoint_selector.list_available_checkpoints.return_value = [
            {'path': 'checkpoint1.pt'}
        ]
        
        # Simulate an error during scenario evaluation
        self.mock_create_scenario_evaluator.return_value.evaluate_scenario.side_effect = Exception("Test Error")

        # Act
        self.mock_progress_bridge.evaluation_error.reset_mock()
        result = self.service.run_evaluation(scenarios=scenarios, checkpoints=checkpoints)

        # Assert
        self.assertTrue(result['status'] == 'error')
        self.assertIn("No test data found for position_variation", result['error'])
        self.mock_progress_bridge.evaluation_error.assert_called_once()
        self.mock_inference_service.cleanup.assert_called_once() # Ensure cleanup is called on error

    def test_run_scenario_success(self):
        # Arrange
        scenario_name = 'single_scenario'
        checkpoint_path = 'single_checkpoint.pt'
        
        self.service._load_checkpoint.return_value = {
            'path': checkpoint_path,
            'model_loaded': True,
            'optimized_run': False,
            'display_name': 'Single Checkpoint',
            'backbone': 'yolov5s'
        }
        self.mock_create_scenario_evaluator.return_value.evaluate_scenario.return_value = {
            'metrics': {'mAP': 0.90}, 'additional_data': {}}

        # Act
        result = self.service.run_scenario(scenario_name=scenario_name, checkpoint_path=checkpoint_path)

        # Assert
        self.assertTrue(result['status'] == 'success')
        self.assertEqual(result['scenario_name'], scenario_name)
        self.assertEqual(result['metrics']['mAP'], 0.90)
        self.service._load_checkpoint.assert_called_once_with(checkpoint_path)
        self.mock_create_scenario_evaluator.return_value.evaluate_scenario.assert_called_once_with(
            scenario_name=scenario_name,
            checkpoint_info=self.service._load_checkpoint.return_value,
            inference_service=self.mock_inference_service,
            progress_callback=self.mock_progress_bridge.update_metrics
        )
        self.mock_chart_generator.generate_all_charts.assert_called_once()

    def test_cleanup_on_exit(self):
        # Arrange
        # The cleanup is registered via atexit, so we need to simulate program exit
        # We can directly call _cleanup for testing purposes
        
        # Act
        self.mock_inference_service.cleanup.reset_mock()
        self.service._cleanup()

        # Assert
        self.mock_inference_service.cleanup.assert_called_once()

if __name__ == '__main__':
    unittest.main()
