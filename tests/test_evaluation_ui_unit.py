#!/usr/bin/env python3
"""
Unit tests for evaluation UI components.
"""

import sys
import os
import pytest
import unittest
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, '/Users/masdevid/Projects/smartcash')

class TestEvaluationUIComponents(unittest.TestCase):
    """Unit tests for evaluation UI components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            'evaluation': {
                'execution': {
                    'run_mode': 'all_scenarios',
                    'parallel_execution': False,
                    'save_intermediate_results': True
                },
                'models': {
                    'auto_select_best': True
                }
            }
        }
    
    def test_constants_import(self):
        """Test that all constants can be imported."""
        from smartcash.ui.model.evaluation.constants import (
            UI_CONFIG, EVALUATION_METRICS, MODEL_COMBINATIONS, RESEARCH_SCENARIOS
        )
        
        # Check that all required constants exist
        self.assertIsInstance(UI_CONFIG, dict)
        self.assertIsInstance(EVALUATION_METRICS, dict)
        self.assertIsInstance(MODEL_COMBINATIONS, list)
        self.assertIsInstance(RESEARCH_SCENARIOS, dict)
        
        # Check accuracy metric exists
        self.assertIn('accuracy', EVALUATION_METRICS)
        
        # Check we have correct number of scenarios and models
        self.assertEqual(len(RESEARCH_SCENARIOS), 2)
        self.assertEqual(len(MODEL_COMBINATIONS), 4)
        
    def test_ui_module_creation(self):
        """Test evaluation UI module creation."""
        from smartcash.ui.model.evaluation.evaluation_uimodule import EvaluationUIModule
        
        # Should create without errors
        ui_module = EvaluationUIModule()
        self.assertIsNotNone(ui_module)
        
    def test_ui_module_initialization(self):
        """Test evaluation UI module initialization."""
        from smartcash.ui.model.evaluation.evaluation_uimodule import EvaluationUIModule
        
        ui_module = EvaluationUIModule()
        
        # Should initialize with test config
        ui_module.initialize(config=self.test_config)
        
        # Check that UI components were created
        ui_components = ui_module.get_ui_components()
        self.assertIsInstance(ui_components, dict)
        self.assertGreater(len(ui_components), 0)
        
    def test_operation_manager_creation(self):
        """Test operation manager creation."""
        from smartcash.ui.model.evaluation.evaluation_uimodule import EvaluationUIModule
        
        ui_module = EvaluationUIModule()
        ui_module.initialize(config=self.test_config)
        
        # Check operation manager
        operation_manager = ui_module.get_operation_manager()
        self.assertIsNotNone(operation_manager)
        
        # Check available operations
        operations = operation_manager.get_operations()
        self.assertIsInstance(operations, dict)
        self.assertIn('all_scenarios', operations)
        
    def test_form_value_extraction(self):
        """Test form value extraction."""
        from smartcash.ui.model.evaluation.evaluation_uimodule import EvaluationUIModule
        
        ui_module = EvaluationUIModule()
        ui_module.initialize(config=self.test_config)
        
        # Extract form values
        form_values = ui_module._extract_form_values()
        
        self.assertIsInstance(form_values, dict)
        self.assertIn('run_mode', form_values)
        
    def test_ui_component_creation(self):
        """Test UI component creation."""
        from smartcash.ui.model.evaluation.components.evaluation_ui import create_evaluation_ui
        
        ui_components = create_evaluation_ui(self.test_config)
        
        self.assertIsInstance(ui_components, dict)
        self.assertIn('main_container', ui_components)
        self.assertIn('action_container', ui_components)
        self.assertIn('operation_container', ui_components)
        
    def test_compact_form_layout(self):
        """Test compact form layout creation."""
        from smartcash.ui.model.evaluation.components.evaluation_ui import _create_execution_model_row, _create_metrics_form_section
        
        # Test execution model row (2-column layout)
        execution_model_row = _create_execution_model_row(self.test_config)
        self.assertIsNotNone(execution_model_row)
        
        # Test metrics section
        metrics_section = _create_metrics_form_section(self.test_config)
        self.assertIsNotNone(metrics_section)
        
    def test_backend_integration(self):
        """Test backend service integration."""
        from smartcash.ui.model.evaluation.operations.evaluation_operation_manager import EvaluationOperationManager
        
        # Create operation manager
        operation_manager = EvaluationOperationManager(
            config=self.test_config,
            operation_container=None
        )
        operation_manager.initialize()
        
        # Check backend service exists
        self.assertIsNotNone(operation_manager.evaluation_service)
        
    def test_logging_functionality(self):
        """Test logging functionality."""
        from smartcash.ui.model.evaluation.evaluation_uimodule import EvaluationUIModule
        
        ui_module = EvaluationUIModule()
        ui_module.initialize(config=self.test_config)
        
        # Test logging (should not raise exceptions)
        try:
            ui_module.log("Test message", 'info')
            ui_module.log("Test success", 'success')
            ui_module.log("Test warning", 'warning')
            ui_module.log("Test error", 'error')
        except Exception as e:
            self.fail(f"Logging failed: {e}")

class TestEvaluationIntegration(unittest.TestCase):
    """Integration tests for evaluation UI."""
    
    def test_end_to_end_ui_creation(self):
        """Test complete UI creation from start to finish."""
        from smartcash.ui.model.evaluation.evaluation_uimodule import initialize_evaluation_ui
        
        # Create UI without display
        ui_module = initialize_evaluation_ui(display=False)
        self.assertIsNotNone(ui_module)
        
        # Get UI components
        ui_components = ui_module.get_ui_components()
        self.assertIsInstance(ui_components, dict)
        
        # Check all expected components exist
        expected_components = [
            'main_container', 'header_container', 'execution_model_row',
            'metrics_section', 'action_container', 'operation_container',
            'summary_container'
        ]
        
        for component in expected_components:
            self.assertIn(component, ui_components, f"Missing component: {component}")
        
        # Check main container is displayable
        main_container = ui_components['main_container']
        self.assertIsNotNone(main_container)
        
        # Check action container has correct button
        action_container = ui_components['action_container']
        self.assertIsInstance(action_container, dict)
        self.assertIn('buttons', action_container)
        self.assertIn('run_scenario', action_container['buttons'])
        
    def test_configuration_flow(self):
        """Test configuration flow through the system."""
        test_config = {
            'evaluation': {
                'execution': {
                    'run_mode': 'position_only',
                    'parallel_execution': True,
                    'save_intermediate_results': False
                }
            }
        }
        
        from smartcash.ui.model.evaluation.evaluation_uimodule import EvaluationUIModule
        
        ui_module = EvaluationUIModule()
        ui_module.initialize(config=test_config)
        
        # Check config was applied
        module_config = ui_module.get_config()
        self.assertIsInstance(module_config, dict)
        
        # Extract form values and verify config is used
        form_values = ui_module._extract_form_values()
        self.assertEqual(form_values.get('run_mode'), 'position_only')
        self.assertEqual(form_values.get('parallel_execution'), True)
        self.assertEqual(form_values.get('save_intermediate_results'), False)

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)