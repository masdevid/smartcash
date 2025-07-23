"""
Integration test for training module button state management.

This test verifies that the training module's enhanced button handler 
integration works correctly with the BaseUIModule pattern.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any


class MockButton:
    def __init__(self, button_id: str):
        self.button_id = button_id
        self.disabled = False
        self.description = f"Mock {button_id} button"
        self._on_click_handlers = []
    
    def on_click(self, handler):
        self._on_click_handlers.append(handler)


class MockUIComponents:
    def __init__(self):
        self.action_container = {
            'buttons': {
                'start_training': MockButton('start_training'),
                'stop_training': MockButton('stop_training'),
                'resume_training': MockButton('resume_training'),
                'validate_model': MockButton('validate_model'),
                'refresh_backbone_config': MockButton('refresh_backbone_config'),
                'save': MockButton('save'),
                'reset': MockButton('reset')
            }
        }
    
    def get(self, key):
        if key == 'action_container':
            return self.action_container
        return None


class TestTrainingButtonStates(unittest.TestCase):
    """Test training module button state management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock UI components
        self.mock_ui_components = MockUIComponents()
        
    def tearDown(self):
        """Clean up test fixtures."""
        pass
    
    def test_training_button_dependency_setup(self):
        """Test that training button dependencies are set up correctly."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        # Create module instance
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Verify dependencies were set
        self.assertTrue(hasattr(module, '_button_dependencies'))
        expected_buttons = ['start_training', 'resume_training', 'stop_training', 'validate_model']
        
        for button in expected_buttons:
            self.assertIn(button, module._button_dependencies)
    
    def test_start_training_dependency_with_model(self):
        """Test start training button dependency when model is available."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Set state: has model, not training
        module._has_model = True
        module._is_training_active = False
        
        # Test dependency check
        result = module._check_start_training_dependency()
        self.assertTrue(result)
    
    def test_start_training_dependency_without_model(self):
        """Test start training button dependency when no model is available."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Set state: no model
        module._has_model = False
        module._is_training_active = False
        
        # Test dependency check
        result = module._check_start_training_dependency()
        self.assertFalse(result)
    
    def test_start_training_dependency_while_training_active(self):
        """Test start training button dependency when training is already active."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Set state: has model but training is active
        module._has_model = True
        module._is_training_active = True
        
        # Test dependency check
        result = module._check_start_training_dependency()
        self.assertFalse(result)
    
    def test_resume_training_dependency_with_checkpoint(self):
        """Test resume training button dependency when checkpoint is available."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Set state: has checkpoint, not training
        module._has_checkpoint = True
        module._is_training_active = False
        
        # Test dependency check
        result = module._check_resume_training_dependency()
        self.assertTrue(result)
    
    def test_resume_training_dependency_without_checkpoint(self):
        """Test resume training button dependency when no checkpoint is available."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Set state: no checkpoint
        module._has_checkpoint = False
        module._is_training_active = False
        
        # Test dependency check
        result = module._check_resume_training_dependency()
        self.assertFalse(result)
    
    def test_stop_training_dependency_while_training_active(self):
        """Test stop training button dependency when training is active."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Set state: training is active
        module._is_training_active = True
        
        # Test dependency check
        result = module._check_stop_training_dependency()
        self.assertTrue(result)
    
    def test_stop_training_dependency_while_training_inactive(self):
        """Test stop training button dependency when training is not active."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Set state: training is not active
        module._is_training_active = False
        
        # Test dependency check
        result = module._check_stop_training_dependency()
        self.assertFalse(result)
    
    def test_validate_model_dependency_with_model(self):
        """Test validate model button dependency when model is available."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Set state: has model
        module._has_model = True
        module._has_checkpoint = False
        
        # Test dependency check
        result = module._check_validate_model_dependency()
        self.assertTrue(result)
    
    def test_validate_model_dependency_with_checkpoint(self):
        """Test validate model button dependency when checkpoint is available."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Set state: has checkpoint
        module._has_model = False
        module._has_checkpoint = True
        
        # Test dependency check
        result = module._check_validate_model_dependency()
        self.assertTrue(result)
    
    def test_validate_model_dependency_without_model_or_checkpoint(self):
        """Test validate model button dependency when neither model nor checkpoint is available."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Set state: no model, no checkpoint
        module._has_model = False
        module._has_checkpoint = False
        
        # Test dependency check
        result = module._check_validate_model_dependency()
        self.assertFalse(result)
    
    def test_training_state_update_button_refresh(self):
        """Test that button states are updated when training state changes."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Mock the button state update methods
        module.update_button_states_based_on_condition = Mock()
        
        # Update training state
        module._update_training_state(
            has_model=True,
            has_checkpoint=False,
            is_training_active=False
        )
        
        # Verify button states were updated
        module.update_button_states_based_on_condition.assert_called_once()
        call_args = module.update_button_states_based_on_condition.call_args
        
        # Check button conditions
        button_conditions = call_args[0][0]
        
        # start_training should be enabled (has model, not training)
        self.assertTrue(button_conditions['start_training'])
        
        # resume_training should be disabled (no checkpoint)
        self.assertFalse(button_conditions['resume_training'])
        
        # stop_training should be disabled (not training)
        self.assertFalse(button_conditions['stop_training'])
        
        # validate_model should be enabled (has model)
        self.assertTrue(button_conditions['validate_model'])
    
    def test_training_state_during_active_training(self):
        """Test button states during active training."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Mock the button state update methods
        module.update_button_states_based_on_condition = Mock()
        
        # Update training state to active
        module._update_training_state(
            has_model=True,
            has_checkpoint=True,
            is_training_active=True
        )
        
        # Verify button states were updated
        module.update_button_states_based_on_condition.assert_called_once()
        call_args = module.update_button_states_based_on_condition.call_args
        
        # Check button conditions during active training
        button_conditions = call_args[0][0]
        
        # start_training should be disabled (training active)
        self.assertFalse(button_conditions['start_training'])
        
        # resume_training should be disabled (training active)
        self.assertFalse(button_conditions['resume_training'])
        
        # stop_training should be enabled (training active)
        self.assertTrue(button_conditions['stop_training'])
        
        # validate_model should be enabled (has model and checkpoint)
        self.assertTrue(button_conditions['validate_model'])
    
    def test_disable_reasons_generation(self):
        """Test that proper disable reasons are generated for buttons."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Set state: no model, no checkpoint, not training
        module._has_model = False
        module._has_checkpoint = False
        module._is_training_active = False
        
        # Test disable reasons
        start_reason = module._get_start_training_disable_reason()
        self.assertEqual(start_reason, "No model selected - configure backbone first")
        
        resume_reason = module._get_resume_training_disable_reason()
        self.assertEqual(resume_reason, "No checkpoint available to resume from")
    
    def test_disable_reasons_during_training(self):
        """Test disable reasons when training is active."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Set state: has model and checkpoint, training active
        module._has_model = True
        module._has_checkpoint = True
        module._is_training_active = True
        
        # Test disable reasons during training
        start_reason = module._get_start_training_disable_reason()
        self.assertEqual(start_reason, "Training already in progress")
        
        resume_reason = module._get_resume_training_disable_reason()
        self.assertEqual(resume_reason, "Training already in progress")


class TestTrainingButtonHandlers(unittest.TestCase):
    """Test training module button handler integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ui_components = MockUIComponents()
    
    def test_button_handler_registration(self):
        """Test that button handlers are registered correctly."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Get module button handlers
        handlers = module._get_module_button_handlers()
        
        expected_handlers = [
            'start_training', 'stop_training', 'resume_training',
            'refresh_backbone_config', 'save', 'reset'
        ]
        
        for handler_name in expected_handlers:
            self.assertIn(handler_name, handlers)
            self.assertTrue(callable(handlers[handler_name]))
    
    def test_start_training_handler_state_management(self):
        """Test that start training handler manages state correctly."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Mock the execute method and logging
        module.execute_start_training = Mock(return_value={'success': True, 'message': 'Training started'})
        module.log_info = Mock()
        module.log_success = Mock()
        
        # Initially not training
        module._is_training_active = False
        
        # Call handler
        module._handle_start_training()
        
        # Verify state was updated to training active
        self.assertTrue(module._is_training_active)
        
        # Verify execute was called
        module.execute_start_training.assert_called_once()
        
        # Verify logging
        module.log_info.assert_called_with("Starting training...")
        module.log_success.assert_called_with("Training started: Training started")
    
    def test_stop_training_handler_state_management(self):
        """Test that stop training handler manages state correctly."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Mock the execute method and logging
        module.execute_stop_training = Mock(return_value={'success': True, 'message': 'Training stopped'})
        module.log_info = Mock()
        module.log_success = Mock()
        
        # Initially training
        module._is_training_active = True
        
        # Call handler
        module._handle_stop_training()
        
        # Verify state was updated to not training
        self.assertFalse(module._is_training_active)
        
        # Verify execute was called
        module.execute_stop_training.assert_called_once()
        
        # Verify logging
        module.log_info.assert_called_with("Stopping training...")
        module.log_success.assert_called_with("Training stopped: Training stopped")
    
    def test_handler_error_recovery(self):
        """Test that handlers recover correctly from errors."""
        from smartcash.ui.model.training.training_uimodule import TrainingUIModule
        
        module = TrainingUIModule()
        module._ui_components = self.mock_ui_components
        module._is_initialized = True
        
        # Mock the execute method to fail and logging
        module.execute_start_training = Mock(return_value={'success': False, 'message': 'Training failed to start'})
        module.log_info = Mock()
        module.log_error = Mock()
        
        # Initially not training
        module._is_training_active = False
        
        # Call handler
        module._handle_start_training()
        
        # Verify state was reset to not training on failure
        self.assertFalse(module._is_training_active)
        
        # Verify error was logged
        module.log_error.assert_called_with("Training start failed: Training failed to start")


if __name__ == '__main__':
    unittest.main()