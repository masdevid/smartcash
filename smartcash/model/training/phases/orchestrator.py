"""
Phase orchestrator for training coordination.

Handles phase setup, configuration, and component coordination.
"""

from typing import Dict, Any, Optional
from .base import BasePhaseManager
from .mixins.model_configuration import ModelConfigurationMixin
from .mixins.component_setup import ComponentSetupMixin


class PhaseOrchestrator(BasePhaseManager, ModelConfigurationMixin, ComponentSetupMixin):
    """Manages training phase setup, configuration, and high-level coordination."""
    
    def __init__(self, model, model_api, config, progress_tracker):
        """
        Initialize phase orchestrator.
        
        Args:
            model: PyTorch model
            model_api: Model API instance
            config: Training configuration
            progress_tracker: Progress tracking instance
        """
        super().__init__(model, model_api, config, progress_tracker)
        self.logger.info("🎭 PhaseOrchestrator initialized")
    
    def setup_phase(self, phase_num: int, epochs: int, save_best_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Set up training phase with all required components.
        
        Args:
            phase_num: Training phase number (1 or 2)
            epochs: Total number of epochs to train for
            save_best_path: Path to save the best model checkpoint
            
        Returns:
            Dictionary containing all setup components
        """
        self._set_current_phase(phase_num)
        self.logger.info(f"🎪 Setting up Phase {phase_num} with {epochs} epochs")
        
        # Configure model for this phase
        self.configure_model_phase(phase_num)
        
        # Set up all training components
        components = self.setup_training_components(phase_num, epochs, save_best_path)
        
        self.logger.info(f"✅ Phase {phase_num} setup completed")
        return components
    
    def execute_phase(self, phase_num: int, **kwargs) -> Dict[str, Any]:
        """
        Execute a training phase (delegated to TrainingPhaseManager).
        
        Args:
            phase_num: Phase number to execute
            **kwargs: Additional execution parameters
            
        Returns:
            Dictionary containing phase execution results
        """
        # PhaseOrchestrator focuses on setup; execution is handled by TrainingPhaseManager
        raise NotImplementedError("Phase execution should be handled by TrainingPhaseManager")
    
    def get_phase_config(self, phase_num: int) -> Dict[str, Any]:
        """Get configuration for a specific phase."""
        try:
            return self.config['training_phases'][f'phase_{phase_num}']
        except KeyError:
            self.logger.warning(f"No configuration found for phase_{phase_num}")
            return {}
    
    def validate_phase_config(self, phase_num: int) -> bool:
        """Validate that a phase has proper configuration."""
        phase_config = self.get_phase_config(phase_num)
        
        required_keys = ['learning_rate']
        missing_keys = [key for key in required_keys if key not in phase_config]
        
        if missing_keys:
            self.logger.error(f"Phase {phase_num} missing required config: {missing_keys}")
            return False
        
        return True