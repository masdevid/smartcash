"""
Model Configuration Manager

Handles core model configuration concerns including backbone freezing,
phase transitions, and model state management.
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class ModelConfigurationManager:
    """Manages core model configuration and state changes."""
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize model configuration manager.
        
        Args:
            model: PyTorch model instance
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.current_phase = None
        
    def configure_model_for_phase(self, phase_num: int):
        """
        Configure model for specific training phase.
        
        Args:
            phase_num: Training phase number (1 or 2)
        """
        logger.info(f"🔧 Configuring model for Phase {phase_num}")
        
        self._set_model_phase(phase_num)
        self._configure_layer_mode(phase_num)
        self._configure_backbone_freezing(phase_num)
        
        self.current_phase = phase_num
        logger.info(f"✅ Model configured for Phase {phase_num}")
    
    def _set_model_phase(self, phase_num: int):
        """Set phase on model and propagate to children."""
        if hasattr(self.model, '__setattr__'):
            self.model.current_phase = phase_num
            self._propagate_phase_to_children(self.model, phase_num)
    
    def _propagate_phase_to_children(self, module, phase_num: int):
        """Propagate current_phase to all child modules."""
        for name, child in module.named_children():
            child.current_phase = phase_num
            logger.debug(f"Set current_phase={phase_num} on {name} ({type(child).__name__})")
            self._propagate_phase_to_children(child, phase_num)
    
    def _configure_layer_mode(self, phase_num: int):
        """Configure layer mode based on phase and configuration."""
        if not hasattr(self.model, 'force_single_layer'):
            logger.debug("Model does not support layer mode configuration")
            return
            
        training_mode = self.config.get('training', {}).get('training_mode', 'two_phase')
        
        if training_mode == 'single_phase':
            # Single-phase mode: respect the configured layer mode
            single_layer_mode = self.config.get('model', {}).get('layer_mode', 'multi')
            if single_layer_mode == 'single':
                self.model.force_single_layer = True
                logger.info("🎯 Single-phase mode: forcing single layer output")
            else:
                self.model.force_single_layer = False
                logger.info("🎯 Single-phase mode: using multi-layer output")
            
        elif training_mode == 'two_phase':
            # Two-phase mode: Phase 1 = single layer focus, Phase 2 = multi layer
            if phase_num == 1:
                logger.info(f"🎯 Phase {phase_num}: single layer focus (layer_1 primary)")
            else:
                logger.info(f"🎯 Phase {phase_num}: multi-layer mode (all layers)")
            
            self.model.force_single_layer = False  # Use phase-based logic
    
    def _configure_backbone_freezing(self, phase_num: int):
        """Configure backbone freezing based on phase and training mode."""
        training_mode = self.config.get('training', {}).get('training_mode', 'two_phase')
        
        if training_mode == 'two_phase':
            # Two-phase: Phase 1 = frozen, Phase 2 = unfrozen
            should_freeze = (phase_num == 1)
            self._set_backbone_freezing_state(should_freeze, phase_num)
            
        elif training_mode == 'single_phase':
            # Single-phase: respect configuration
            should_freeze = self.config.get('single_phase_freeze_backbone', False)
            self._set_backbone_freezing_state(should_freeze, phase_num)
    
    def _set_backbone_freezing_state(self, freeze: bool, phase_num: int):
        """Set backbone freezing state on the model."""
        if hasattr(self.model, 'freeze_backbone'):
            self.model.freeze_backbone(freeze)
            status = "frozen" if freeze else "unfrozen"
            logger.info(f"🧊 Phase {phase_num}: Backbone {status}")
        else:
            logger.debug("Model does not support freeze_backbone method")
    
    def get_phase_config(self, phase_num: int) -> Dict[str, Any]:
        """Get configuration for a specific phase."""
        try:
            return self.config['training_phases'][f'phase_{phase_num}']
        except KeyError:
            logger.warning(f"No configuration found for phase_{phase_num}")
            return {}
    
    def validate_phase_config(self, phase_num: int) -> bool:
        """Validate that a phase has proper configuration."""
        phase_config = self.get_phase_config(phase_num)
        
        required_keys = ['learning_rate']
        missing_keys = [key for key in required_keys if key not in phase_config]
        
        if missing_keys:
            logger.error(f"Phase {phase_num} missing required config: {missing_keys}")
            return False
        
        return True
    
    def transition_to_phase(self, from_phase: int, to_phase: int) -> Dict[str, Any]:
        """
        Handle model configuration transition between phases.
        
        Args:
            from_phase: Current phase number
            to_phase: Target phase number
            
        Returns:
            Dictionary containing transition information
        """
        logger.info(f"🔄 Transitioning model configuration: Phase {from_phase} → Phase {to_phase}")
        
        # Configure for new phase
        self.configure_model_for_phase(to_phase)
        
        transition_info = {
            'from_phase': from_phase,
            'to_phase': to_phase,
            'model_updated': True,
            'timestamp': __import__('time').time()
        }
        
        logger.info(f"✅ Model transition completed: Phase {from_phase} → Phase {to_phase}")
        return transition_info


def create_model_configuration_manager(model, config: Dict[str, Any]) -> ModelConfigurationManager:
    """
    Factory function to create ModelConfigurationManager.
    
    Args:
        model: PyTorch model instance
        config: Training configuration
        
    Returns:
        ModelConfigurationManager instance
    """
    return ModelConfigurationManager(model, config)