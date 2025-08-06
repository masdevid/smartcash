"""
Model configuration mixin for phase management.

Handles model configuration and phase-specific setup.
"""

from typing import Optional, Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class ModelConfigurationMixin:
    """Mixin for model configuration and phase setup."""
    
    def configure_model_phase(self, phase_num: int):
        """Configure model for specific training phase."""
        self._set_model_phase(phase_num)
        self._configure_layer_mode(phase_num)
        self._configure_backbone_freezing(phase_num)
    
    def _set_model_phase(self, phase_num: int):
        """Set phase on model and propagate to children."""
        if hasattr(self, 'model') and self.model:
            self.model.current_phase = phase_num
            self._propagate_phase_to_children(self.model, phase_num)
    
    def _propagate_phase_to_children(self, module, phase_num: int):
        """Propagate current_phase to all child modules."""
        logger = get_logger(self.__class__.__name__)
        
        for name, child in module.named_children():
            child.current_phase = phase_num
            logger.debug(f"Set current_phase={phase_num} on {name} ({type(child).__name__})")
            self._propagate_phase_to_children(child, phase_num)
    
    def _configure_layer_mode(self, phase_num: int):
        """Configure layer mode based on phase and configuration."""
        if not hasattr(self, 'config') or not hasattr(self, 'model'):
            return
            
        training_mode = self.config.get('training_mode', 'two_phase')
        logger = get_logger(self.__class__.__name__)
        
        if training_mode == 'single_phase':
            # Single-phase mode: respect the configured layer mode
            single_layer_mode = self.config.get('model', {}).get('layer_mode', 'multi')
            if single_layer_mode == 'single':
                self.model.force_single_layer = True
                logger.info(f"🎯 Single-phase mode: forcing single layer output")
            else:
                self.model.force_single_layer = False
                logger.info(f"🎯 Single-phase mode: using multi-layer output")
            
            # Handle backbone freezing for single-phase mode
            single_freeze_backbone = self.config.get('single_phase_freeze_backbone', False)
            self._configure_backbone_freezing_state(single_freeze_backbone, phase_num)
            
        elif training_mode == 'two_phase':
            # Two-phase mode: Phase 1 = single layer + frozen backbone, Phase 2 = multi layer + unfrozen backbone
            if phase_num == 1:
                logger.info(f"🎯 Phase {phase_num}: single layer mode (layer_1 only)")
                self._configure_backbone_freezing_state(freeze=True, phase_num=phase_num)
            else:
                logger.info(f"🎯 Phase {phase_num}: multi-layer mode (all layers)")
                # Phase 2: Unfreeze backbone (handled by pipeline executor)
                
            self.model.force_single_layer = False  # Use phase-based logic
    
    def _configure_backbone_freezing(self, phase_num: int):
        """Configure backbone freezing based on phase."""
        training_mode = self.config.get('training_mode', 'two_phase')
        
        if training_mode == 'two_phase' and phase_num == 1:
            self._configure_backbone_freezing_state(freeze=True, phase_num=phase_num)
        elif training_mode == 'single_phase':
            freeze = self.config.get('single_phase_freeze_backbone', False)
            self._configure_backbone_freezing_state(freeze=freeze, phase_num=phase_num)
    
    def _configure_backbone_freezing_state(self, freeze: bool, phase_num: int):
        """Set backbone freezing state."""
        logger = get_logger(self.__class__.__name__)
        
        if hasattr(self.model, 'freeze_backbone'):
            self.model.freeze_backbone(freeze)
            status = "frozen" if freeze else "unfrozen"
            logger.info(f"🧊 Phase {phase_num}: Backbone {status}")
        else:
            logger.debug(f"Model does not support freeze_backbone method")



