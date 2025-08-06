"""
Phase configurator for model and training configuration.

Handles model configuration, freezing, and phase-specific setup.
"""

from typing import Dict, Any
from .base import BasePhaseManager
from .mixins.model_configuration import ModelConfigurationMixin


class PhaseConfigurator(BasePhaseManager, ModelConfigurationMixin):
    """Specialized phase manager for model configuration tasks."""
    
    def __init__(self, model, model_api, config, progress_tracker):
        """Initialize phase configurator."""
        super().__init__(model, model_api, config, progress_tracker)
        self.logger.info("⚙️ PhaseConfigurator initialized")
    
    def setup_phase(self, phase_num: int, **kwargs) -> Dict[str, Any]:
        """
        Set up phase configuration.
        
        Args:
            phase_num: Phase number to configure
            **kwargs: Additional configuration parameters
            
        Returns:
            Configuration status
        """
        self._set_current_phase(phase_num)
        self.configure_model_phase(phase_num)
        
        return {
            'phase': phase_num,
            'configured': True,
            'model_configured': True
        }
    
    def execute_phase(self, phase_num: int, **kwargs) -> Dict[str, Any]:
        """Phase configurator doesn't execute training, only configuration."""
        raise NotImplementedError("PhaseConfigurator only handles configuration, not execution")
    
    def apply_phase_configuration(self, phase_num: int) -> bool:
        """
        Apply all configuration for a specific phase.
        
        Args:
            phase_num: Phase number to configure
            
        Returns:
            True if configuration was successful
        """
        try:
            self.setup_phase(phase_num)
            self.logger.info(f"✅ Phase {phase_num} configuration applied successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to configure Phase {phase_num}: {e}")
            return False
    
    def get_phase_configuration_summary(self, phase_num: int) -> Dict[str, Any]:
        """Get a summary of the phase configuration."""
        training_mode = self.config.get('training_mode', 'two_phase')
        
        summary = {
            'phase': phase_num,
            'training_mode': training_mode,
            'model_configured': hasattr(self.model, 'current_phase'),
            'current_phase': getattr(self.model, 'current_phase', None)
        }
        
        # Add phase-specific information
        if training_mode == 'two_phase':
            if phase_num == 1:
                summary.update({
                    'layer_mode': 'single',
                    'backbone_frozen': True,
                    'description': 'Phase 1: Single layer, frozen backbone'
                })
            else:
                summary.update({
                    'layer_mode': 'multi',
                    'backbone_frozen': False,
                    'description': 'Phase 2: Multi layer, unfrozen backbone'
                })
        else:
            layer_mode = self.config.get('model', {}).get('layer_mode', 'multi')
            freeze_backbone = self.config.get('single_phase_freeze_backbone', False)
            summary.update({
                'layer_mode': layer_mode,
                'backbone_frozen': freeze_backbone,
                'description': f'Single phase: {layer_mode} layer, {"frozen" if freeze_backbone else "unfrozen"} backbone'
            })
        
        return summary