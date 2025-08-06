"""
Factory functions for creating early stopping instances.

Provides convenient factory functions to create different types of early stopping
based on configuration dictionaries.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from .standard import StandardEarlyStopping
from .adaptive import AdaptiveEarlyStopping
from .phase_specific import PhaseSpecificEarlyStopping


def create_early_stopping(config: Dict[str, Any], save_best_path: Optional[str] = None) -> StandardEarlyStopping:
    """
    Factory function to create standard early stopping from config.
    
    Args:
        config: Training configuration dictionary
        save_best_path: Optional path to save best model
        
    Returns:
        StandardEarlyStopping instance or disabled early stopping
    """
    es_config = config.get('training', {}).get('early_stopping', {})
    
    if not es_config or not es_config.get('enabled', True):
        # Return dummy early stopping that never stops
        class NoEarlyStopping:
            def __init__(self):
                self.patience = 0
                self.best_score = 0.0
                self.best_epoch = 0
                self.metric = 'disabled'
                self.mode = 'max'
                self.should_stop = False
                
            def __call__(self, score, model=None, epoch=0):
                # Silent operation - no prints, no stopping
                return False
                
            def reset(self): 
                pass
                
            def get_best_info(self): 
                return {'best_score': None, 'should_stop': False}
        
        return NoEarlyStopping()
    
    # Use provided save_best_path if available
    if save_best_path is None and es_config.get('save_best_model', True):
        # Get checkpoint configuration
        checkpoint_config = config.get('checkpoint', {})
        save_dir = Path(checkpoint_config.get('save_dir', 'data/checkpoints'))
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate best checkpoint filename using similar logic as CheckpointManager
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        
        backbone = model_config.get('backbone', 'unknown')
        training_mode = training_config.get('training_mode', 'two_phase')
        layer_mode = model_config.get('layer_mode', 'multi')
        freeze_backbone = model_config.get('freeze_backbone', False)
        freeze_status = 'frozen' if freeze_backbone else 'unfrozen'
        pretrained = model_config.get('pretrained', False)
        
        # Build best checkpoint filename
        name_parts = ['best', backbone, training_mode, layer_mode, freeze_status]
        if pretrained:
            name_parts.append('pretrained')
        
        # Add timestamp for early stopping best model (separate from checkpoint manager)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name_parts.append(f'es_{timestamp}')  # 'es' prefix to distinguish from regular best checkpoints
        
        filename = '_'.join(name_parts) + '.pt'
        save_best_path = str(save_dir / filename)
    
    return StandardEarlyStopping(
        patience=es_config.get('patience', 15),  # Match default from args parser
        min_delta=es_config.get('min_delta', 0.001),
        metric=es_config.get('metric', 'val_accuracy'),
        mode=es_config.get('mode', 'max'),
        save_best_path=save_best_path,
        verbose=True
    )


def create_adaptive_early_stopping(config: Dict[str, Any]) -> AdaptiveEarlyStopping:
    """
    Create adaptive early stopping from config.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        AdaptiveEarlyStopping instance
    """
    es_config = config.get('training', {}).get('early_stopping', {})
    
    return AdaptiveEarlyStopping(
        initial_patience=es_config.get('patience', 15),  # Match default from args parser
        patience_factor=es_config.get('adaptive_factor', 1.5),
        max_patience=es_config.get('max_patience', 50),
        improvement_threshold=es_config.get('improvement_threshold', 0.01),
        min_delta=es_config.get('min_delta', 0.001),
        metric=es_config.get('metric', 'val_accuracy'),
        mode=es_config.get('mode', 'max')
    )


def create_phase_specific_early_stopping(config: Dict[str, Any]) -> PhaseSpecificEarlyStopping:
    """
    Create phase-specific early stopping from config with SmartCash training criteria.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        PhaseSpecificEarlyStopping instance or disabled early stopping
    """
    es_config = config.get('training', {}).get('early_stopping', {})
    training_config = config.get('training', {})
    
    # Determine if phase-specific early stopping is enabled
    if not es_config.get('enabled', True) or not es_config.get('phase_specific', False):
        # Return disabled early stopping
        class DisabledPhaseEarlyStopping:
            def __init__(self):
                self.current_phase = 1
                self.should_stop = False
                
            def set_phase(self, phase): 
                self.current_phase = phase
                
            def __call__(self, metrics, model=None, epoch=0): 
                return False
                
            def get_status_summary(self): 
                return {'should_stop': False}
                
            def reset(self): 
                pass
        
        return DisabledPhaseEarlyStopping()
    
    # Phase 1 configuration from config with smart defaults
    phase1_defaults = {
        'loss_patience': 8,
        'loss_min_delta': 0.01,
        'metric_patience': 6,
        'metric_min_delta': 0.005,
        'metric_name': 'val_accuracy',  # or 'f1' depending on preference
        'stability_threshold': 0.002
    }
    
    phase1_config = es_config.get('phase1', {})
    for key, default_value in phase1_defaults.items():
        phase1_config.setdefault(key, default_value)
    
    # Phase 2 configuration from config with smart defaults
    phase2_defaults = {
        'f1_patience': 10,
        'map_patience': 10,
        'min_improvement': 0.01,
        'overfitting_threshold': 0.05,
        'overfitting_patience': 5,
        'combo_mode': 'both'  # or 'any' for less strict
    }
    
    phase2_config = es_config.get('phase2', {})
    for key, default_value in phase2_defaults.items():
        phase2_config.setdefault(key, default_value)
    
    # Apply global patience setting if specified (allows --patience to influence phase-specific early stopping)
    if 'patience' in es_config and es_config['patience'] != 15:  # 15 is the default, so only override if user specified different value
        user_patience = es_config['patience']
        # Scale the phase-specific patience values proportionally
        phase1_config['loss_patience'] = max(int(user_patience * 0.5), 3)  # At least 3 epochs
        phase1_config['metric_patience'] = max(int(user_patience * 0.4), 3)  # At least 3 epochs
        phase2_config['f1_patience'] = max(int(user_patience * 0.7), 5)  # At least 5 epochs
        phase2_config['map_patience'] = max(int(user_patience * 0.7), 5)  # At least 5 epochs
        phase2_config['overfitting_patience'] = max(int(user_patience * 0.3), 3)  # At least 3 epochs
    
    # Adjust based on training mode
    training_mode = training_config.get('training_mode', 'two_phase')
    if training_mode == 'single_phase':
        # Only Phase 1 will be used - use full patience value
        phase1_config['loss_patience'] = es_config.get('patience', 15)
        phase1_config['metric_patience'] = es_config.get('patience', 15)
    
    return PhaseSpecificEarlyStopping(
        phase1_config=phase1_config,
        phase2_config=phase2_config,
        verbose=es_config.get('verbose', True),
        save_best_path=None  # Will be set by phase orchestrator if needed
    )