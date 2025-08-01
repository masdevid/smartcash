"""
File: smartcash/model/training/utils/early_stopping.py
Deskripsi: Early stopping implementation untuk training optimization dengan currency detection focus
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
import time

class EarlyStopping:
    """Early stopping implementation dengan multi-metric support dan checkpoint management"""
    
    def __init__(self, patience: int = 30, min_delta: float = 0.001, 
                 metric: str = 'val_accuracy', mode: str = 'max',
                 restore_best_weights: bool = True, save_best_path: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs dengan no improvement sebelum stopping
            min_delta: Minimum change yang dianggap sebagai improvement
            metric: Metric yang akan dimonitor untuk early stopping
            mode: 'max' untuk metrics yang higher is better, 'min' untuk lower is better
            restore_best_weights: Restore model ke best weights saat stopping
            save_best_path: Path untuk save best model weights
            verbose: Print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode.lower()
        self.restore_best_weights = restore_best_weights
        self.save_best_path = save_best_path
        self.verbose = verbose
        
        # State tracking
        self.best_score = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
        
        # Model state untuk restoration
        self.best_weights = None
        
        # History tracking
        self.history = {
            'scores': [],
            'improvements': [],
            'wait_counts': []
        }
        
        # Validation
        if self.mode not in ['max', 'min']:
            raise ValueError(f"❌ Mode harus 'max' atau 'min', got: {self.mode}")
    
    def __call__(self, current_score: float, model: Optional[torch.nn.Module] = None, 
                 epoch: int = None) -> bool:
        """
        Check if training should stop berdasarkan current metric
        
        Args:
            current_score: Current metric value
            model: Model untuk weight saving/restoration
            epoch: Current epoch number
            
        Returns:
            True jika training should stop, False otherwise
        """
        score = float(current_score)
        epoch = epoch or len(self.history['scores'])
        
        # Record history
        self.history['scores'].append(score)
        
        # Check untuk improvement
        is_improvement = self._is_improvement(score)
        self.history['improvements'].append(is_improvement)
        
        if is_improvement:
            self.best_score = score
            self.best_epoch = epoch
            self.wait = 0
            
            # Save best weights
            if model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
                # Save ke file jika path disediakan
                if self.save_best_path:
                    self._save_best_model(model, epoch, score)
            
            if self.verbose:
                direction = "↗️" if self.mode == 'max' else "↘️"
                print(f"✨ Early stopping: {self.metric} improved {direction} {score:.6f} (epoch {epoch})")
        
        else:
            self.wait += 1
            if self.verbose and self.wait > 0:
                print(f"⏳ Early stopping: {self.metric} tidak improve ({self.wait}/{self.patience})")
        
        # Record wait count
        self.history['wait_counts'].append(self.wait)
        
        # Check untuk stopping
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.should_stop = True
            
            if self.verbose:
                print(f"🛑 Early stopping triggered! Best {self.metric}: {self.best_score:.6f} (epoch {self.best_epoch})")
            
            # Restore best weights
            if self.restore_best_weights and model is not None and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print(f"🔄 Restored model weights dari epoch {self.best_epoch}")
        
        return self.should_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if current score is improvement"""
        if self.best_score is None:
            return True
        
        if self.mode == 'max':
            return score > (self.best_score + self.min_delta)
        else:  # mode == 'min'
            return score < (self.best_score - self.min_delta)
    
    def _save_best_model(self, model: torch.nn.Module, epoch: int, score: float) -> None:
        """Save best model ke file"""
        try:
            save_path = Path(self.save_best_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare checkpoint data
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_score': score,
                'metric': self.metric,
                'early_stopping_config': {
                    'patience': self.patience,
                    'min_delta': self.min_delta,
                    'mode': self.mode
                },
                'timestamp': time.time()
            }
            
            torch.save(checkpoint, save_path)
            
            if self.verbose:
                print(f"💾 Best model disimpan ke: {save_path}")
                
        except Exception as e:
            print(f"⚠️ Error saving best model: {str(e)}")
    
    def reset(self) -> None:
        """Reset early stopping state untuk training baru"""
        self.best_score = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
        self.best_weights = None
        
        # Clear history
        self.history = {
            'scores': [],
            'improvements': [],
            'wait_counts': []
        }
        
        if self.verbose:
            print("🔄 Early stopping state reset")
    
    def get_best_info(self) -> Dict[str, Any]:
        """Get information tentang best model"""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'metric': self.metric,
            'mode': self.mode,
            'current_wait': self.wait,
            'patience': self.patience,
            'stopped': self.should_stop,
            'stopped_epoch': self.stopped_epoch
        }
    
    def get_improvement_ratio(self) -> float:
        """Get ratio dari epochs dengan improvement"""
        if not self.history['improvements']:
            return 0.0
        
        improvements = sum(self.history['improvements'])
        total_epochs = len(self.history['improvements'])
        
        return improvements / total_epochs
    
    def get_plateau_duration(self) -> int:
        """Get current plateau duration (epochs tanpa improvement)"""
        return self.wait
    
    def get_summary(self) -> str:
        """Get summary string untuk logging"""
        if self.best_score is None:
            return f"Early stopping: No scores recorded yet"
        
        status = "🛑 STOPPED" if self.should_stop else f"⏳ Waiting ({self.wait}/{self.patience})"
        direction = "↗️" if self.mode == 'max' else "↘️"
        
        return f"Early stopping: {status} | Best {self.metric}: {self.best_score:.6f} {direction} (epoch {self.best_epoch})"

class MultiMetricEarlyStopping:
    """Early stopping dengan multiple metrics monitoring"""
    
    def __init__(self, metrics_config: List[Dict[str, Any]], 
                 combination_mode: str = 'any', verbose: bool = True):
        """
        Initialize multi-metric early stopping
        
        Args:
            metrics_config: List dari metric configs, each containing:
                - metric: metric name
                - patience: patience untuk metric ini
                - min_delta: minimum delta
                - mode: 'max' atau 'min'
                - weight: weight untuk combination (optional)
            combination_mode: 'any' (stop jika any metric triggers), 'all' (stop jika all trigger)
            verbose: Print messages
        """
        self.combination_mode = combination_mode.lower()
        self.verbose = verbose
        
        # Create individual early stopping untuk each metric
        self.stoppers = {}
        for config in metrics_config:
            metric_name = config['metric']
            self.stoppers[metric_name] = EarlyStopping(
                patience=config.get('patience', 10),
                min_delta=config.get('min_delta', 0.001),
                metric=metric_name,
                mode=config.get('mode', 'max'),
                restore_best_weights=False,  # Handle ini di level atas
                verbose=False  # Handle messaging di level ini
            )
        
        # State tracking
        self.should_stop = False
        self.stopped_epoch = 0
        self.triggered_metrics = []
    
    def __call__(self, metrics: Dict[str, float], model: Optional[torch.nn.Module] = None,
                 epoch: int = None) -> bool:
        """
        Check multiple metrics untuk early stopping
        
        Args:
            metrics: Dictionary dari metric values
            model: Model untuk weight management
            epoch: Current epoch
            
        Returns:
            True jika should stop
        """
        epoch = epoch or 0
        triggered_stoppers = []
        
        # Check each metric
        for metric_name, stopper in self.stoppers.items():
            if metric_name in metrics:
                should_stop = stopper(metrics[metric_name], model, epoch)
                if should_stop:
                    triggered_stoppers.append(metric_name)
        
        # Determine overall stopping decision
        if self.combination_mode == 'any':
            self.should_stop = len(triggered_stoppers) > 0
        else:  # combination_mode == 'all'
            self.should_stop = len(triggered_stoppers) == len(self.stoppers)
        
        if self.should_stop and not self.triggered_metrics:
            self.stopped_epoch = epoch
            self.triggered_metrics = triggered_stoppers.copy()
            
            if self.verbose:
                triggered_str = ", ".join(triggered_stoppers)
                print(f"🛑 Multi-metric early stopping triggered! Metrics: {triggered_str}")
        
        return self.should_stop
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary untuk semua metrics"""
        summary = {
            'should_stop': self.should_stop,
            'stopped_epoch': self.stopped_epoch,
            'triggered_metrics': self.triggered_metrics,
            'individual_status': {}
        }
        
        for metric_name, stopper in self.stoppers.items():
            summary['individual_status'][metric_name] = stopper.get_best_info()
        
        return summary
    
    def reset(self) -> None:
        """Reset semua stoppers"""
        for stopper in self.stoppers.values():
            stopper.reset()
        
        self.should_stop = False
        self.stopped_epoch = 0
        self.triggered_metrics = []

class AdaptiveEarlyStopping(EarlyStopping):
    """Early stopping dengan adaptive patience berdasarkan training progress"""
    
    def __init__(self, initial_patience: int = 10, patience_factor: float = 1.5,
                 max_patience: int = 50, improvement_threshold: float = 0.01, **kwargs):
        """
        Initialize adaptive early stopping
        
        Args:
            initial_patience: Initial patience value
            patience_factor: Factor untuk increase patience saat ada improvement
            max_patience: Maximum patience value
            improvement_threshold: Threshold untuk significant improvement
            **kwargs: Additional arguments untuk base EarlyStopping
        """
        super().__init__(patience=initial_patience, **kwargs)
        
        self.initial_patience = initial_patience
        self.patience_factor = patience_factor
        self.max_patience = max_patience
        self.improvement_threshold = improvement_threshold
        
        # Track significant improvements
        self.significant_improvements = 0
        self.last_significant_score = None
    
    def __call__(self, current_score: float, model: Optional[torch.nn.Module] = None,
                 epoch: int = None) -> bool:
        """Check dengan adaptive patience logic"""
        
        # Check untuk significant improvement
        if self._is_significant_improvement(current_score):
            self.significant_improvements += 1
            self.last_significant_score = current_score
            
            # Increase patience setelah significant improvement
            new_patience = min(
                int(self.patience * self.patience_factor),
                self.max_patience
            )
            
            if new_patience != self.patience:
                old_patience = self.patience
                self.patience = new_patience
                if self.verbose:
                    print(f"📈 Significant improvement detected! Patience increased: {old_patience} → {new_patience}")
        
        # Call parent implementation
        return super().__call__(current_score, model, epoch)
    
    def _is_significant_improvement(self, score: float) -> bool:
        """Check apakah improvement significant enough untuk adjust patience"""
        if self.last_significant_score is None:
            self.last_significant_score = score
            return True
        
        if self.mode == 'max':
            improvement = score - self.last_significant_score
            return improvement > self.improvement_threshold
        else:
            improvement = self.last_significant_score - score
            return improvement > self.improvement_threshold
    
    def get_adaptive_info(self) -> Dict[str, Any]:
        """Get adaptive early stopping information"""
        base_info = self.get_best_info()
        adaptive_info = {
            'initial_patience': self.initial_patience,
            'current_patience': self.patience,
            'max_patience': self.max_patience,
            'significant_improvements': self.significant_improvements,
            'patience_factor': self.patience_factor,
            'improvement_threshold': self.improvement_threshold
        }
        
        return {**base_info, **adaptive_info}

# Convenience functions
def create_early_stopping(config: Dict[str, Any]) -> EarlyStopping:
    """Factory function untuk create early stopping dari config"""
    es_config = config.get('training', {}).get('early_stopping', {})
    
    if not es_config.get('enabled', True):
        # Return dummy early stopping yang never stops
        class NoEarlyStopping:
            def __init__(self):
                self.patience = 0
                self.best_score = 0.0
                self.best_epoch = 0
                self.metric = 'disabled'
                self.should_stop = False
                
            def __call__(self, score, model=None, epoch=0):
                # Silent operation - no prints, no stopping
                return False
                
            def reset(self): 
                pass
                
            def get_best_info(self): 
                return {'best_score': None, 'should_stop': False}
        
        return NoEarlyStopping()
    
    return EarlyStopping(
        patience=es_config.get('patience', 10),
        min_delta=es_config.get('min_delta', 0.001),
        metric=es_config.get('metric', 'val_accuracy'),
        mode=es_config.get('mode', 'max'),
        verbose=True
    )

def create_adaptive_early_stopping(config: Dict[str, Any]) -> AdaptiveEarlyStopping:
    """Create adaptive early stopping dari config"""
    es_config = config.get('training', {}).get('early_stopping', {})
    
    return AdaptiveEarlyStopping(
        initial_patience=es_config.get('patience', 10),
        patience_factor=es_config.get('adaptive_factor', 1.5),
        max_patience=es_config.get('max_patience', 50),
        improvement_threshold=es_config.get('improvement_threshold', 0.01),
        min_delta=es_config.get('min_delta', 0.001),
        metric=es_config.get('metric', 'val_accuracy'),
        mode=es_config.get('mode', 'max')
    )