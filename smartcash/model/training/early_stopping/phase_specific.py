"""
Phase-specific early stopping implementation.

Early stopping with different criteria for different training phases,
specifically designed for SmartCash two-phase training.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any
from .base import BaseEarlyStopping


class PhaseSpecificEarlyStopping(BaseEarlyStopping):
    """Early stopping with phase-specific criteria for SmartCash two-phase training."""
    
    def __init__(self, phase1_config: Dict[str, Any] = None, phase2_config: Dict[str, Any] = None, 
                 verbose: bool = True, save_best_path: Optional[str] = None):
        """
        Initialize phase-specific early stopping.
        
        Args:
            phase1_config: Configuration for Phase 1 early stopping criteria
            phase2_config: Configuration for Phase 2 early stopping criteria  
            verbose: Print status messages
            save_best_path: Path to save best model weights
        """
        # Use reasonable default patience for base class
        super().__init__(patience=15, save_best_path=save_best_path, verbose=verbose)
        
        self.current_phase = 1
        self.stop_reason = ""
        
        # Phase 1 configuration - Train loss plateau + val_accuracy/f1 stability
        self.phase1_config = phase1_config or {
            'loss_patience': 8,           # Epochs to monitor train loss plateau
            'loss_min_delta': 0.01,       # Minimum delta for loss improvement
            'metric_patience': 6,         # Epochs to monitor val_accuracy/f1 stability
            'metric_min_delta': 0.005,    # Minimum delta for metric improvement
            'metric_name': 'val_accuracy', # Primary metric (val_accuracy or f1)
            'stability_threshold': 0.002   # Threshold for "stable or slow improvement"
        }
        
        # Phase 2 configuration - F1 and mAP improvement with overfitting detection  
        self.phase2_config = phase2_config or {
            'f1_patience': 10,            # Patience for F1 improvement
            'map_patience': 10,           # Patience for mAP improvement  
            'min_improvement': 0.01,      # Minimum improvement threshold
            'overfitting_threshold': 0.05, # Max train/val gap before overfitting
            'overfitting_patience': 5,    # Epochs to monitor overfitting
            'combo_mode': 'both'          # 'both' (F1 and mAP), 'any' (F1 or mAP)
        }
        
        # State tracking for Phase 1
        self.phase1_loss_history = []
        self.phase1_metric_history = []
        self.phase1_loss_wait = 0
        self.phase1_metric_wait = 0
        self.phase1_best_loss = None
        self.phase1_best_metric = None
        
        # State tracking for Phase 2
        self.phase2_f1_history = []
        self.phase2_map_history = []
        self.phase2_train_val_gaps = []
        self.phase2_f1_wait = 0
        self.phase2_map_wait = 0
        self.phase2_overfitting_wait = 0
        self.phase2_best_f1 = None
        self.phase2_best_map = None
        
        # Overall best tracking for cross-phase comparison
        self.global_best_score = None
        self.global_best_epoch = 0
        self.global_best_phase = 1
        
        if self.verbose:
            print(f"   Phase 1: Loss patience={self.phase1_config['loss_patience']}, "
                  f"Metric patience={self.phase1_config['metric_patience']}")
            print(f"   Phase 2: F1/mAP patience={self.phase2_config['f1_patience']}, "
                  f"Overfitting threshold={self.phase2_config['overfitting_threshold']}")
    
    def set_phase(self, phase: int) -> None:
        """Set current training phase."""
        if phase != self.current_phase:
            self.current_phase = phase
            if self.verbose:
                print(f"🔄 Early stopping switched to Phase {phase}")
    
    def __call__(self, metrics: Dict[str, float], model: Optional[torch.nn.Module] = None,
                 epoch: int = None) -> bool:
        """
        Check early stopping based on current phase and metrics.
        
        Args:
            metrics: Dictionary containing training metrics
            model: Model for weight management and best model saving
            epoch: Current epoch number
            
        Returns:
            True if should stop
        """
        epoch = epoch or 0
        
        if self.current_phase == 1:
            return self._check_phase1_stopping(metrics, epoch, model)
        elif self.current_phase == 2:
            return self._check_phase2_stopping(metrics, epoch, model)
        else:
            if self.verbose:
                print(f"⚠️ Unknown phase {self.current_phase}, no early stopping applied")
            return False
    
    def _check_metric_improvement(self, current_value: float, best_value: Optional[float], 
                                min_delta: float, mode: str = 'min') -> bool:
        """Generic method to check metric improvement."""
        if best_value is None:
            return True
        
        if mode == 'min':
            return current_value < (best_value - min_delta)
        else:  # mode == 'max'
            return current_value > (best_value + min_delta)
    
    def _check_phase1_stopping(self, metrics: Dict[str, float], epoch: int, 
                              model: Optional[torch.nn.Module] = None) -> bool:
        """Phase 1 criteria: Train loss plateau + val_accuracy/f1 stability."""
        train_loss = metrics.get('train_loss')
        metric_name = self.phase1_config['metric_name']
        metric_value = metrics.get(metric_name)
        
        if train_loss is None or metric_value is None:
            return False
        
        # Track histories
        self.phase1_loss_history.append(train_loss)
        self.phase1_metric_history.append(metric_value)
        
        # Update global best tracking
        combined_score = metric_value  # Use primary metric as global score
        if self.global_best_score is None or combined_score > self.global_best_score:
            self.global_best_score = combined_score
            self.global_best_epoch = epoch
            self.global_best_phase = 1
            # Save best model
            if model is not None:
                self.save_best_model(model, epoch, metrics, {
                    'phase': self.current_phase,
                    'global_best_info': {
                        'score': self.global_best_score,
                        'epoch': self.global_best_epoch,
                        'phase': self.global_best_phase
                    }
                })
        
        # Check train loss plateau
        loss_plateaued = self._check_loss_plateau(train_loss)
        
        # Check metric stability/slow improvement
        metric_stable = self._check_metric_stability(metric_value)
        
        # Early stop if BOTH conditions met
        if loss_plateaued and metric_stable:
            self.should_stop = True
            self.stopped_epoch = epoch
            self.stop_reason = f"Phase 1: Train loss plateau ({self.phase1_loss_wait}/{self.phase1_config['loss_patience']}) + {metric_name} stable ({self.phase1_metric_wait}/{self.phase1_config['metric_patience']})"
            
            if self.verbose:
                print(f"🛑 Phase 1 early stopping triggered!")
                print(f"   📉 Train loss plateaued: {train_loss:.6f} (wait: {self.phase1_loss_wait})")
                print(f"   📊 {metric_name} stable: {metric_value:.6f} (wait: {self.phase1_metric_wait})")
            
            return True
        
        return False
    
    def _check_loss_plateau(self, current_loss: float) -> bool:
        """Check if train loss has plateaued."""
        if self._check_metric_improvement(current_loss, self.phase1_best_loss, 
                                        self.phase1_config['loss_min_delta'], 'min'):
            # Loss improved
            old_best = self.phase1_best_loss
            self.phase1_best_loss = current_loss
            self.phase1_loss_wait = 0
            
            if self.verbose and old_best is not None:
                improvement = old_best - current_loss
                print(f"📉 Phase 1: Train loss improved ↘️ {current_loss:.6f} (▼{improvement:.6f})")
            
            return False
        else:
            self.phase1_loss_wait += 1
            
            if self.verbose:
                print(f"⏳ Phase 1: Train loss plateau {current_loss:.6f} ({self.phase1_loss_wait}/{self.phase1_config['loss_patience']})")
            
            return self.phase1_loss_wait >= self.phase1_config['loss_patience']
    
    def _check_metric_stability(self, current_metric: float) -> bool:
        """Check if validation metric is stable or improving slowly."""
        stability_threshold = self.phase1_config['stability_threshold']
        metric_name = self.phase1_config['metric_name']
        
        if self.phase1_best_metric is None:
            self.phase1_best_metric = current_metric
            self.phase1_metric_wait = 0
            return False
        
        # Check for significant improvement
        improvement = current_metric - self.phase1_best_metric
        
        if improvement > self.phase1_config['metric_min_delta']:
            # Significant improvement - reset wait
            self.phase1_best_metric = current_metric
            self.phase1_metric_wait = 0
            
            if self.verbose:
                print(f"📈 Phase 1: {metric_name} improved ↗️ {current_metric:.6f} (▲{improvement:.6f})")
            
            return False
        elif improvement > stability_threshold:
            # Small improvement (slow progress) - increment wait but slower
            self.phase1_metric_wait += 0.5
            
            if self.verbose:
                print(f"📊 Phase 1: {metric_name} small improvement ↗️ {current_metric:.6f} (▲{improvement:.6f}) - slow progress ({self.phase1_metric_wait:.1f}/{self.phase1_config['metric_patience']})")
            
        else:
            # No improvement or decline (stable) - increment wait
            self.phase1_metric_wait += 1
            
            if self.verbose:
                decline_info = f"(▼{abs(improvement):.6f})" if improvement < 0 else "(stable)"
                print(f"⏳ Phase 1: {metric_name} no improvement {current_metric:.6f} {decline_info} ({self.phase1_metric_wait}/{self.phase1_config['metric_patience']})")
        
        return self.phase1_metric_wait >= self.phase1_config['metric_patience']
    
    def _check_phase2_stopping(self, metrics: Dict[str, float], epoch: int, 
                              model: Optional[torch.nn.Module] = None) -> bool:
        """Phase 2 criteria: F1 and mAP improvement with overfitting detection."""
        f1_score = metrics.get('f1') or metrics.get('val_f1')
        map_score = metrics.get('map50') or metrics.get('val_map50') 
        train_loss = metrics.get('train_loss')
        val_loss = metrics.get('val_loss')
        
        if f1_score is None or map_score is None:
            return False
        
        # Track histories
        self.phase2_f1_history.append(f1_score)
        self.phase2_map_history.append(map_score)
        
        # Update global best tracking (use combined F1 + mAP score)
        combined_score = (f1_score + map_score) / 2
        if self.global_best_score is None or combined_score > self.global_best_score:
            self.global_best_score = combined_score
            self.global_best_epoch = epoch
            self.global_best_phase = 2
            # Save best model
            if model is not None:
                self.save_best_model(model, epoch, metrics, {
                    'phase': self.current_phase,
                    'global_best_info': {
                        'score': self.global_best_score,
                        'epoch': self.global_best_epoch,
                        'phase': self.global_best_phase
                    }
                })
        
        # Track train/val gap for overfitting detection
        if train_loss is not None and val_loss is not None:
            gap = abs(val_loss - train_loss)
            self.phase2_train_val_gaps.append(gap)
        
        # Check F1 stagnation
        f1_stagnant = self._check_phase2_metric_stagnation(f1_score, 'f1')
        
        # Check mAP stagnation  
        map_stagnant = self._check_phase2_metric_stagnation(map_score, 'map')
        
        # Check overfitting
        overfitting_detected = self._check_overfitting()
        
        # Determine stopping condition based on combo_mode
        combo_mode = self.phase2_config['combo_mode']
        
        if overfitting_detected:
            self.should_stop = True
            self.stopped_epoch = epoch
            self.stop_reason = f"Phase 2: Overfitting detected (train/val gap > {self.phase2_config['overfitting_threshold']})"
            
            if self.verbose:
                print(f"🛑 Phase 2 early stopping triggered!")
                print(f"   📈 Overfitting detected: train/val gap too large")
            
            return True
        
        elif combo_mode == 'both' and f1_stagnant and map_stagnant:
            self.should_stop = True
            self.stopped_epoch = epoch
            self.stop_reason = f"Phase 2: Both F1 and mAP stagnant"
            
            if self.verbose:
                print(f"🛑 Phase 2 early stopping triggered!")
                print(f"   📊 F1 stagnant: {f1_score:.6f} (wait: {self.phase2_f1_wait})")
                print(f"   📊 mAP stagnant: {map_score:.6f} (wait: {self.phase2_map_wait})")
            
            return True
        
        elif combo_mode == 'any' and (f1_stagnant or map_stagnant):
            stagnant_metric = "F1" if f1_stagnant else "mAP"
            self.should_stop = True
            self.stopped_epoch = epoch
            self.stop_reason = f"Phase 2: {stagnant_metric} stagnant"
            
            if self.verbose:
                print(f"🛑 Phase 2 early stopping triggered!")
                print(f"   📊 {stagnant_metric} stagnant")
            
            return True
        
        # Show overfitting warning without stopping
        if self.verbose and self.phase2_overfitting_wait > 0 and not overfitting_detected:
            print(f"⚠️ Phase 2: Overfitting risk ({self.phase2_overfitting_wait}/{self.phase2_config['overfitting_patience']})")
        
        return False
    
    def _check_phase2_metric_stagnation(self, current_score: float, metric_type: str) -> bool:
        """Check if F1 or mAP has stagnated."""
        is_f1 = metric_type == 'f1'
        best_score = self.phase2_best_f1 if is_f1 else self.phase2_best_map
        patience = self.phase2_config['f1_patience'] if is_f1 else self.phase2_config['map_patience']
        min_improvement = self.phase2_config['min_improvement']
        
        if self._check_metric_improvement(current_score, best_score, min_improvement, 'max'):
            # Improvement detected
            old_best = best_score
            if is_f1:
                self.phase2_best_f1 = current_score
                self.phase2_f1_wait = 0
            else:
                self.phase2_best_map = current_score
                self.phase2_map_wait = 0
            
            if self.verbose and old_best is not None:
                improvement = current_score - old_best
                metric_display = metric_type.upper()
                print(f"📈 Phase 2: {metric_display} improved ↗️ {current_score:.6f} (▲{improvement:.6f})")
            
            return False
        else:
            # No improvement
            if is_f1:
                self.phase2_f1_wait += 1
                wait_count = self.phase2_f1_wait
                metric_display = "F1"
            else:
                self.phase2_map_wait += 1
                wait_count = self.phase2_map_wait
                metric_display = "mAP"
            
            if self.verbose:
                decline_info = f"(▼{abs(current_score - best_score):.6f})" if current_score < best_score else "(stable)"
                print(f"⏳ Phase 2: {metric_display} no improvement {current_score:.6f} {decline_info} ({wait_count}/{patience})")
            
            return wait_count >= patience
    
    def _check_overfitting(self) -> bool:
        """Check for overfitting based on train/val gap."""
        if len(self.phase2_train_val_gaps) < 3:  # Need at least 3 epochs for trend
            return False
        
        # Check recent gaps
        recent_gaps = self.phase2_train_val_gaps[-3:]
        avg_recent_gap = np.mean(recent_gaps)
        
        overfitting_threshold = self.phase2_config['overfitting_threshold']
        
        if avg_recent_gap > overfitting_threshold:
            self.phase2_overfitting_wait += 1
            
            if self.verbose:
                print(f"⚠️ Phase 2: Train/val gap high {avg_recent_gap:.6f} > {overfitting_threshold:.6f} ({self.phase2_overfitting_wait}/{self.phase2_config['overfitting_patience']})")
        else:
            # Gap improved/normalized
            if self.phase2_overfitting_wait > 0 and self.verbose:
                print(f"✅ Phase 2: Train/val gap normalized {avg_recent_gap:.6f} ≤ {overfitting_threshold:.6f}")
            
            self.phase2_overfitting_wait = 0
        
        return self.phase2_overfitting_wait >= self.phase2_config['overfitting_patience']
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary."""
        return {
            'current_phase': self.current_phase,
            'should_stop': self.should_stop,
            'stopped_epoch': self.stopped_epoch,
            'stop_reason': self.stop_reason,
            'phase1_status': {
                'loss_wait': self.phase1_loss_wait,
                'loss_patience': self.phase1_config['loss_patience'],
                'metric_wait': self.phase1_metric_wait,
                'metric_patience': self.phase1_config['metric_patience'],
                'best_loss': self.phase1_best_loss,
                'best_metric': self.phase1_best_metric
            },
            'phase2_status': {
                'f1_wait': self.phase2_f1_wait,
                'f1_patience': self.phase2_config['f1_patience'],
                'map_wait': self.phase2_map_wait,
                'map_patience': self.phase2_config['map_patience'],
                'overfitting_wait': self.phase2_overfitting_wait,
                'overfitting_patience': self.phase2_config['overfitting_patience'],
                'best_f1': self.phase2_best_f1,
                'best_map': self.phase2_best_map
            },
            'global_best': {
                'score': self.global_best_score,
                'epoch': self.global_best_epoch,
                'phase': self.global_best_phase,
                'has_saved_model': self.best_weights is not None
            }
        }
    
    def reset(self) -> None:
        """Reset all state for new training session."""
        self.should_stop = False
        self.stopped_epoch = 0
        self.stop_reason = ""
        
        # Reset Phase 1 state
        self.phase1_loss_history = []
        self.phase1_metric_history = []
        self.phase1_loss_wait = 0
        self.phase1_metric_wait = 0
        self.phase1_best_loss = None
        self.phase1_best_metric = None
        
        # Reset Phase 2 state
        self.phase2_f1_history = []
        self.phase2_map_history = []
        self.phase2_train_val_gaps = []
        self.phase2_f1_wait = 0
        self.phase2_map_wait = 0
        self.phase2_overfitting_wait = 0
        self.phase2_best_f1 = None
        self.phase2_best_map = None
        
        # Reset global best tracking
        self.global_best_score = None
        self.global_best_epoch = 0
        self.global_best_phase = 1
        self.best_weights = None
        
        # Clear history
        self.history = {
            'scores': [],
            'improvements': [],
            'wait_counts': []
        }
        
        if self.verbose:
            print("🔄 Phase-specific early stopping state reset")
    
    def get_best_info(self) -> Dict[str, Any]:
        """Get information about best model across all phases."""
        return {
            'global_best_score': self.global_best_score,
            'global_best_epoch': self.global_best_epoch,
            'global_best_phase': self.global_best_phase,
            'has_best_weights': self.best_weights is not None,
            'save_best_path': self.save_best_path,
            'current_phase': self.current_phase,
            'should_stop': self.should_stop,
            'stopped_epoch': self.stopped_epoch,
            'stop_reason': self.stop_reason
        }