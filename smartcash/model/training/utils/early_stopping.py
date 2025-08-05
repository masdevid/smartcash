"""
File: smartcash/model/training/utils/early_stopping.py
Deskripsi: Early stopping implementation untuk training optimization dengan currency detection focus
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
import time
from datetime import datetime

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
        
        if self.verbose:
            print(f"🔍 Early stopping initialized with patience={patience}, metric={metric}, mode={mode}")
        
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
                print(f"✨ Early stopping: {self.metric} improved {direction} {score:.6f} (epoch {epoch + 1})")
        
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
                print(f"🛑 Early stopping triggered! Best {self.metric}: {self.best_score:.6f} (epoch {self.best_epoch + 1})")
            
            # Restore best weights
            if self.restore_best_weights and model is not None and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print(f"🔄 Restored model weights dari epoch {self.best_epoch + 1}")
        
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
        
        return f"Early stopping: {status} | Best {self.metric}: {self.best_score:.6f} {direction} (epoch {self.best_epoch + 1})"

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
                patience=config.get('patience', 15),  # Match default from args parser
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

class PhaseSpecificEarlyStopping:
    """Early stopping dengan phase-specific criteria untuk SmartCash two-phase training"""
    
    def __init__(self, phase1_config: Dict[str, Any] = None, phase2_config: Dict[str, Any] = None, 
                 verbose: bool = True):
        """
        Initialize phase-specific early stopping
        
        Args:
            phase1_config: Configuration untuk Phase 1 early stopping criteria
            phase2_config: Configuration untuk Phase 2 early stopping criteria  
            verbose: Print status messages
        """
        self.verbose = verbose
        self.current_phase = 1
        self.should_stop = False
        self.stopped_epoch = 0
        self.stop_reason = ""
        
        # Phase 1 configuration - Train loss plateau + val_accuracy/f1 stability
        self.phase1_config = phase1_config or {
            'loss_patience': 8,           # Epochs untuk monitor train loss plateau
            'loss_min_delta': 0.01,       # Minimum delta untuk loss improvement
            'metric_patience': 6,         # Epochs untuk monitor val_accuracy/f1 stability
            'metric_min_delta': 0.005,    # Minimum delta untuk metric improvement
            'metric_name': 'val_accuracy', # Primary metric (val_accuracy atau f1)
            'stability_threshold': 0.002   # Threshold untuk "stabil atau naik pelan"
        }
        
        # Phase 2 configuration - F1 dan mAP improvement with overfitting detection  
        self.phase2_config = phase2_config or {
            'f1_patience': 10,            # Patience untuk F1 improvement
            'map_patience': 10,           # Patience untuk mAP improvement  
            'min_improvement': 0.01,      # Minimum improvement threshold
            'overfitting_threshold': 0.05, # Max train/val gap sebelum overfitting
            'overfitting_patience': 5,    # Epochs untuk monitor overfitting
            'combo_mode': 'both'          # 'both' (F1 and mAP), 'any' (F1 or mAP)
        }
        
        # State tracking untuk Phase 1
        self.phase1_loss_history = []
        self.phase1_metric_history = []
        self.phase1_loss_wait = 0
        self.phase1_metric_wait = 0
        self.phase1_best_loss = None
        self.phase1_best_metric = None
        
        # State tracking untuk Phase 2
        self.phase2_f1_history = []
        self.phase2_map_history = []
        self.phase2_train_val_gaps = []
        self.phase2_f1_wait = 0
        self.phase2_map_wait = 0
        self.phase2_overfitting_wait = 0
        self.phase2_best_f1 = None
        self.phase2_best_map = None
        
        if self.verbose:
            print(f"🎯 Phase-specific early stopping initialized")
            print(f"   Phase 1: Loss patience={self.phase1_config['loss_patience']}, "
                  f"Metric patience={self.phase1_config['metric_patience']}")
            print(f"   Phase 2: F1/mAP patience={self.phase2_config['f1_patience']}, "
                  f"Overfitting threshold={self.phase2_config['overfitting_threshold']}")
    
    def set_phase(self, phase: int) -> None:
        """Set current training phase"""
        if phase != self.current_phase:
            self.current_phase = phase
            if self.verbose:
                print(f"🔄 Early stopping switched to Phase {phase}")
    
    def __call__(self, metrics: Dict[str, float], model: Optional[torch.nn.Module] = None,
                 epoch: int = None) -> bool:
        """
        Check early stopping berdasarkan current phase dan metrics
        
        Args:
            metrics: Dictionary containing training metrics
            model: Model untuk weight management
            epoch: Current epoch number
            
        Returns:
            True jika should stop
        """
        epoch = epoch or 0
        
        if self.current_phase == 1:
            return self._check_phase1_stopping(metrics, epoch)
        elif self.current_phase == 2:
            return self._check_phase2_stopping(metrics, epoch)
        else:
            if self.verbose:
                print(f"⚠️ Unknown phase {self.current_phase}, no early stopping applied")
            return False
    
    def _check_phase1_stopping(self, metrics: Dict[str, float], epoch: int) -> bool:
        """
        Phase 1 criteria: Train loss mulai mendatar + val_accuracy atau f1 stabil atau naik pelan
        """
        train_loss = metrics.get('train_loss')
        metric_name = self.phase1_config['metric_name']
        metric_value = metrics.get(metric_name)
        
        if train_loss is None or metric_value is None:
            return False
        
        # Track histories
        self.phase1_loss_history.append(train_loss)
        self.phase1_metric_history.append(metric_value)
        
        # Check train loss plateau
        loss_plateaued = self._check_loss_plateau(train_loss)
        
        # Check metric stability/slow improvement
        metric_stable = self._check_metric_stability(metric_value)
        
        # Early stop jika BOTH conditions met
        if loss_plateaued and metric_stable:
            self.should_stop = True
            self.stopped_epoch = epoch
            self.stop_reason = f"Phase 1: Train loss plateau ({self.phase1_loss_wait}/{self.phase1_config['loss_patience']}) + {metric_name} stable ({self.phase1_metric_wait}/{self.phase1_config['metric_patience']})"
            
            if self.verbose:
                print(f"🛑 Phase 1 early stopping triggered!")
                print(f"   📉 Train loss plateaued: {train_loss:.6f} (wait: {self.phase1_loss_wait})")
                print(f"   📊 {metric_name} stable: {metric_value:.6f} (wait: {self.phase1_metric_wait})")
            
            return True
        
        # Note: Individual improvement/non-improvement logging is now handled in 
        # _check_loss_plateau and _check_metric_stability methods
        
        return False
    
    def _check_loss_plateau(self, current_loss: float) -> bool:
        """Check if train loss has plateaued"""
        if self.phase1_best_loss is None or current_loss < (self.phase1_best_loss - self.phase1_config['loss_min_delta']):
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
        """Check if validation metric is stable atau naik pelan"""
        stability_threshold = self.phase1_config['stability_threshold']
        
        if self.phase1_best_metric is None:
            self.phase1_best_metric = current_metric
            self.phase1_metric_wait = 0
            return False
        
        # Check untuk improvement yang significant
        improvement = current_metric - self.phase1_best_metric
        
        if improvement > self.phase1_config['metric_min_delta']:
            # Significant improvement - reset wait
            self.phase1_best_metric = current_metric
            self.phase1_metric_wait = 0
            
            if self.verbose:
                metric_name = self.phase1_config['metric_name']
                print(f"📈 Phase 1: {metric_name} improved ↗️ {current_metric:.6f} (▲{improvement:.6f})")
            
            return False
        elif improvement > stability_threshold:
            # Small improvement (naik pelan) - increment wait but slower
            self.phase1_metric_wait += 0.5
            
            if self.verbose:
                metric_name = self.phase1_config['metric_name']
                print(f"📊 Phase 1: {metric_name} small improvement ↗️ {current_metric:.6f} (▲{improvement:.6f}) - slow progress ({self.phase1_metric_wait:.1f}/{self.phase1_config['metric_patience']})")
            
        else:
            # No improvement atau decline (stabil) - increment wait
            self.phase1_metric_wait += 1
            
            if self.verbose:
                metric_name = self.phase1_config['metric_name']
                decline_info = f"(▼{abs(improvement):.6f})" if improvement < 0 else "(stable)"
                print(f"⏳ Phase 1: {metric_name} no improvement {current_metric:.6f} {decline_info} ({self.phase1_metric_wait}/{self.phase1_config['metric_patience']})")
        
        return self.phase1_metric_wait >= self.phase1_config['metric_patience']
    
    def _check_phase2_stopping(self, metrics: Dict[str, float], epoch: int) -> bool:
        """
        Phase 2 criteria: F1 dan mAP meningkat dan stabil, overfitting belum terlihat (gap train/val kecil)
        """
        f1_score = metrics.get('f1') or metrics.get('val_f1')
        map_score = metrics.get('map50') or metrics.get('val_map50') 
        train_loss = metrics.get('train_loss')
        val_loss = metrics.get('val_loss')
        
        if f1_score is None or map_score is None:
            return False
        
        # Track histories
        self.phase2_f1_history.append(f1_score)
        self.phase2_map_history.append(map_score)
        
        # Track train/val gap untuk overfitting detection
        if train_loss is not None and val_loss is not None:
            gap = abs(val_loss - train_loss)
            self.phase2_train_val_gaps.append(gap)
        
        # Check F1 stagnation
        f1_stagnant = self._check_phase2_metric_stagnation(f1_score, 'f1')
        
        # Check mAP stagnation  
        map_stagnant = self._check_phase2_metric_stagnation(map_score, 'map')
        
        # Check overfitting
        overfitting_detected = self._check_overfitting()
        
        # Determine stopping condition berdasarkan combo_mode
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
        
        # Note: Individual improvement/non-improvement logging is now handled in 
        # _check_phase2_metric_stagnation method. Only overfitting feedback here.
        if self.verbose and overfitting_detected and not (f1_stagnant or map_stagnant):
            print(f"⚠️ Phase 2: Overfitting risk detected ({self.phase2_overfitting_wait}/{self.phase2_config['overfitting_patience']})")
        
        return False
    
    def _check_phase2_metric_stagnation(self, current_score: float, metric_type: str) -> bool:
        """Check if F1 or mAP has stagnated"""
        if metric_type == 'f1':
            best_score = self.phase2_best_f1
            wait_counter = self.phase2_f1_wait
            patience = self.phase2_config['f1_patience']
        else:  # map
            best_score = self.phase2_best_map
            wait_counter = self.phase2_map_wait
            patience = self.phase2_config['map_patience']
        
        min_improvement = self.phase2_config['min_improvement']
        
        if best_score is None or current_score > (best_score + min_improvement):
            # Improvement detected
            old_best = best_score
            if metric_type == 'f1':
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
            if metric_type == 'f1':
                self.phase2_f1_wait += 1
                
                if self.verbose:
                    decline_info = f"(▼{abs(current_score - best_score):.6f})" if current_score < best_score else "(stable)"
                    print(f"⏳ Phase 2: F1 no improvement {current_score:.6f} {decline_info} ({self.phase2_f1_wait}/{patience})")
                
                return self.phase2_f1_wait >= patience
            else:
                self.phase2_map_wait += 1
                
                if self.verbose:
                    decline_info = f"(▼{abs(current_score - best_score):.6f})" if current_score < best_score else "(stable)"
                    print(f"⏳ Phase 2: mAP no improvement {current_score:.6f} {decline_info} ({self.phase2_map_wait}/{patience})")
                
                return self.phase2_map_wait >= patience
    
    def _check_overfitting(self) -> bool:
        """Check for overfitting berdasarkan train/val gap"""
        if len(self.phase2_train_val_gaps) < 3:  # Need at least 3 epochs untuk trend
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
        """Get comprehensive status summary"""
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
            }
        }
    
    def reset(self) -> None:
        """Reset all state untuk new training session"""
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
        
        if self.verbose:
            print("🔄 Phase-specific early stopping state reset")

# Convenience functions
def create_early_stopping(config: Dict[str, Any], save_best_path: Optional[str] = None) -> EarlyStopping:
    """Factory function untuk create early stopping dari config"""
    es_config = config.get('training', {}).get('early_stopping', {})
    
    if not es_config or not es_config.get('enabled', True):
        # Return dummy early stopping yang never stops
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
    
    return EarlyStopping(
        patience=es_config.get('patience', 15),  # Match default from args parser
        min_delta=es_config.get('min_delta', 0.001),
        metric=es_config.get('metric', 'val_accuracy'),
        mode=es_config.get('mode', 'max'),
        save_best_path=save_best_path,
        verbose=True
    )

def create_adaptive_early_stopping(config: Dict[str, Any]) -> AdaptiveEarlyStopping:
    """Create adaptive early stopping dari config"""
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
    """Create phase-specific early stopping dari config dengan SmartCash training criteria"""
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
    
    # Phase 1 configuration dari config dengan smart defaults
    phase1_defaults = {
        'loss_patience': 8,
        'loss_min_delta': 0.01,
        'metric_patience': 6,
        'metric_min_delta': 0.005,
        'metric_name': 'val_accuracy',  # atau 'f1' tergantung preference
        'stability_threshold': 0.002
    }
    
    phase1_config = es_config.get('phase1', {})
    for key, default_value in phase1_defaults.items():
        phase1_config.setdefault(key, default_value)
    
    # Phase 2 configuration dari config dengan smart defaults
    phase2_defaults = {
        'f1_patience': 10,
        'map_patience': 10,
        'min_improvement': 0.01,
        'overfitting_threshold': 0.05,
        'overfitting_patience': 5,
        'combo_mode': 'both'  # atau 'any' untuk less strict
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
        verbose=es_config.get('verbose', True)
    )