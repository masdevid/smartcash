"""
File: smartcash/model/training/utils/training_progress_bridge.py
Deskripsi: Bridge untuk menghubungkan training progress dengan UI Progress Tracker API
"""

import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass

@dataclass
class ProgressData:
    """Data structure untuk progress information"""
    overall_progress: float = 0.0
    epoch_progress: float = 0.0
    batch_progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    current_batch: int = 0
    total_batches: int = 0
    phase: str = "training"  # training, validation, completed, error
    message: str = ""
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

class TrainingProgressBridge:
    """Bridge untuk menghubungkan training loop dengan UI Progress Tracker"""
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None,
                 progress_callback: Optional[Callable] = None,
                 metrics_callback: Optional[Callable] = None):
        """
        Initialize progress bridge
        
        Args:
            ui_components: Dictionary UI components (untuk UI integration)
            progress_callback: Callback function untuk progress updates
            metrics_callback: Callback function untuk metrics updates
        """
        self.ui_components = ui_components or {}
        self.progress_callback = progress_callback
        self.metrics_callback = metrics_callback
        
        # Progress state
        self.current_progress = ProgressData()
        self.start_time = None
        self.epoch_start_time = None
        self.batch_start_time = None
        
        # Timing statistics
        self.timing_stats = {
            'total_time': 0.0,
            'epoch_times': [],
            'batch_times': [],
            'avg_epoch_time': 0.0,
            'avg_batch_time': 0.0
        }
    
    def start_training(self, total_epochs: int, total_batches_per_epoch: int = None,
                      operation_name: str = "Model Training") -> None:
        """
        Initialize training progress tracking
        
        Args:
            total_epochs: Total number of epochs
            total_batches_per_epoch: Total batches per epoch (optional)
            operation_name: Display name untuk operation
        """
        self.start_time = time.time()
        self.current_progress.total_epochs = total_epochs
        self.current_progress.total_batches = total_batches_per_epoch or 0
        self.current_progress.phase = "training"
        self.current_progress.message = f"🚀 Memulai {operation_name}..."
        
        # Initialize UI progress tracker
        self._update_ui_progress_tracker("show", operation_name, ["Training", "Validation", "Completion"])
        
        # Send initial progress update
        self._send_progress_update()
    
    def start_epoch(self, epoch: int, phase: str = "training") -> None:
        """
        Mark start of new epoch
        
        Args:
            epoch: Current epoch number (0-based)
            phase: Current phase (training/validation)
        """
        self.epoch_start_time = time.time()
        self.current_progress.current_epoch = epoch + 1  # 1-based untuk display
        self.current_progress.phase = phase
        self.current_progress.batch_progress = 0.0
        self.current_progress.current_batch = 0
        
        # Calculate overall progress
        self.current_progress.overall_progress = (epoch / self.current_progress.total_epochs) * 100
        
        phase_emoji = "🔥" if phase == "training" else "📊"
        self.current_progress.message = f"{phase_emoji} Epoch {self.current_progress.current_epoch}/{self.current_progress.total_epochs} - {phase.capitalize()}"
        
        self._send_progress_update()
    
    def update_batch(self, batch_idx: int, total_batches: int, loss: float = None,
                    metrics: Dict[str, float] = None) -> None:
        """
        Update batch progress
        
        Args:
            batch_idx: Current batch index (0-based)
            total_batches: Total batches dalam epoch
            loss: Current batch loss
            metrics: Additional metrics
        """
        self.batch_start_time = time.time()
        self.current_progress.current_batch = batch_idx + 1  # 1-based display
        self.current_progress.total_batches = total_batches
        
        # Calculate batch progress within epoch
        self.current_progress.batch_progress = ((batch_idx + 1) / total_batches) * 100
        
        # Calculate epoch progress (training + validation combined)
        epoch_base_progress = (self.current_progress.current_epoch - 1) / self.current_progress.total_epochs
        epoch_increment = (1 / self.current_progress.total_epochs) * (self.current_progress.batch_progress / 100)
        
        if self.current_progress.phase == "validation":
            # Validation is second half of epoch progress
            epoch_increment = (1 / self.current_progress.total_epochs) * (0.5 + (self.current_progress.batch_progress / 100) * 0.5)
        elif self.current_progress.phase == "training":
            # Training is first half of epoch progress
            epoch_increment = (1 / self.current_progress.total_epochs) * ((self.current_progress.batch_progress / 100) * 0.5)
        
        self.current_progress.overall_progress = (epoch_base_progress + epoch_increment) * 100
        
        # Update message dengan batch info
        phase_emoji = "🔥" if self.current_progress.phase == "training" else "📊"
        loss_str = f", Loss: {loss:.4f}" if loss is not None else ""
        self.current_progress.message = f"{phase_emoji} Epoch {self.current_progress.current_epoch}/{self.current_progress.total_epochs} - Batch {self.current_progress.current_batch}/{total_batches}{loss_str}"
        
        # Update metrics
        if metrics:
            self.current_progress.metrics.update(metrics)
        if loss is not None:
            self.current_progress.metrics['current_loss'] = loss
        
        # Send updates
        self._send_progress_update()
        if self.current_progress.metrics:
            self._send_metrics_update()
    
    def complete_epoch(self, epoch_metrics: Dict[str, float] = None) -> None:
        """
        Mark completion of epoch
        
        Args:
            epoch_metrics: Metrics dari completed epoch
        """
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.timing_stats['epoch_times'].append(epoch_time)
            self.timing_stats['avg_epoch_time'] = sum(self.timing_stats['epoch_times']) / len(self.timing_stats['epoch_times'])
        
        # Update progress
        self.current_progress.epoch_progress = 100.0
        self.current_progress.batch_progress = 100.0
        
        # Update metrics dengan epoch results
        if epoch_metrics:
            self.current_progress.metrics.update(epoch_metrics)
            self._send_metrics_update()
        
        # Estimate remaining time
        if self.timing_stats['avg_epoch_time'] > 0:
            remaining_epochs = self.current_progress.total_epochs - self.current_progress.current_epoch
            estimated_remaining = remaining_epochs * self.timing_stats['avg_epoch_time']
            self.current_progress.metrics['estimated_remaining_time'] = estimated_remaining
    
    def complete_training(self, final_metrics: Dict[str, float] = None,
                         best_checkpoint: str = None) -> None:
        """
        Mark completion of training
        
        Args:
            final_metrics: Final training metrics
            best_checkpoint: Path ke best checkpoint
        """
        self.current_progress.overall_progress = 100.0
        self.current_progress.epoch_progress = 100.0
        self.current_progress.batch_progress = 100.0
        self.current_progress.phase = "completed"
        self.current_progress.message = "✅ Training selesai!"
        
        # Calculate total time
        if self.start_time:
            total_time = time.time() - self.start_time
            self.timing_stats['total_time'] = total_time
            self.current_progress.metrics['total_training_time'] = total_time
        
        # Add final metrics
        if final_metrics:
            self.current_progress.metrics.update(final_metrics)
        
        if best_checkpoint:
            self.current_progress.metrics['best_checkpoint'] = best_checkpoint
        
        # Send final updates
        self._send_progress_update()
        self._send_metrics_update()
        
        # Complete UI progress tracker
        self._update_ui_progress_tracker("complete", "Training selesai dengan sukses!")
    
    def training_error(self, error_message: str, exception: Exception = None) -> None:
        """
        Handle training error
        
        Args:
            error_message: Error message untuk display
            exception: Exception object (optional)
        """
        self.current_progress.phase = "error"
        self.current_progress.message = f"❌ Error: {error_message}"
        
        if exception:
            self.current_progress.metrics['error_type'] = type(exception).__name__
            self.current_progress.metrics['error_details'] = str(exception)
        
        # Send error update
        self._send_progress_update()
        
        # Update UI progress tracker
        self._update_ui_progress_tracker("error", error_message)
    
    def _send_progress_update(self) -> None:
        """Send progress update ke callbacks"""
        if self.progress_callback:
            try:
                # Format data untuk callback
                progress_data = {
                    'overall_progress': self.current_progress.overall_progress,
                    'epoch_progress': self.current_progress.epoch_progress,
                    'batch_progress': self.current_progress.batch_progress,
                    'current_epoch': self.current_progress.current_epoch,
                    'total_epochs': self.current_progress.total_epochs,
                    'current_batch': self.current_progress.current_batch,
                    'total_batches': self.current_progress.total_batches,
                    'phase': self.current_progress.phase,
                    'message': self.current_progress.message,
                    'metrics': self.current_progress.metrics.copy()
                }
                
                self.progress_callback(progress_data)
            except Exception as e:
                # Silent fail untuk avoid disrupting training
                pass
        
        # Update UI progress tracker
        self._update_ui_tracker_progress()
    
    def _send_metrics_update(self) -> None:
        """Send metrics update ke callback"""
        if self.metrics_callback and self.current_progress.metrics:
            try:
                metrics_data = {
                    'epoch': self.current_progress.current_epoch,
                    'phase': self.current_progress.phase,
                    'metrics': self.current_progress.metrics.copy(),
                    'timing': self.timing_stats.copy()
                }
                
                self.metrics_callback(metrics_data)
            except Exception as e:
                # Silent fail
                pass
    
    def _update_ui_progress_tracker(self, action: str, message: str = "", steps: List[str] = None) -> None:
        """Update UI Progress Tracker jika tersedia"""
        tracker = self.ui_components.get('progress_tracker')
        if not tracker:
            return
        
        try:
            if action == "show":
                tracker.show(message, steps or [])
            elif action == "complete":
                tracker.complete(message)
            elif action == "error":
                tracker.error(message)
        except Exception:
            # Silent fail
            pass
    
    def _update_ui_tracker_progress(self) -> None:
        """Update progress bars dalam UI tracker"""
        tracker = self.ui_components.get('progress_tracker')
        if not tracker:
            return
        
        try:
            # Update overall progress
            tracker.update_overall(
                self.current_progress.overall_progress,
                f"Epoch {self.current_progress.current_epoch}/{self.current_progress.total_epochs}"
            )
            
            # Update current progress (batch level)
            if self.current_progress.total_batches > 0:
                tracker.update_current(
                    self.current_progress.batch_progress,
                    f"Batch {self.current_progress.current_batch}/{self.current_progress.total_batches}"
                )
        except Exception:
            # Silent fail
            pass
    
    def get_timing_summary(self) -> Dict[str, Any]:
        """Get timing statistics summary"""
        return {
            'total_time': self.timing_stats['total_time'],
            'avg_epoch_time': self.timing_stats['avg_epoch_time'],
            'completed_epochs': len(self.timing_stats['epoch_times']),
            'estimated_total_time': self.timing_stats['avg_epoch_time'] * self.current_progress.total_epochs if self.timing_stats['avg_epoch_time'] > 0 else 0
        }
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current progress state"""
        return {
            'progress': {
                'overall': self.current_progress.overall_progress,
                'epoch': self.current_progress.epoch_progress,
                'batch': self.current_progress.batch_progress
            },
            'current': {
                'epoch': self.current_progress.current_epoch,
                'batch': self.current_progress.current_batch,
                'phase': self.current_progress.phase
            },
            'total': {
                'epochs': self.current_progress.total_epochs,
                'batches': self.current_progress.total_batches
            },
            'message': self.current_progress.message,
            'metrics': self.current_progress.metrics.copy(),
            'timing': self.get_timing_summary()
        }

# Convenience functions
def create_training_progress_bridge(ui_components: Dict[str, Any] = None,
                                   progress_callback: Callable = None,
                                   metrics_callback: Callable = None) -> TrainingProgressBridge:
    """Factory function untuk create training progress bridge"""
    return TrainingProgressBridge(ui_components, progress_callback, metrics_callback)

def create_simple_progress_callback(ui_components: Dict[str, Any]) -> Callable:
    """Create simple progress callback untuk basic UI updates"""
    def progress_callback(progress_data: Dict[str, Any]) -> None:
        message = progress_data.get('message', 'Processing...')
        overall = progress_data.get('overall_progress', 0)
        
        # Log ke UI jika ada logger
        logger = ui_components.get('logger')
        if logger:
            logger.info(f"📊 {message} ({overall:.1f}%)")
    
    return progress_callback