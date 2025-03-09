# File: smartcash/handlers/model/observers/colab_observer.py
# Author: Alfrida Sabar
# Deskripsi: Observer khusus untuk Google Colab (diperbarui)

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

from smartcash.handlers.model.observers.model_observer_interface import ModelObserverInterface
from smartcash.utils.logger import SmartCashLogger, get_logger

class ColabObserver(ModelObserverInterface):
    """
    Observer khusus untuk monitoring dan visualisasi di Google Colab.
    Menggunakan interface observer terpadu.
    """
    
    def __init__(
        self, 
        logger: Optional[SmartCashLogger] = None, 
        create_plots: bool = True, 
        update_every: int = 1
    ):
        """
        Inisialisasi Colab observer.
        
        Args:
            logger: Custom logger (opsional)
            create_plots: Flag untuk membuat plot
            update_every: Interval update plot
        """
        super().__init__(name="colab_observer", logger=logger)
        
        self.create_plots = create_plots
        self.update_every = update_every
        
        # Tracking data
        self.start_time = None
        self.total_epochs = None
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Output widgets
        self.progress_bar = None
        self.training_info = None
        self.metric_output = None
        self.plot_output = None
        
        # Setup widgets jika di Colab
        if self._is_in_colab():
            self._setup_widgets()
            
        # Register event handlers
        self.register('training_start', self._handle_training_start)
        self.register('training_end', self._handle_training_end)
        self.register('epoch_start', self._handle_epoch_start)
        self.register('epoch_end', self._handle_epoch_end)
    
    def _is_in_colab(self):
        """Deteksi apakah running di Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _setup_widgets(self):
        """Setup widget untuk monitoring di Colab."""
        try:
            from IPython.display import display, HTML
            import ipywidgets as widgets
            
            # Setup style dan header
            display(HTML("""
                <style>
                    .metric-box { padding: 5px; margin: 5px 0; border-radius: 4px; display: inline-block; margin-right: 10px; }
                    .good-metric { background-color: rgba(52, 168, 83, 0.15); }
                    .warning-metric { background-color: rgba(251, 188, 5, 0.15); }
                </style>
                <div style="padding:10px; border-left:5px solid #4285F4; background:#f8f9fa; margin-bottom:10px;">
                    <h3>🚀 SmartCash Model Training</h3>
                </div>
            """))
            
            # Create widgets
            self.progress_bar = widgets.FloatProgress(
                value=0, min=0, max=100, description='Training:',
                bar_style='info', style={'description_width': '100px', 'bar_color': '#4285F4'}
            )
            self.training_info = widgets.Output()
            self.metric_output = widgets.Output()
            self.plot_output = widgets.Output()
            
            # Display widgets
            display(self.progress_bar)
            display(self.training_info)
            display(self.metric_output)
            display(self.plot_output)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Tidak dapat setup widgets Colab: {str(e)}")
            self.progress_bar = None
    
    def _handle_training_start(self, data: Dict[str, Any]):
        """
        Handle training start event.
        
        Args:
            data: Data event
        """
        self.start_time = time.time()
        self.total_epochs = data.get('epochs', 100)
        
        # Reset tracking data
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Update progress bar
        if self.progress_bar:
            self.progress_bar.max = self.total_epochs
            self.progress_bar.value = 0
        
        # Update info
        if self.training_info:
            with self.training_info:
                self.training_info.clear_output(wait=True)
                print(f"🚀 Training dimulai: {self.total_epochs} epochs")
                print(f"⏱️ Mulai: {time.strftime('%H:%M:%S')}")
                
                # Show model info if available
                model_type = data.get('model').__class__.__name__ if 'model' in data else 'Unknown'
                optimizer_type = data.get('optimizer').__class__.__name__ if 'optimizer' in data else 'Unknown'
                print(f"🤖 Model: {model_type}, Optimizer: {optimizer_type}")
        
        # Setup plots if needed
        if self.create_plots:
            self._setup_plots()
    
    def _handle_training_end(self, data: Dict[str, Any]):
        """
        Handle training end event.
        
        Args:
            data: Data event
        """
        duration = time.time() - self.start_time if self.start_time else 0
        h, m, s = int(duration // 3600), int((duration % 3600) // 60), int(duration % 60)
        
        # Update progress bar
        if self.progress_bar:
            self.progress_bar.bar_style = 'success'
            self.progress_bar.value = self.total_epochs
        
        # Update info
        if self.training_info:
            with self.training_info:
                self.training_info.clear_output(wait=True)
                print(f"✅ Training selesai dalam {h}h {m}m {s}s")
                print(f"📊 Epochs: {len(self.epochs)}/{self.total_epochs}")
                
                # Show best result
                if self.val_losses:
                    best_val = min(self.val_losses)
                    best_epoch = self.epochs[self.val_losses.index(best_val)]
                    print(f"🏆 Best val_loss: {best_val:.4f} (Epoch {best_epoch})")
                
                # Show early stopping info
                if data.get('early_stopped', False):
                    print(f"🛑 Early stopping pada epoch {data.get('epoch', 0)}")
                
                # Show checkpoint info
                if 'best_checkpoint_path' in data:
                    print(f"💾 Best checkpoint: {data['best_checkpoint_path']}")
    
    def _handle_epoch_start(self, data: Dict[str, Any]):
        """
        Handle epoch start event.
        
        Args:
            data: Data event
        """
        if self.progress_bar:
            self.progress_bar.value = data.get('epoch', 0)
    
    def _handle_epoch_end(self, data: Dict[str, Any]):
        """
        Handle epoch end event.
        
        Args:
            data: Data event
        """
        epoch = data.get('epoch', 0)
        metrics = data.get('metrics', {})
        
        # Update tracking data
        self.epochs.append(epoch)
        
        train_loss = metrics.get('train_loss')
        if train_loss is not None:
            self.train_losses.append(train_loss)
            
        val_loss = metrics.get('val_loss')
        if val_loss is not None:
            self.val_losses.append(val_loss)
            
        lr = metrics.get('learning_rate')
        if lr is not None:
            self.learning_rates.append(lr)
        
        # Update progress bar
        if self.progress_bar:
            self.progress_bar.value = epoch + 1
        
        # Update metric output
        if self.metric_output:
            with self.metric_output:
                self.metric_output.clear_output(wait=True)
                
                # Format metrics compactly
                print(f"⏱️ Epoch {epoch}")
                print(f"📉 Train: {metrics.get('train_loss', 0):.4f} | Val: {metrics.get('val_loss', 0):.4f}")
                
                # Other metrics
                other_metrics = ' | '.join([
                    f"{k}: {v:.4f}" for k, v in metrics.items() 
                    if k not in ['train_loss', 'val_loss', 'learning_rate'] and isinstance(v, (int, float))
                ])
                if other_metrics:
                    print(f"📊 {other_metrics}")
        
        # Update plot periodically
        if self.create_plots and epoch % self.update_every == 0:
            self._update_plots()
    
    def _setup_plots(self):
        """Setup plots untuk visualisasi."""
        if not self.plot_output:
            return
            
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            
            # Create figure with subplots
            self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Setup axes
            self.axes[0].set_title('Training & Validation Loss')
            self.axes[0].set_xlabel('Epoch')
            self.axes[0].set_ylabel('Loss')
            self.axes[0].grid(True, alpha=0.3)
            
            self.axes[1].set_title('Learning Rate')
            self.axes[1].set_xlabel('Epoch')
            self.axes[1].set_ylabel('Learning Rate')
            self.axes[1].set_yscale('log')
            self.axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def _update_plots(self):
        """Update plots dengan data terbaru."""
        if not self.plot_output or not hasattr(self, 'fig'):
            return
            
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            
            # Update loss plot
            self.axes[0].clear()
            self.axes[0].set_title('Training & Validation Loss')
            self.axes[0].set_xlabel('Epoch')
            self.axes[0].set_ylabel('Loss')
            
            if self.epochs and self.train_losses:
                self.axes[0].plot(self.epochs, self.train_losses, 'b-', label='Train')
            
            if self.epochs and self.val_losses:
                self.axes[0].plot(self.epochs, self.val_losses, 'r-', label='Validation')
                
                # Highlight best point
                if len(self.val_losses) > 1:
                    best_idx = np.argmin(self.val_losses)
                    best_epoch = self.epochs[best_idx]
                    best_val = self.val_losses[best_idx]
                    
                    self.axes[0].scatter([best_epoch], [best_val], c='gold', s=100, zorder=5, edgecolor='k')
                    self.axes[0].annotate(
                        f'Best: {best_val:.4f}', (best_epoch, best_val),
                        xytext=(5, 5), textcoords='offset points',
                        backgroundcolor='white', fontsize=8
                    )
            
            self.axes[0].legend()
            self.axes[0].grid(True, alpha=0.3)
            
            # Update LR plot
            self.axes[1].clear()
            self.axes[1].set_title('Learning Rate')
            self.axes[1].set_xlabel('Epoch')
            self.axes[1].set_ylabel('Learning Rate')
            self.axes[1].set_yscale('log')
            
            if self.epochs and self.learning_rates:
                self.axes[1].plot(self.epochs, self.learning_rates, 'g-')
                
            self.axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()