# File: smartcash/handlers/model/observers/colab_observer.py
# Author: Alfrida Sabar
# Deskripsi: Observer khusus untuk monitoring dan visualisasi di Google Colab

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.handlers.model.observers.base_observer import BaseObserver

class ColabObserver(BaseObserver):
    """
    Observer khusus untuk monitoring dan visualisasi di Google Colab.
    Menyediakan progress bar dan visualisasi real-time yang kompatibel dengan Colab.
    """
    
    def __init__(
        self, 
        logger: Optional[SmartCashLogger] = None,
        create_plots: bool = True,
        update_frequency: int = 1
    ):
        """
        Inisialisasi Colab observer.
        
        Args:
            logger: Custom logger (opsional)
            create_plots: Flag untuk membuat visualisasi (opsional)
            update_frequency: Frekuensi update visualisasi dalam epoch (opsional)
        """
        super().__init__(logger, "colab_observer")
        
        self.create_plots = create_plots
        self.update_frequency = update_frequency
        
        # Tracking progress
        self.progress_bar = None
        self.epoch_progress = None
        self.start_time = None
        self.total_epochs = None
        self.epoch_start_time = None
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epochs = []
        self.metrics = {}
        
        # Tracking plot
        self.fig = None
        self.axes = None
        self.plot_output = None
        
        # Setup colab progress widgets
        self._setup_colab_widgets()
    
    def _setup_colab_widgets(self) -> None:
        """Setup widget untuk monitoring di Colab."""
        try:
            from google.colab import output
            import ipywidgets as widgets
            from IPython.display import display, HTML
            
            # Buat HTML header
            display(HTML("""
            <style>
                .colab-training-header {
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 10px;
                    border-left: 5px solid #4285F4;
                }
                .colab-training-info {
                    font-family: monospace;
                    margin: 5px 0;
                }
                .training-metric {
                    display: inline-block;
                    margin-right: 15px;
                    padding: 5px;
                    border-radius: 3px;
                }
                .good-metric {
                    background-color: rgba(52, 168, 83, 0.15);
                }
                .warning-metric {
                    background-color: rgba(251, 188, 5, 0.15);
                }
            </style>
            <div class="colab-training-header">
                <h3>🚀 SmartCash Model Training</h3>
                <p>Monitor progres training secara real-time</p>
            </div>
            """))
            
            # Buat progress bar
            self.progress_bar = widgets.FloatProgress(
                value=0,
                min=0,
                max=100,
                description='Training:',
                bar_style='info',
                style={'description_width': '100px', 'bar_color': '#4285F4'},
                layout={'width': '60%'}
            )
            
            # Buat output widget untuk info training
            self.training_info = widgets.Output()
            
            # Buat output widget untuk metrik
            self.metric_output = widgets.Output()
            
            # Buat output widget untuk plot
            self.plot_output = widgets.Output()
            
            # Tampilkan widgets
            display(self.progress_bar)
            display(self.training_info)
            display(self.metric_output)
            display(self.plot_output)
            
            self.logger.info("🎛️ Colab widgets berhasil disetup")
        except Exception as e:
            self.logger.warning(f"⚠️ Tidak dapat membuat widgets di Colab: {str(e)}")
            self.progress_bar = None
    
    def update(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Update dari training process.
        
        Args:
            event: Nama event
            data: Data tambahan
        """
        data = data or {}
        
        if event == 'training_start':
            self._handle_training_start(data)
        elif event == 'training_end':
            self._handle_training_end(data)
        elif event == 'epoch_start':
            self._handle_epoch_start(data)
        elif event == 'epoch_end':
            self._handle_epoch_end(data)
        elif event == 'batch_end':
            self._handle_batch_end(data)
    
    def _handle_training_start(self, data: Dict[str, Any]) -> None:
        """Handle training start event."""
        self.start_time = time.time()
        self.total_epochs = data.get('epochs', 100)
        
        # Reset metrics
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epochs = []
        self.metrics = {}
        
        # Update progress bar max
        if self.progress_bar:
            self.progress_bar.max = self.total_epochs
            self.progress_bar.value = 0
        
        # Update training info
        if hasattr(self, 'training_info'):
            with self.training_info:
                self.training_info.clear_output(wait=True)
                print(f"🚀 Training dimulai: {self.total_epochs} epochs")
                print(f"⏱️ Mulai: {time.strftime('%H:%M:%S')}")
                if 'model' in data:
                    print(f"🤖 Model: {data.get('model').__class__.__name__}")
                if 'optimizer' in data:
                    print(f"⚙️ Optimizer: {data.get('optimizer').__class__.__name__}")
        
        # Inisialisasi plot
        if self.create_plots:
            self._setup_plots()
    
    def _handle_training_end(self, data: Dict[str, Any]) -> None:
        """Handle training end event."""
        training_time = time.time() - self.start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Update progress bar
        if self.progress_bar:
            self.progress_bar.bar_style = 'success'
            self.progress_bar.value = self.total_epochs
        
        # Update training info
        if hasattr(self, 'training_info'):
            with self.training_info:
                self.training_info.clear_output(wait=True)
                print(f"✅ Training selesai dalam {int(hours)}h {int(minutes)}m {int(seconds)}s")
                print(f"📊 Epochs selesai: {len(self.epochs)}/{self.total_epochs}")
                if self.val_losses:
                    print(f"📉 Best val loss: {min(self.val_losses):.4f}")
                
                # Tampilkan early stopping info jika ada
                if 'early_stopped' in data and data['early_stopped']:
                    print(f"🛑 Early stopping pada epoch {data.get('epoch', 0)}")
                
                # Tampilkan checkpoint path jika ada
                if 'best_checkpoint_path' in data:
                    print(f"💾 Best checkpoint: {data['best_checkpoint_path']}")
    
    def _handle_epoch_start(self, data: Dict[str, Any]) -> None:
        """Handle epoch start event."""
        epoch = data.get('epoch', 0)
        
        # Simpan waktu mulai epoch
        self.epoch_start_time = time.time()
        
        # Update epoch progress
        if self.progress_bar:
            self.progress_bar.value = epoch
    
    def _handle_epoch_end(self, data: Dict[str, Any]) -> None:
        """Handle epoch end event."""
        epoch = data.get('epoch', 0)
        metrics = data.get('metrics', {})
        
        # Hitung waktu epoch
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        # Tambahkan metrik ke history
        self.epochs.append(epoch)
        
        if 'train_loss' in metrics:
            self.train_losses.append(metrics['train_loss'])
        
        if 'val_loss' in metrics:
            self.val_losses.append(metrics['val_loss'])
        
        if 'learning_rate' in metrics:
            self.learning_rates.append(metrics['learning_rate'])
        
        # Update progress bar
        if self.progress_bar:
            self.progress_bar.value = epoch + 1
        
        # Update metric output
        if hasattr(self, 'metric_output'):
            with self.metric_output:
                self.metric_output.clear_output(wait=True)
                
                # Tampilkan metrik dalam format compact
                print(f"⏱️ Epoch {epoch} ({epoch_time:.2f}s)")
                print(f"📉 Train loss: {metrics.get('train_loss', 0):.4f} | Val loss: {metrics.get('val_loss', 0):.4f}")
                
                # Tampilkan metrik tambahan
                other_metrics = []
                for k, v in metrics.items():
                    if k not in ['train_loss', 'val_loss', 'learning_rate'] and isinstance(v, (int, float)):
                        other_metrics.append(f"{k}: {v:.4f}")
                
                if other_metrics:
                    print(f"📊 {' | '.join(other_metrics)}")
        
        # Update plot jika perlu
        if self.create_plots and epoch % self.update_frequency == 0:
            self._update_plots()
    
    def _handle_batch_end(self, data: Dict[str, Any]) -> None:
        """Handle batch end event."""
        # Tidak perlu implementasi khusus di sini karena progress per batch
        # sudah dihandle oleh tqdm di ModelTrainer
        pass
    
    def _setup_plots(self) -> None:
        """Setup plots untuk visualisasi training."""
        if not hasattr(self, 'plot_output') or self.plot_output is None:
            return
            
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            
            # Buat figure dengan 2 subplot
            self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Setup subplot untuk loss
            self.axes[0].set_title('Training & Validation Loss')
            self.axes[0].set_xlabel('Epoch')
            self.axes[0].set_ylabel('Loss')
            self.axes[0].grid(True, alpha=0.3)
            
            # Setup subplot untuk learning rate
            self.axes[1].set_title('Learning Rate')
            self.axes[1].set_xlabel('Epoch')
            self.axes[1].set_ylabel('Learning Rate')
            self.axes[1].set_yscale('log')
            self.axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def _update_plots(self) -> None:
        """Update plots dengan data terbaru."""
        if not hasattr(self, 'plot_output') or self.plot_output is None or self.fig is None:
            return
            
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            
            # Update loss plot
            self.axes[0].clear()
            self.axes[0].set_title('Training & Validation Loss')
            self.axes[0].set_xlabel('Epoch')
            self.axes[0].set_ylabel('Loss')
            
            if self.epochs and self.train_losses:
                self.axes[0].plot(self.epochs, self.train_losses, 'b-', label='Train Loss')
            
            if self.epochs and self.val_losses:
                self.axes[0].plot(self.epochs, self.val_losses, 'r-', label='Val Loss')
                
                # Tandai loss terbaik
                if self.val_losses:
                    best_idx = np.argmin(self.val_losses)
                    best_epoch = self.epochs[best_idx]
                    best_val_loss = self.val_losses[best_idx]
                    
                    self.axes[0].scatter([best_epoch], [best_val_loss], c='gold', s=100, zorder=5, edgecolor='k')
                    self.axes[0].annotate(
                        f'Best: {best_val_loss:.4f}',
                        (best_epoch, best_val_loss),
                        xytext=(5, 5),
                        textcoords='offset points',
                        backgroundcolor='white',
                        fontsize=8
                    )
            
            self.axes[0].legend()
            self.axes[0].grid(True, alpha=0.3)
            
            # Update learning rate plot
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