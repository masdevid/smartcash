# File: smartcash/handlers/model/observers/metrics_observer.py
# Author: Alfrida Sabar
# Deskripsi: Observer untuk monitoring dan tracking metrik training

import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

from smartcash.handlers.model.observers.base_observer import BaseObserver

class MetricsObserver(BaseObserver):
    """Observer untuk monitoring dan tracking metrik training."""
    
    def __init__(self, output_dir="runs/train", logger=None, experiment_name=None, save_metrics=True, visualize=True):
        """Inisialisasi metrics observer."""
        super().__init__(logger, "metrics_observer")
        
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or f"exp_{int(time.time())}"
        self.save_metrics = save_metrics
        self.visualize = visualize
        
        # Setup output directory
        self.metrics_dir = self.output_dir / "metrics" / self.experiment_name
        if save_metrics:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi metrics history
        self.metrics_history = {
            'train_loss': [], 'val_loss': [], 'learning_rates': [],
            'epochs': [], 'batch_metrics': [], 'epoch_metrics': []
        }
        
        # Timestamp tracking
        self.start_time = None
        self.epoch_times = {}
        
        self.logger.info(f"🔍 MetricsObserver diinisialisasi ({self.experiment_name})")
    
    def update(self, event: str, data: Dict[str, Any] = None) -> None:
        """Update dari proses training."""
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
    
    def _handle_training_start(self, data):
        """Handle training start event."""
        self.start_time = time.time()
        
        # Reset metrics
        self.metrics_history = {
            'train_loss': [], 'val_loss': [], 'learning_rates': [],
            'epochs': [], 'batch_metrics': [], 'epoch_metrics': []
        }
        
        # Save config jika ada
        if 'config' in data and self.save_metrics:
            with open(self.metrics_dir / "config.json", 'w') as f:
                json.dump(data['config'], f, indent=2)
    
    def _handle_training_end(self, data):
        """Handle training end event."""
        if not self.start_time:
            return
            
        duration = time.time() - self.start_time
        h, m = int(duration // 3600), int((duration % 3600) // 60)
        
        # Log ringkasan
        last_train = self.metrics_history['train_loss'][-1] if self.metrics_history['train_loss'] else 'N/A'
        last_val = self.metrics_history['val_loss'][-1] if self.metrics_history['val_loss'] else 'N/A'
        self.logger.info(
            f"✅ Training selesai dalam {h}h {m}m\n"
            f"   • Epochs: {len(self.metrics_history['epochs'])}\n"
            f"   • Final loss: {last_train} (train), {last_val} (val)"
        )
        
        # Save metrics
        if self.save_metrics:
            self._save_metrics_history()
        
        # Create visualizations
        if self.visualize:
            self._create_visualizations()
    
    def _handle_epoch_start(self, data):
        """Handle epoch start event."""
        epoch = data.get('epoch', 0)
        self.epoch_times[epoch] = time.time()
    
    def _handle_epoch_end(self, data):
        """Handle epoch end event."""
        epoch = data.get('epoch', 0)
        metrics = data.get('metrics', {})
        
        # Record metrics
        self.metrics_history['epochs'].append(epoch)
        
        # Core metrics
        for key in ['train_loss', 'val_loss', 'learning_rate']:
            if key in metrics:
                target_key = 'learning_rates' if key == 'learning_rate' else key
                self.metrics_history[target_key].append(metrics[key])
        
        # Complete metrics record
        self.metrics_history['epoch_metrics'].append({
            'epoch': epoch,
            **metrics
        })
        
        # Log untuk epoch
        if epoch in self.epoch_times:
            epoch_time = time.time() - self.epoch_times[epoch]
            epoch_metrics = [f"{k}: {v:.4f}" for k, v in metrics.items() 
                            if isinstance(v, (int, float)) and k in ['train_loss', 'val_loss']]
            self.logger.info(
                f"📊 Epoch {epoch} selesai dalam {epoch_time:.2f}s - " + ", ".join(epoch_metrics)
            )
        
        # Periodic visualization (every 5 epochs or at the end)
        if self.visualize and (epoch % 5 == 0 or epoch == metrics.get('total_epochs', 0) - 1):
            self._create_visualizations()
            
        # Auto-save metrics
        if self.save_metrics and epoch % 10 == 0:
            self._save_metrics_history()
    
    def _handle_batch_end(self, data):
        """Handle batch end event."""
        batch_idx = data.get('batch_idx', 0)
        # Hanya simpan setiap 10 batch untuk mengurangi ukuran data
        if batch_idx % 10 == 0:
            self.metrics_history['batch_metrics'].append({
                'batch_idx': batch_idx,
                'epoch': data.get('epoch', 0),
                'loss': data.get('loss', 0),
                **data.get('metrics', {})
            })
    
    def _save_metrics_history(self):
        """Save metrics ke disk."""
        if not self.save_metrics:
            return
            
        # Function to clean values (np.float to float, etc)
        def clean_value(v):
            if isinstance(v, (np.float32, np.float64)):
                return float(v)
            elif isinstance(v, (np.int32, np.int64)):
                return int(v)
            return v
        
        # Clean metrics
        clean_history = {}
        for key, value in self.metrics_history.items():
            if isinstance(value, list) and value:
                if isinstance(value[0], dict):
                    clean_history[key] = [
                        {k: clean_value(v) for k, v in item.items()}
                        for item in value
                    ]
                else:
                    clean_history[key] = [clean_value(v) for v in value]
            else:
                clean_history[key] = value
        
        # Save JSON and CSV
        metrics_path = self.metrics_dir / "metrics_history.json"
        with open(metrics_path, 'w') as f:
            json.dump(clean_history, f, indent=2)
            
        if self.metrics_history['epoch_metrics']:
            pd.DataFrame(self.metrics_history['epoch_metrics']).to_csv(
                self.metrics_dir / "epoch_metrics.csv", index=False
            )
    
    def _create_visualizations(self):
        """Create visualizations dari metrics."""
        if not self.visualize or not self.metrics_history['epochs']:
            return
            
        try:
            figs_dir = self.metrics_dir / "figures"
            figs_dir.mkdir(exist_ok=True)
            
            # Create loss plot
            if self.metrics_history['train_loss'] and self.metrics_history['val_loss']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics_history['epochs'], self.metrics_history['train_loss'], 'b-', label='Train Loss')
                plt.plot(self.metrics_history['epochs'], self.metrics_history['val_loss'], 'r-', label='Val Loss')
                
                # Highlight best point
                if len(self.metrics_history['val_loss']) > 1:
                    best_idx = self.metrics_history['val_loss'].index(min(self.metrics_history['val_loss']))
                    best_epoch = self.metrics_history['epochs'][best_idx]
                    best_val = self.metrics_history['val_loss'][best_idx]
                    plt.scatter([best_epoch], [best_val], c='gold', s=100, zorder=5, edgecolor='k')
                
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Progress')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(figs_dir / "training_curve.png", dpi=150)
                plt.close()
                
            # Create LR plot if available
            if self.metrics_history['learning_rates']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics_history['epochs'], self.metrics_history['learning_rates'], 'g-')
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.title('Learning Rate Schedule')
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
                plt.savefig(figs_dir / "lr_schedule.png", dpi=150)
                plt.close()
        except Exception as e:
            self.logger.warning(f"⚠️ Error visualisasi: {str(e)}")