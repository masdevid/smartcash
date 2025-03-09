# File: smartcash/handlers/model/observers/metrics_observer.py
# Author: Alfrida Sabar
# Deskripsi: Observer untuk monitoring dan tracking metrik training

import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.handlers.model.observers.base_observer import BaseObserver

class MetricsObserver(BaseObserver):
    """
    Observer untuk monitoring dan tracking metrik training.
    Menyimpan dan memvisualisasikan metrik selama training.
    """
    
    def __init__(
        self, 
        output_dir: str = "runs/train",
        logger: Optional[SmartCashLogger] = None,
        experiment_name: Optional[str] = None,
        save_metrics: bool = True,
        visualize: bool = True
    ):
        """
        Inisialisasi metrics observer.
        
        Args:
            output_dir: Direktori output untuk menyimpan metrik
            logger: Custom logger (opsional)
            experiment_name: Nama eksperimen (opsional)
            save_metrics: Flag untuk menyimpan metrik ke disk
            visualize: Flag untuk visualisasi metrik
        """
        super().__init__(logger, "metrics_observer")
        
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or f"exp_{int(time.time())}"
        self.save_metrics = save_metrics
        self.visualize = visualize
        
        # Buat direktori output
        self.metrics_dir = self.output_dir / "metrics" / self.experiment_name
        if self.save_metrics:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epochs': [],
            'batch_metrics': [],
            'epoch_metrics': []
        }
        
        # Track timestamp
        self.start_time = None
        self.epoch_start_times = {}
        
        self.logger.info(f"🔍 MetricsObserver diinisialisasi (save_metrics={save_metrics}, visualize={visualize})")
    
    def update(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Update dari training process.
        
        Args:
            event: Nama event ('training_start', 'epoch_end', dll)
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
        self.logger.info(f"🚀 Mulai tracking metrik untuk eksperimen '{self.experiment_name}'")
        
        # Reset metrics history
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epochs': [],
            'batch_metrics': [],
            'epoch_metrics': []
        }
        
        # Save config if provided
        if 'config' in data:
            if self.save_metrics:
                config_path = self.metrics_dir / "config.json"
                with open(config_path, 'w') as f:
                    json.dump(data['config'], f, indent=2)
    
    def _handle_training_end(self, data: Dict[str, Any]) -> None:
        """Handle training end event."""
        training_time = time.time() - self.start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.logger.info(
            f"✅ Training selesai dalam {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
            f"   • Total epoch: {len(self.metrics_history['epochs'])}\n"
            f"   • Final loss: {self.metrics_history['train_loss'][-1]:.4f} (train), "
            f"{self.metrics_history['val_loss'][-1]:.4f} (val)"
        )
        
        # Add final metrics
        if 'metrics' in data:
            self.metrics_history['final_metrics'] = data['metrics']
        
        # Save metrics history
        if self.save_metrics:
            self._save_metrics_history()
        
        # Create final visualizations
        if self.visualize:
            self._create_visualizations()
    
    def _handle_epoch_start(self, data: Dict[str, Any]) -> None:
        """Handle epoch start event."""
        epoch = data.get('epoch', 0)
        self.epoch_start_times[epoch] = time.time()
    
    def _handle_epoch_end(self, data: Dict[str, Any]) -> None:
        """Handle epoch end event."""
        epoch = data.get('epoch', 0)
        metrics = data.get('metrics', {})
        
        # Record epoch
        self.metrics_history['epochs'].append(epoch)
        
        # Record losses
        train_loss = metrics.get('train_loss', None)
        if train_loss is not None:
            self.metrics_history['train_loss'].append(train_loss)
            
        val_loss = metrics.get('val_loss', None)
        if val_loss is not None:
            self.metrics_history['val_loss'].append(val_loss)
        
        # Record learning rate
        lr = metrics.get('learning_rate', None)
        if lr is not None:
            self.metrics_history['learning_rates'].append(lr)
        
        # Save all metrics
        self.metrics_history['epoch_metrics'].append({
            'epoch': epoch,
            **metrics
        })
        
        # Calculate epoch time
        if epoch in self.epoch_start_times:
            epoch_time = time.time() - self.epoch_start_times[epoch]
            
            self.logger.info(
                f"📊 Epoch {epoch} selesai dalam {epoch_time:.2f}s:\n"
                f"   • Train loss: {train_loss:.4f}\n"
                f"   • Val loss: {val_loss:.4f}\n"
                f"   • Learning rate: {lr:.6f}"
            )
        
        # Create periodic visualizations (every 5 epochs or at the end)
        if self.visualize and (epoch % 5 == 0 or epoch == data.get('total_epochs', 0) - 1):
            self._create_visualizations()
    
    def _handle_batch_end(self, data: Dict[str, Any]) -> None:
        """Handle batch end event."""
        # Only record every 10th batch to avoid too much data
        batch_idx = data.get('batch_idx', 0)
        if batch_idx % 10 == 0:
            self.metrics_history['batch_metrics'].append({
                'batch_idx': batch_idx,
                'epoch': data.get('epoch', 0),
                'loss': data.get('loss', 0),
                **data.get('metrics', {})
            })
    
    def _save_metrics_history(self) -> None:
        """Save metrics history to disk."""
        if not self.save_metrics:
            return
            
        # Convert to clean JSON-serializable format
        clean_history = {}
        for key, value in self.metrics_history.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                # List of dicts
                clean_list = []
                for item in value:
                    clean_item = {}
                    for k, v in item.items():
                        if isinstance(v, (np.float32, np.float64)):
                            clean_item[k] = float(v)
                        elif isinstance(v, (np.int32, np.int64)):
                            clean_item[k] = int(v)
                        else:
                            clean_item[k] = v
                    clean_list.append(clean_item)
                clean_history[key] = clean_list
            elif isinstance(value, list) and value and isinstance(value[0], (np.float32, np.float64)):
                # List of floats
                clean_history[key] = [float(x) for x in value]
            elif isinstance(value, list) and value and isinstance(value[0], (np.int32, np.int64)):
                # List of ints
                clean_history[key] = [int(x) for x in value]
            else:
                clean_history[key] = value
        
        # Save to JSON
        metrics_path = self.metrics_dir / "metrics_history.json"
        with open(metrics_path, 'w') as f:
            json.dump(clean_history, f, indent=2)
        
        # Save to CSV for easier analysis
        if self.metrics_history['epoch_metrics']:
            df = pd.DataFrame(self.metrics_history['epoch_metrics'])
            csv_path = self.metrics_dir / "epoch_metrics.csv"
            df.to_csv(csv_path, index=False)
            
        self.logger.info(f"💾 Metrik disimpan di {self.metrics_dir}")
    
    def _create_visualizations(self) -> None:
        """Create visualizations of metrics."""
        if not self.visualize or not self.metrics_history['epochs']:
            return
            
        try:
            # Create figures directory
            figs_dir = self.metrics_dir / "figures"
            figs_dir.mkdir(exist_ok=True)
            
            # Training curve
            if self.metrics_history['train_loss'] and self.metrics_history['val_loss']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics_history['epochs'], self.metrics_history['train_loss'], 'b-', label='Train Loss')
                plt.plot(self.metrics_history['epochs'], self.metrics_history['val_loss'], 'r-', label='Val Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(figs_dir / "training_curve.png", dpi=150)
                plt.close()
            
            # Learning rate curve
            if self.metrics_history['learning_rates']:
                plt.figure(figsize=(10, 6))
                plt.plot(self.metrics_history['epochs'], self.metrics_history['learning_rates'], 'g-')
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.title('Learning Rate Schedule')
                plt.grid(True, alpha=0.3)
                plt.yscale('log')
                plt.tight_layout()
                plt.savefig(figs_dir / "lr_schedule.png", dpi=150)
                plt.close()
            
            # Additional metrics if available
            if self.metrics_history['epoch_metrics']:
                metrics_df = pd.DataFrame(self.metrics_history['epoch_metrics'])
                
                # Plot metrics other than loss and lr
                for col in metrics_df.columns:
                    if col not in ['epoch', 'train_loss', 'val_loss', 'learning_rate']:
                        if metrics_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                            plt.figure(figsize=(10, 6))
                            plt.plot(metrics_df['epoch'], metrics_df[col], 'o-')
                            plt.xlabel('Epoch')
                            plt.ylabel(col)
                            plt.title(f'{col} vs Epoch')
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            plt.savefig(figs_dir / f"{col}_curve.png", dpi=150)
                            plt.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membuat visualisasi: {str(e)}")
    
    def get_last_metrics(self) -> Dict[str, Any]:
        """
        Dapatkan metrik terakhir.
        
        Returns:
            Dict metrik terakhir
        """
        if not self.metrics_history['epoch_metrics']:
            return {}
            
        return self.metrics_history['epoch_metrics'][-1]
    
    def get_best_metrics(self, key: str = 'val_loss', mode: str = 'min') -> Dict[str, Any]:
        """
        Dapatkan metrik terbaik berdasarkan key tertentu.
        
        Args:
            key: Kunci metrik untuk dioptimalkan
            mode: Mode optimasi ('min' atau 'max')
            
        Returns:
            Dict metrik terbaik
        """
        if not self.metrics_history['epoch_metrics']:
            return {}
        
        metrics_df = pd.DataFrame(self.metrics_history['epoch_metrics'])
        
        if key not in metrics_df.columns:
            self.logger.warning(f"⚠️ Metrik '{key}' tidak ditemukan")
            return {}
        
        idx = metrics_df[key].idxmin() if mode == 'min' else metrics_df[key].idxmax()
        return self.metrics_history['epoch_metrics'][idx]