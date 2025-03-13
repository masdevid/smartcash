# File: smartcash/handlers/model/integration/metrics_observer_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Adapter untuk integrasi dengan MetricsObserver

from typing import Dict, Optional, Any, List, Union
from pathlib import Path
import json

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.utils.visualization import ExperimentVisualizer
from smartcash.handlers.model.integration.experiment_adapter import ExperimentAdapter

class MetricsObserverAdapter:
    """
    Adapter untuk integrasi dengan MetricsObserver.
    Memantau dan menyimpan metrik selama training dan eksperimen.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        experiment_name: str = "default",
        save_metrics: bool = True,
        visualize: bool = True
    ):
        """
        Inisialisasi metrics observer adapter.
        
        Args:
            output_dir: Direktori output untuk menyimpan metrik (opsional)
            logger: Custom logger (opsional)
            experiment_name: Nama eksperimen untuk identifikasi
            save_metrics: Flag untuk menyimpan metrik ke file
            visualize: Flag untuk visualisasi metrik
        """
        self.logger = logger or get_logger("metrics_observer")
        self.experiment_name = experiment_name
        self.save_metrics = save_metrics
        self.visualize = visualize
        
        # Setup output directory
        self.output_dir = Path(output_dir) if output_dir else Path("runs/train/metrics")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage untuk metrics history
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epochs': [],
            'metrics': {}
        }
        
        # Setup visualizer
        if visualize:
            self.visualizer = ExperimentVisualizer(
                output_dir=str(self.output_dir / "visualizations")
            )
        
        self.logger.info(f"📊 MetricsObserverAdapter diinisialisasi untuk '{experiment_name}'")
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update metrics history dengan nilai baru.
        
        Args:
            epoch: Nomor epoch saat ini
            train_loss: Loss pada training set
            val_loss: Loss pada validation set
            lr: Learning rate saat ini
            additional_metrics: Metrik tambahan (opsional)
        """
        # Update metrics history
        self.metrics_history['epochs'].append(epoch)
        self.metrics_history['train_loss'].append(float(train_loss))
        self.metrics_history['val_loss'].append(float(val_loss))
        self.metrics_history['learning_rate'].append(float(lr))
        
        # Update additional metrics
        if additional_metrics:
            for name, value in additional_metrics.items():
                if name not in self.metrics_history['metrics']:
                    self.metrics_history['metrics'][name] = []
                self.metrics_history['metrics'][name].append(float(value))
        
        # Save metrics jika diminta
        if self.save_metrics:
            self._save_metrics()
        
        # Visualisasi jika diminta
        if self.visualize and epoch > 0 and epoch % 5 == 0:
            self._visualize_metrics()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Dapatkan seluruh metrics history.
        
        Returns:
            Dictionary metrics history
        """
        return self.metrics_history
    
    def get_best_metrics(self, metric: str = 'val_loss', mode: str = 'min') -> Dict[str, Any]:
        """
        Dapatkan metrics terbaik berdasarkan metrik tertentu.
        
        Args:
            metric: Nama metrik untuk mencari nilai terbaik
            mode: 'min' untuk nilai terkecil, 'max' untuk nilai terbesar
            
        Returns:
            Dictionary dengan metrik terbaik dan indeksnya
        """
        if metric == 'val_loss':
            values = self.metrics_history['val_loss']
        elif metric == 'train_loss':
            values = self.metrics_history['train_loss']
        elif metric in self.metrics_history['metrics']:
            values = self.metrics_history['metrics'][metric]
        else:
            return {'best_value': None, 'best_epoch': None, 'best_index': None}
        
        if not values:
            return {'best_value': None, 'best_epoch': None, 'best_index': None}
        
        # Temukan indeks terbaik
        best_idx = min(range(len(values)), key=lambda i: values[i]) if mode == 'min' else max(range(len(values)), key=lambda i: values[i])
        best_value = values[best_idx]
        best_epoch = self.metrics_history['epochs'][best_idx] if best_idx < len(self.metrics_history['epochs']) else None
        
        return {
            'best_value': float(best_value),
            'best_epoch': int(best_epoch) if best_epoch is not None else None,
            'best_index': best_idx
        }
    
    def reset(self) -> None:
        """Reset metrics history."""
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epochs': [],
            'metrics': {}
        }
        self.logger.info("🔄 Metrics history direset")
    
    def _save_metrics(self) -> None:
        """Simpan metrics history ke file JSON."""
        try:
            metrics_path = self.output_dir / f"{self.experiment_name}_metrics.json"
            
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan metrics: {str(e)}")
    
    def _visualize_metrics(self) -> None:
        """Visualisasikan metrics history."""
        try:
            if hasattr(self, 'visualizer'):
                # Plot loss
                vis_path = self.visualizer.plot_training_metrics(
                    epochs=self.metrics_history['epochs'],
                    train_loss=self.metrics_history['train_loss'],
                    val_loss=self.metrics_history['val_loss'],
                    lr=self.metrics_history['learning_rate'],
                    additional_metrics=self.metrics_history['metrics'],
                    title=f"Training Metrics - {self.experiment_name}",
                    output_filename=f"{self.experiment_name}_training_metrics"
                )
                
                self.logger.info(f"📊 Visualisasi metrik telah diperbarui")
                
        except Exception as e:
            self.logger.error(f"❌ Gagal visualisasi metrics: {str(e)}")