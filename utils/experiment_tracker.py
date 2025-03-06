# File: smartcash/utils/experiment_tracker.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas untuk melacak dan menyimpan eksperimen training

import os
import json
import yaml
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from smartcash.utils.logger import SmartCashLogger

class ExperimentTracker:
    """
    Melacak dan menyimpan informasi eksperimen training untuk analisis performa.
    Mendukung visualisasi hasil dan perbandingan antara eksperimen.
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "runs/train/experiments",
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi experiment tracker.
        
        Args:
            experiment_name: Nama unik untuk eksperimen
            output_dir: Direktori untuk menyimpan data eksperimen
            logger: Logger kustom
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or SmartCashLogger(__name__)
        
        # Setup experiment directory
        self.experiment_dir = self.output_dir / self.experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Setup metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'epochs': [],
            'timestamp': []
        }
        
        self.config = {}
        self.best_metrics = {}
        self.start_time = None
        self.end_time = None
        
        self.logger.info(f"🧪 Experiment tracker diinisialisasi: {experiment_name}")
    
    def start_experiment(self, config: Dict[str, Any]) -> None:
        """
        Memulai eksperimen baru dengan konfigurasi tertentu.
        
        Args:
            config: Konfigurasi eksperimen
        """
        self.start_time = time.time()
        self.config = config
        
        # Reset metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'epochs': [],
            'timestamp': []
        }
        
        # Simpan konfigurasi
        config_path = self.experiment_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        self.logger.info(f"🚀 Eksperimen {self.experiment_name} dimulai")
    
    def log_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Mencatat metrik untuk satu epoch.
        
        Args:
            epoch: Nomor epoch
            train_loss: Training loss
            val_loss: Validation loss
            lr: Learning rate
            additional_metrics: Metrik tambahan (opsional)
        """
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        
        if lr is not None:
            self.metrics['lr'].append(lr)
            
        self.metrics['timestamp'].append(time.time())
        
        # Track metrik tambahan
        if additional_metrics:
            for key, value in additional_metrics.items():
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)
        
        # Update best metrics
        if not self.best_metrics or val_loss < self.best_metrics.get('val_loss', float('inf')):
            self.best_metrics = {
                'epoch': epoch,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'lr': lr
            }
            
            if additional_metrics:
                self.best_metrics.update(additional_metrics)
        
        # Auto-save setiap epoch
        self.save_metrics()
        
        # Log information
        self.logger.info(
            f"📈 Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"lr={lr:.6f}" if lr is not None else ""
        )
    
    def end_experiment(self, final_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Mengakhiri eksperimen dan menyimpan hasil akhir.
        
        Args:
            final_metrics: Metrik akhir eksperimen (opsional)
        """
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        # Simpan hasil akhir
        results = {
            'experiment_name': self.experiment_name,
            'duration': duration,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.fromtimestamp(self.end_time).isoformat(),
            'best_metrics': self.best_metrics
        }
        
        if final_metrics:
            results['final_metrics'] = final_metrics
            
        # Simpan hasil ke file
        results_path = self.experiment_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.success(
            f"✅ Eksperimen {self.experiment_name} selesai dalam {duration/60:.1f} menit\n"
            f"   Best val_loss: {self.best_metrics.get('val_loss', 'N/A'):.4f} "
            f"(Epoch {self.best_metrics.get('epoch', 'N/A')})"
        )
        
        # Generate laporan
        self.generate_report()
    
    def save_metrics(self) -> str:
        """
        Simpan metrik saat ini ke file.
        
        Returns:
            Path file metrik tersimpan
        """
        metrics_path = self.experiment_dir / 'metrics.json'
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
        return str(metrics_path)
    
    def load_metrics(self) -> Dict[str, List]:
        """
        Muat metrik dari file.
        
        Returns:
            Metrik eksperimen
        """
        metrics_path = self.experiment_dir / 'metrics.json'
        
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
            self.metrics = metrics
            
            # Calculate best metrics
            if self.metrics['val_loss']:
                best_idx = self.metrics['val_loss'].index(min(self.metrics['val_loss']))
                self.best_metrics = {
                    'epoch': self.metrics['epochs'][best_idx],
                    'val_loss': self.metrics['val_loss'][best_idx],
                    'train_loss': self.metrics['train_loss'][best_idx]
                }
                
                if 'lr' in self.metrics and best_idx < len(self.metrics['lr']):
                    self.best_metrics['lr'] = self.metrics['lr'][best_idx]
                    
        return self.metrics
    
    def plot_metrics(self, save_to_file: bool = True) -> Optional[plt.Figure]:
        """
        Plot metrik training dan validation loss.
        
        Args:
            save_to_file: Jika True, simpan plot ke file
            
        Returns:
            Plot figure atau None jika tidak ada metrik
        """
        if not self.metrics['epochs']:
            self.logger.warning("⚠️ Tidak ada metrik untuk diplot")
            return None
            
        # Ukuran figure tergantung apakah ada learning rate
        has_lr = 'lr' in self.metrics and len(self.metrics['lr']) > 0
        
        if has_lr:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
        
        # Plot loss
        ax1.plot(self.metrics['epochs'], self.metrics['train_loss'], 'b-', label='Training Loss')
        ax1.plot(self.metrics['epochs'], self.metrics['val_loss'], 'r-', label='Validation Loss')
        
        # Highlight best epoch
        if self.best_metrics:
            best_epoch = self.best_metrics['epoch']
            best_val_loss = self.best_metrics['val_loss']
            
            ax1.scatter([best_epoch], [best_val_loss], c='gold', s=100, zorder=5, edgecolor='k')
            ax1.annotate(
                f'Best: {best_val_loss:.4f}',
                (best_epoch, best_val_loss),
                xytext=(10, -20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='black')
            )
        
        ax1.set_title(f"Eksperimen: {self.experiment_name}")
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Plot learning rate jika tersedia
        if has_lr:
            ax2.plot(self.metrics['epochs'], self.metrics['lr'], 'g-', label='Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_yscale('log')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()
        
        plt.tight_layout()
        
        # Simpan plot jika diminta
        if save_to_file:
            plot_path = self.experiment_dir / 'training_plot.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"📊 Plot metrik tersimpan: {plot_path}")
        
        return fig
    
    def generate_report(self) -> str:
        """
        Generate laporan eksperimen.
        
        Returns:
            Path file laporan
        """
        report_path = self.experiment_dir / 'report.md'
        
        # Hitung durasi eksperimen
        duration = self.end_time - self.start_time if self.end_time else 0
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Generate plot
        self.plot_metrics(save_to_file=True)
        
        # Buat report
        with open(report_path, 'w') as f:
            f.write(f"# Laporan Eksperimen: {self.experiment_name}\n\n")
            
            # Informasi dasar
            f.write("## Informasi Eksperimen\n\n")
            f.write(f"- **Nama Eksperimen**: {self.experiment_name}\n")
            
            start_time_str = datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S") if self.start_time else "N/A"
            end_time_str = datetime.fromtimestamp(self.end_time).strftime("%Y-%m-%d %H:%M:%S") if self.end_time else "N/A"
            
            f.write(f"- **Waktu Mulai**: {start_time_str}\n")
            f.write(f"- **Waktu Selesai**: {end_time_str}\n")
            f.write(f"- **Durasi**: {duration_str}\n\n")
            
            # Konfigurasi
            f.write("## Konfigurasi\n\n")
            f.write("```yaml\n")
            yaml.dump(self.config, f, default_flow_style=False)
            f.write("```\n\n")
            
            # Best metrics
            f.write("## Hasil Terbaik\n\n")
            if self.best_metrics:
                f.write(f"- **Epoch**: {self.best_metrics.get('epoch', 'N/A')}\n")
                f.write(f"- **Validation Loss**: {self.best_metrics.get('val_loss', 'N/A'):.4f}\n")
                f.write(f"- **Training Loss**: {self.best_metrics.get('train_loss', 'N/A'):.4f}\n")
                
                for key, value in self.best_metrics.items():
                    if key not in ['epoch', 'val_loss', 'train_loss', 'lr']:
                        f.write(f"- **{key}**: {value}\n")
                
                f.write("\n")
            else:
                f.write("Tidak ada metrik tersedia.\n\n")
            
            # Plot
            f.write("## Visualisasi\n\n")
            f.write("![Training Metrics](training_plot.png)\n\n")
            
            # Metrics table
            f.write("## Tabel Metrik\n\n")
            if self.metrics['epochs']:
                f.write("| Epoch | Train Loss | Val Loss | Learning Rate |\n")
                f.write("|-------|------------|----------|---------------|\n")
                
                for i, epoch in enumerate(self.metrics['epochs']):
                    lr_value = self.metrics['lr'][i] if 'lr' in self.metrics and i < len(self.metrics['lr']) else '-'
                    f.write(f"| {epoch} | {self.metrics['train_loss'][i]:.4f} | {self.metrics['val_loss'][i]:.4f} | {lr_value} |\n")
            else:
                f.write("Tidak ada metrik tersedia.\n")
                
        self.logger.info(f"📝 Laporan eksperimen tersimpan: {report_path}")
        return str(report_path)
    
    @classmethod
    def list_experiments(cls, output_dir: str = "runs/train/experiments") -> List[str]:
        """
        Daftar semua eksperimen yang tersedia.
        
        Args:
            output_dir: Direktori eksperimen
            
        Returns:
            List nama eksperimen
        """
        output_path = Path(output_dir)
        
        if not output_path.exists():
            return []
            
        # Filter direktori saja
        experiments = [
            d.name for d in output_path.iterdir() 
            if d.is_dir() and (d / 'config.yaml').exists()
        ]
        
        return sorted(experiments)
    
    @classmethod
    def compare_experiments(
        cls,
        experiment_names: List[str],
        output_dir: str = "runs/train/experiments",
        save_to_file: bool = True
    ) -> Optional[plt.Figure]:
        """
        Bandingkan beberapa eksperimen.
        
        Args:
            experiment_names: List nama eksperimen
            output_dir: Direktori eksperimen
            save_to_file: Jika True, simpan hasil ke file
            
        Returns:
            Plot figure atau None jika tidak ada eksperimen
        """
        if not experiment_names:
            print("⚠️ Tidak ada eksperimen untuk dibandingkan")
            return None
            
        output_path = Path(output_dir)
        
        # Siapkan figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Track best metrics untuk setiap eksperimen
        best_metrics = {}
        
        # Plot setiap eksperimen
        for exp_name in experiment_names:
            exp_dir = output_path / exp_name
            metrics_path = exp_dir / 'metrics.json'
            
            if not metrics_path.exists():
                print(f"⚠️ Metrik tidak ditemukan untuk {exp_name}")
                continue
                
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
            if not metrics.get('epochs'):
                print(f"⚠️ Tidak ada metrik epoch untuk {exp_name}")
                continue
                
            # Plot validation loss
            ax.plot(metrics['epochs'], metrics['val_loss'], marker='o', markersize=3, 
                   label=f"{exp_name}")
            
            # Track best metrics
            if metrics['val_loss']:
                best_idx = metrics['val_loss'].index(min(metrics['val_loss']))
                best_metrics[exp_name] = {
                    'epoch': metrics['epochs'][best_idx],
                    'val_loss': metrics['val_loss'][best_idx]
                }
        
        ax.set_title("Perbandingan Validation Loss")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        
        # Simpan plot jika diminta
        if save_to_file:
            comparison_path = output_path / 'comparison.png'
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot perbandingan tersimpan: {comparison_path}")
            
        # Generate tabel perbandingan
        print("\n📋 Tabel Perbandingan:")
        
        # Prepare comparison table
        if best_metrics:
            comparison_data = {
                'Experiment': list(best_metrics.keys()),
                'Best Epoch': [m['epoch'] for m in best_metrics.values()],
                'Best Val Loss': [m['val_loss'] for m in best_metrics.values()]
            }
            
            try:
                import pandas as pd
                df = pd.DataFrame(comparison_data)
                df = df.sort_values('Best Val Loss')
                print(df.to_string(index=False))
            except ImportError:
                for exp_name, metrics in best_metrics.items():
                    print(f"• {exp_name}: Epoch {metrics['epoch']}, Val Loss {metrics['val_loss']:.4f}")
        
        return fig