"""
File: smartcash/model/config/experiment_config.py
Deskripsi: Konfigurasi untuk eksperimen pelatihan dan evaluasi model SmartCash
"""

import os
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from smartcash.model.config.model_config import ModelConfig

class ExperimentConfig:
    """
    Konfigurasi untuk eksperimen pelatihan dan evaluasi model SmartCash.
    Menyimpan parameter eksperimen, tracking metadata, dan komparasi hasil.
    """
    
    def __init__(
        self,
        name: str,
        base_config: Optional[ModelConfig] = None,
        experiment_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Inisialisasi konfigurasi eksperimen.
        
        Args:
            name: Nama eksperimen
            base_config: Konfigurasi model dasar (opsional)
            experiment_dir: Direktori untuk menyimpan hasil eksperimen (opsional)
            **kwargs: Parameter eksperimen tambahan
        """
        self.name = name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{name}_{self.timestamp}"
        
        # Setup eksperimen base config
        if base_config:
            self.base_config = base_config
        else:
            self.base_config = ModelConfig()
        
        # Setup experiment directory
        if experiment_dir:
            self.experiment_dir = Path(experiment_dir)
        else:
            self.experiment_dir = Path(self.base_config.get('checkpoint.save_dir', 'runs/train')) / 'experiments' / self.experiment_id
        
        # Buat direktori eksperimen
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Variabel untuk tracking metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'mAP': [],
            'f1': [],
            'epochs': []
        }
        
        # Parameter spesifik eksperimen
        self.parameters = {
            'experiment_id': self.experiment_id,
            'name': name,
            'timestamp': self.timestamp,
        }
        
        # Update dengan parameter tambahan
        self.parameters.update(kwargs)
        
        # Simpan konfigurasi awal
        self._save_initial_config()
    
    def _save_initial_config(self) -> None:
        """Simpan konfigurasi awal eksperimen."""
        config_path = self.experiment_dir / 'config.yaml'
        
        # Gabung konfigurasi dasar dengan parameter eksperimen
        full_config = {
            'base_config': self.base_config.config,
            'experiment': self.parameters
        }
        
        # Simpan ke file
        with open(config_path, 'w') as f:
            yaml.dump(full_config, f, default_flow_style=False)
    
    def log_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        test_loss: Optional[float] = None,
        **additional_metrics
    ) -> None:
        """
        Catat metrik untuk satu epoch.
        
        Args:
            epoch: Nomor epoch
            train_loss: Training loss
            val_loss: Validation loss (opsional)
            test_loss: Test loss (opsional)
            **additional_metrics: Metrik tambahan
        """
        # Tambahkan ke list metrics
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        
        if test_loss is not None:
            self.metrics['test_loss'].append(test_loss)
        
        # Tambahkan metrik tambahan
        for key, value in additional_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        # Simpan metrics
        self._save_metrics()
    
    def _save_metrics(self) -> None:
        """Simpan metrik ke file."""
        metrics_path = self.experiment_dir / 'metrics.json'
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load_metrics(self) -> Dict[str, List]:
        """
        Muat metrik dari file.
        
        Returns:
            Dictionary metrik
        """
        metrics_path = self.experiment_dir / 'metrics.json'
        
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
        
        return self.metrics
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """
        Dapatkan metrik terbaik berdasarkan validation loss.
        
        Returns:
            Dictionary metrik terbaik
        """
        if not self.metrics['epochs']:
            return {}
        
        # Cari indeks validation loss terendah
        if 'val_loss' in self.metrics and self.metrics['val_loss']:
            best_idx = self.metrics['val_loss'].index(min(self.metrics['val_loss']))
        # Fallback ke training loss jika val_loss tidak ada
        else:
            best_idx = self.metrics['train_loss'].index(min(self.metrics['train_loss']))
        
        # Kumpulkan metrik terbaik
        best_metrics = {
            'epoch': self.metrics['epochs'][best_idx],
            'train_loss': self.metrics['train_loss'][best_idx]
        }
        
        # Tambahkan metrik lain jika tersedia
        for key in self.metrics:
            if key not in ['epochs', 'train_loss'] and best_idx < len(self.metrics[key]):
                best_metrics[key] = self.metrics[key][best_idx]
        
        return best_metrics
    
    def save_model_checkpoint(self, model, optimizer, epoch: int, is_best: bool = False) -> str:
        """
        Simpan checkpoint model di direktori eksperimen.
        
        Args:
            model: Model yang akan disimpan
            optimizer: Optimizer yang digunakan
            epoch: Nomor epoch
            is_best: Flag apakah ini checkpoint terbaik
            
        Returns:
            Path checkpoint
        """
        checkpoint_dir = self.experiment_dir / 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Tentukan nama file
        filename = f"checkpoint_epoch{epoch}.pt"
        checkpoint_path = checkpoint_dir / filename
        
        # Simpan checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'experiment_id': self.experiment_id
        }
        
        # Tambahkan metrik
        for key, values in self.metrics.items():
            if key != 'epochs' and epoch <= len(values):
                idx = epoch - 1 if epoch <= len(values) else -1
                checkpoint[key] = values[idx]
        
        # Simpan model
        import torch
        torch.save(checkpoint, checkpoint_path)
        
        # Salin juga sebagai best.pt jika diperlukan
        if is_best:
            best_path = checkpoint_dir / 'best.pt'
            import shutil
            shutil.copy2(checkpoint_path, best_path)
        
        return str(checkpoint_path)
    
    def load_model_checkpoint(self, checkpoint_path: Optional[str] = None):
        """
        Muat checkpoint model dari direktori eksperimen.
        
        Args:
            checkpoint_path: Path checkpoint (opsional, default: best.pt)
            
        Returns:
            Checkpoint
        """
        if checkpoint_path is None:
            checkpoint_path = self.experiment_dir / 'checkpoints' / 'best.pt'
        else:
            checkpoint_path = Path(checkpoint_path)
        
        # Muat checkpoint
        import torch
        checkpoint = torch.load(checkpoint_path)
        
        return checkpoint
    
    def generate_report(self, include_plots: bool = True) -> str:
        """
        Generate laporan eksperimen.
        
        Args:
            include_plots: Sertakan plot metrik
            
        Returns:
            Path ke file laporan
        """
        report_path = self.experiment_dir / 'report.md'
        best_metrics = self.get_best_metrics()
        
        # Buat plot jika diminta
        if include_plots:
            self._generate_plots()
        
        # Buat laporan markdown
        with open(report_path, 'w') as f:
            f.write(f"# Laporan Eksperimen: {self.name}\n\n")
            f.write(f"ID: {self.experiment_id}\n\n")
            
            # Metrik terbaik
            f.write("## Hasil Terbaik\n\n")
            if best_metrics:
                f.write(f"- Epoch: {best_metrics.get('epoch', 'N/A')}\n")
                for key, value in best_metrics.items():
                    if key != 'epoch':
                        if isinstance(value, float):
                            f.write(f"- {key}: {value:.4f}\n")
                        else:
                            f.write(f"- {key}: {value}\n")
            else:
                f.write("Tidak ada metrik tersedia\n")
            
            # Parameter eksperimen
            f.write("\n## Parameter Eksperimen\n\n")
            for key, value in self.parameters.items():
                f.write(f"- {key}: {value}\n")
            
            # Gambar plot jika dibuat
            if include_plots:
                f.write("\n## Visualisasi Metrik\n\n")
                f.write("### Learning Curve\n\n")
                f.write("![Learning Curve](plots/learning_curve.png)\n\n")
                
                # Plot tambahan jika metrics mAP atau F1 tersedia
                if 'mAP' in self.metrics and self.metrics['mAP']:
                    f.write("### Metrik Performa\n\n")
                    f.write("![Performance Metrics](plots/performance_metrics.png)\n\n")
        
        return str(report_path)
    
    def _generate_plots(self) -> None:
        """Generate plot metrik untuk laporan."""
        plots_dir = self.experiment_dir / 'plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            
            # Plot learning curve
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['epochs'], self.metrics['train_loss'], 'b-', label='Training Loss')
            
            if 'val_loss' in self.metrics and self.metrics['val_loss']:
                plt.plot(self.metrics['epochs'], self.metrics['val_loss'], 'r-', label='Validation Loss')
            
            plt.title(f'Learning Curve: {self.name}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(plots_dir / 'learning_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot performance metrics jika tersedia
            if 'mAP' in self.metrics and self.metrics['mAP'] or 'f1' in self.metrics and self.metrics['f1']:
                plt.figure(figsize=(10, 6))
                
                if 'mAP' in self.metrics and self.metrics['mAP']:
                    plt.plot(self.metrics['epochs'], self.metrics['mAP'], 'g-', label='mAP')
                
                if 'f1' in self.metrics and self.metrics['f1']:
                    plt.plot(self.metrics['epochs'], self.metrics['f1'], 'm-', label='F1 Score')
                
                plt.title(f'Performance Metrics: {self.name}')
                plt.xlabel('Epoch')
                plt.ylabel('Metric Value')
                plt.legend()
                plt.grid(True)
                plt.savefig(plots_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
                plt.close()
        except ImportError:
            pass
        
    @classmethod
    def compare_experiments(
        cls,
        experiment_dirs: List[str],
        output_dir: Optional[str] = None,
        include_plots: bool = True
    ) -> str:
        """
        Bandingkan beberapa eksperimen dan buat laporan perbandingan.
        
        Args:
            experiment_dirs: List direktori eksperimen
            output_dir: Direktori output untuk laporan (opsional)
            include_plots: Sertakan plot perbandingan
            
        Returns:
            Path ke file laporan perbandingan
        """
        if output_dir is None:
            base_dir = Path(experiment_dirs[0]).parent
            output_dir = base_dir / 'comparison'
        else:
            output_dir = Path(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Dapatkan metrik dari setiap eksperimen
        experiments = []
        for exp_dir in experiment_dirs:
            try:
                # Muat konfigurasi
                config_path = Path(exp_dir) / 'config.yaml'
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Muat metrik
                metrics_path = Path(exp_dir) / 'metrics.json'
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                # Tambahkan ke daftar eksperimen
                experiments.append({
                    'dir': exp_dir,
                    'name': config['experiment']['name'],
                    'metrics': metrics,
                    'config': config
                })
            except Exception as e:
                print(f"⚠️ Gagal memuat eksperimen {exp_dir}: {str(e)}")
        
        # Generate plot perbandingan
        if include_plots:
            cls._generate_comparison_plots(experiments, output_dir)
        
        # Tulis laporan perbandingan
        report_path = output_dir / 'comparison_report.md'
        with open(report_path, 'w') as f:
            f.write("# Perbandingan Eksperimen\n\n")
            
            # Tabel ringkasan
            f.write("## Ringkasan\n\n")
            f.write("| Eksperimen | Best Epoch | Best Val Loss | mAP | F1 |\n")
            f.write("|------------|-----------|--------------|-----|----|\n")
            
            for exp in experiments:
                name = exp['name']
                
                # Hitung metrik terbaik
                best_idx = None
                if 'val_loss' in exp['metrics'] and exp['metrics']['val_loss']:
                    best_idx = exp['metrics']['val_loss'].index(min(exp['metrics']['val_loss']))
                    best_val_loss = exp['metrics']['val_loss'][best_idx]
                elif 'train_loss' in exp['metrics'] and exp['metrics']['train_loss']:
                    best_idx = exp['metrics']['train_loss'].index(min(exp['metrics']['train_loss']))
                    best_val_loss = 'N/A'
                else:
                    best_val_loss = 'N/A'
                
                # Epoch terbaik
                best_epoch = exp['metrics']['epochs'][best_idx] if best_idx is not None else 'N/A'
                
                # Metrik tambahan
                best_map = (exp['metrics']['mAP'][best_idx] if 'mAP' in exp['metrics'] and 
                           best_idx is not None and best_idx < len(exp['metrics']['mAP']) else 'N/A')
                
                best_f1 = (exp['metrics']['f1'][best_idx] if 'f1' in exp['metrics'] and 
                          best_idx is not None and best_idx < len(exp['metrics']['f1']) else 'N/A')
                
                # Format nilai untuk tabel
                if isinstance(best_val_loss, float): best_val_loss = f"{best_val_loss:.4f}"
                if isinstance(best_map, float): best_map = f"{best_map:.4f}"
                if isinstance(best_f1, float): best_f1 = f"{best_f1:.4f}"
                
                f.write(f"| {name} | {best_epoch} | {best_val_loss} | {best_map} | {best_f1} |\n")
            
            # Tampilkan plot perbandingan jika dibuat
            if include_plots:
                f.write("\n## Learning Curves\n\n")
                f.write("![Learning Curves](learning_curves_comparison.png)\n\n")
                
                f.write("\n## Performance Metrics\n\n")
                f.write("![Performance Metrics](performance_metrics_comparison.png)\n\n")
        
        return str(report_path)
    
    @staticmethod
    def _generate_comparison_plots(experiments: List[Dict], output_dir: Path) -> None:
        """Generate plot perbandingan antar eksperimen."""
        try:
            import matplotlib.pyplot as plt
            
            # Plot learning curves
            plt.figure(figsize=(12, 8))
            
            for exp in experiments:
                name = exp['name']
                if 'val_loss' in exp['metrics'] and exp['metrics']['val_loss']:
                    plt.plot(exp['metrics']['epochs'], exp['metrics']['val_loss'], marker='o', markersize=3, label=f"{name}")
            
            plt.title('Perbandingan Learning Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(output_dir / 'learning_curves_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot performance metrics jika tersedia
            metrics_available = False
            for exp in experiments:
                if ('mAP' in exp['metrics'] and exp['metrics']['mAP']) or \
                   ('f1' in exp['metrics'] and exp['metrics']['f1']):
                    metrics_available = True
                    break
            
            if metrics_available:
                plt.figure(figsize=(12, 8))
                for exp in experiments:
                    name = exp['name']
                    
                    if 'mAP' in exp['metrics'] and exp['metrics']['mAP']:
                        plt.plot(exp['metrics']['epochs'], exp['metrics']['mAP'], marker='o', markersize=3, linestyle='-', label=f"{name} (mAP)")
                
                plt.title('Perbandingan mAP')
                plt.xlabel('Epoch')
                plt.ylabel('Metric Value')
                plt.legend()
                plt.grid(True)
                plt.savefig(output_dir / 'performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
        except ImportError:
            pass