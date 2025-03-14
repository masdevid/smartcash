"""
File: smartcash/model/visualization/experiment_visualizer.py
Deskripsi: Komponen untuk visualisasi hasil eksperimen model
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd

class ExperimentVisualizer:
    """
    Komponen untuk visualisasi hasil eksperimen model.
    Digunakan untuk membuat perbandingan visual antara backbone, hyperparameter, dll.
    """
    
    def __init__(
        self,
        output_dir: str = "runs/experiments/visualizations",
        figsize: tuple = (12, 8),
        dpi: int = 150,
        theme: str = 'default'
    ):
        """
        Inisialisasi experiment visualizer.
        
        Args:
            output_dir: Direktori output untuk visualisasi
            figsize: Ukuran default untuk plot
            dpi: DPI untuk output gambar
            theme: Tema visualisasi ('default', 'dark', 'light')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        
        # Setup theme
        self._setup_theme(theme)
    
    def _setup_theme(self, theme: str) -> None:
        """Setup tema visualisasi."""
        if theme == 'dark':
            plt.style.use('dark_background')
            self.colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#FF6D01', '#46BDC6', '#F94892']
        elif theme == 'light':
            plt.style.use('seaborn-v0_8-whitegrid')
            self.colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#FF6D01', '#46BDC6', '#F94892']
        else:
            # Default theme
            plt.style.use('default')
            self.colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#FF6D01', '#46BDC6', '#F94892']
    
    def visualize_backbone_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        metrics: List[str] = None,
        title: str = "Perbandingan Backbone",
        output_filename: str = "backbone_comparison",
        show_values: bool = True
    ) -> Dict[str, str]:
        """
        Visualisasikan perbandingan antar backbone.
        
        Args:
            results: Dictionary hasil eksperimen per backbone
            metrics: List metrik yang akan divisualisasikan
            title: Judul visualisasi
            output_filename: Nama file output (tanpa ekstensi)
            show_values: Tampilkan nilai pada bar chart
            
        Returns:
            Dictionary path visualisasi yang dibuat
        """
        if metrics is None:
            metrics = ['mAP', 'precision', 'recall', 'f1', 'inference_time']
            
        # Filter metrik yang tersedia
        available_metrics = {}
        
        # Proses data
        for backbone, result in results.items():
            if 'error' in result:
                continue
                
            if 'evaluation' in result:
                eval_metrics = result['evaluation']
                for metric in metrics:
                    if metric in eval_metrics:
                        if metric not in available_metrics:
                            available_metrics[metric] = {}
                        available_metrics[metric][backbone] = eval_metrics[metric]
        
        visualization_paths = {}
        
        # Buat visualisasi untuk setiap metrik
        for metric, values in available_metrics.items():
            plt.figure(figsize=self.figsize)
            
            # Siapkan data
            backbones = list(values.keys())
            metric_values = list(values.values())
            
            # Inversi untuk inference_time (nilai lebih kecil = lebih baik)
            if metric == 'inference_time':
                title_suffix = ' (nilai lebih rendah = lebih baik)'
            else:
                title_suffix = ' (nilai lebih tinggi = lebih baik)'
            
            # Buat bar chart
            bars = plt.bar(backbones, metric_values, color=self.colors[:len(backbones)])
            
            # Tambahkan nilai di atas bar
            if show_values:
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f'{height:.4f}',
                        ha='center',
                        va='bottom',
                        fontsize=10
                    )
            
            # Label dan judul
            plt.xlabel('Backbone')
            plt.ylabel(metric)
            plt.title(f'{title} - {metric}{title_suffix}')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Simpan visualisasi
            output_path = self.output_dir / f"{output_filename}_{metric}.png"
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
            visualization_paths[metric] = str(output_path)
        
        # Buat radar chart jika minimal 3 metrik tersedia
        radar_metrics = [m for m in ['mAP', 'precision', 'recall', 'f1'] 
                        if m in available_metrics and len(available_metrics[m]) > 1]
        
        if len(radar_metrics) >= 3:
            radar_path = self._create_radar_chart(
                available_metrics, 
                radar_metrics,
                title=f"{title} - Radar Chart",
                output_filename=f"{output_filename}_radar"
            )
            visualization_paths['radar'] = radar_path
        
        return visualization_paths
    
    def visualize_training_curves(
        self,
        metrics_history: Dict[str, List],
        title: str = "Training Progress",
        output_filename: str = "training_curves"
    ) -> Dict[str, str]:
        """
        Visualisasikan kurva training dan validasi.
        
        Args:
            metrics_history: Dictionary berisi metrics history
            title: Judul visualisasi
            output_filename: Nama file output (tanpa ekstensi)
            
        Returns:
            Dictionary path visualisasi yang dibuat
        """
        visualization_paths = {}
        
        # Training and validation loss
        if 'train_loss' in metrics_history and 'val_loss' in metrics_history and 'epochs' in metrics_history:
            plt.figure(figsize=self.figsize)
            
            epochs = metrics_history['epochs']
            train_loss = metrics_history['train_loss']
            val_loss = metrics_history['val_loss']
            
            plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
            plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{title} - Loss Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            output_path = self.output_dir / f"{output_filename}_loss.png"
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
            visualization_paths['loss'] = str(output_path)
        
        # Learning rate curve
        if 'learning_rates' in metrics_history and 'epochs' in metrics_history:
            plt.figure(figsize=self.figsize)
            
            epochs = metrics_history['epochs']
            learning_rates = metrics_history['learning_rates']
            
            plt.plot(epochs, learning_rates, 'g-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title(f'{title} - Learning Rate Schedule')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            output_path = self.output_dir / f"{output_filename}_lr.png"
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
            visualization_paths['learning_rate'] = str(output_path)
        
        # Metrik tambahan dari epoch_metrics
        if 'epoch_metrics' in metrics_history and len(metrics_history['epoch_metrics']) > 0:
            df = pd.DataFrame(metrics_history['epoch_metrics'])
            
            # Plot metrik selain loss dan learning rate
            for col in df.columns:
                if col not in ['epoch', 'train_loss', 'val_loss', 'learning_rate']:
                    if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                        plt.figure(figsize=self.figsize)
                        plt.plot(df['epoch'], df[col], 'o-', linewidth=2, color=self.colors[0])
                        plt.xlabel('Epoch')
                        plt.ylabel(col)
                        plt.title(f'{title} - {col} per Epoch')
                        plt.grid(True, alpha=0.3)
                        
                        output_path = self.output_dir / f"{output_filename}_{col}.png"
                        plt.savefig(output_path, dpi=self.dpi)
                        plt.close()
                        
                        visualization_paths[col] = str(output_path)
        
        return visualization_paths

    def visualize_parameter_comparison(
        self,
        results: Dict[str, Dict[str, Dict[str, Any]]],
        parameter_name: str,
        metrics: List[str] = None,
        title: str = "Perbandingan Parameter",
        output_filename: str = "parameter_comparison"
    ) -> Dict[str, str]:
        """
        Visualisasikan perbandingan hasil dengan parameter berbeda.
        
        Args:
            results: Dictionary hasil eksperimen per backbone dan parameter
            parameter_name: Nama parameter yang dibandingkan
            metrics: List metrik yang akan divisualisasikan
            title: Judul visualisasi
            output_filename: Nama file output (tanpa ekstensi)
            
        Returns:
            Dictionary path visualisasi yang dibuat
        """
        if metrics is None:
            metrics = ['mAP', 'precision', 'recall', 'f1', 'inference_time']
            
        visualization_paths = {}
        
        # Siapkan data
        backbones = list(results.keys())
        parameters = set()
        
        # Dapatkan semua parameter values
        for backbone, params in results.items():
            for param in params.keys():
                if param != 'error':
                    parameters.add(param)
        
        parameters = sorted(list(parameters))
        
        # Buat visualisasi untuk setiap metrik
        for metric in metrics:
            # Buat figure
            plt.figure(figsize=self.figsize)
            
            # Width untuk bar groups
            width = 0.8 / len(backbones)
            
            # Siapkan data untuk plot
            for i, backbone in enumerate(backbones):
                param_values = []
                for param in parameters:
                    if param in results[backbone] and 'error' not in results[backbone][param]:
                        if 'evaluation' in results[backbone][param] and metric in results[backbone][param]['evaluation']:
                            param_values.append(results[backbone][param]['evaluation'][metric])
                        else:
                            param_values.append(0)
                    else:
                        param_values.append(0)
                
                # Plot bar chart
                x = np.arange(len(parameters))
                offset = i * width - 0.4 + width/2
                plt.bar(x + offset, param_values, width, label=backbone, color=self.colors[i % len(self.colors)])
            
            # Atur x-axis
            plt.xlabel(parameter_name)
            plt.ylabel(metric)
            plt.title(f'{title} - {metric} per {parameter_name}')
            plt.xticks(np.arange(len(parameters)), parameters)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
            # Simpan visualisasi
            output_path = self.output_dir / f"{output_filename}_{metric}.png"
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
            visualization_paths[metric] = str(output_path)
        
        return visualization_paths
    
    def _create_radar_chart(
        self,
        metrics_data: Dict[str, Dict[str, float]],
        metrics_to_show: List[str],
        title: str = "Radar Chart Comparison",
        output_filename: str = "radar_chart"
    ) -> str:
        """
        Buat radar chart untuk perbandingan metrik utama.
        
        Args:
            metrics_data: Dictionary metrik dan nilai untuk backbone
            metrics_to_show: List metrik yang akan ditampilkan
            title: Judul chart
            output_filename: Nama file output (tanpa ekstensi)
            
        Returns:
            Path ke file visualisasi
        """
        # Setup radar chart
        fig = plt.figure(figsize=(10, 10))
        
        # Setup angles for radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics_to_show), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax = plt.subplot(111, polar=True)
        
        # Plot untuk setiap backbone
        backbones = set()
        for metric in metrics_to_show:
            if metric in metrics_data:
                for backbone in metrics_data[metric].keys():
                    backbones.add(backbone)
        
        for i, backbone in enumerate(backbones):
            values = []
            for metric in metrics_to_show:
                if metric in metrics_data and backbone in metrics_data[metric]:
                    values.append(metrics_data[metric][backbone])
                else:
                    values.append(0)
            
            # Close the loop
            values += values[:1]
            
            # Plot values
            ax.plot(angles, values, 'o-', linewidth=2, label=backbone, color=self.colors[i % len(self.colors)])
            ax.fill(angles, values, alpha=0.1, color=self.colors[i % len(self.colors)])
        
        # Set labels
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics_to_show)
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right')
        plt.title(title)
        
        # Simpan visualisasi
        output_path = self.output_dir / f"{output_filename}.png"
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
        
        return str(output_path)