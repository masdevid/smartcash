"""
File: smartcash/utils/visualization/metrics.py
Author: Alfrida Sabar
Deskripsi: Visualisasi metrik evaluasi model dengan berbagai jenis plot dan grafik
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from smartcash.utils.visualization.base import VisualizationHelper
from smartcash.utils.logger import get_logger

class MetricsVisualizer:
    """
    Visualisasi metrik evaluasi model dengan berbagai tipe plot.
    """
    
    def __init__(
        self, 
        output_dir: str = "results/metrics",
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi metrics visualizer.
        
        Args:
            output_dir: Direktori untuk menyimpan hasil
            logger: Logger untuk logging
        """
        self.output_dir = VisualizationHelper.create_output_directory(output_dir)
        self.logger = logger or get_logger("metrics_visualizer")
        
        # Setup plot style
        VisualizationHelper.set_plot_style()
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        filename: Optional[str] = None,
        normalize: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "Blues"
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: List nama kelas
            title: Judul plot
            filename: Nama file untuk menyimpan plot
            normalize: Normalisasi nilai dalam matriks
            figsize: Ukuran figure
            cmap: Colormap
            
        Returns:
            Figure matplotlib
        """
        # Normalisasi matrix jika diminta
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)  # Handle division by zero
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Buat figure
        plt.figure(figsize=figsize)
        
        # Plot heatmap
        ax = sns.heatmap(
            cm, 
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            cbar_kws={"shrink": .8}
        )
        
        plt.title(title, size=16)
        plt.ylabel('True Label', size=14)
        plt.xlabel('Predicted Label', size=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Simpan jika diminta
        if filename:
            output_path = self.output_dir / filename
            VisualizationHelper.save_figure(plt.gcf(), output_path, logger=self.logger)
        
        fig = plt.gcf()
        plt.close()
        
        return fig
    
    def plot_training_metrics(
        self,
        metrics: Dict[str, List],
        title: str = "Training Metrics",
        filename: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        include_lr: bool = True
    ) -> plt.Figure:
        """
        Plot metrik training.
        
        Args:
            metrics: Dictionary dengan metrik training
            title: Judul plot
            filename: Nama file untuk menyimpan plot
            figsize: Ukuran figure
            include_lr: Tampilkan learning rate jika tersedia
            
        Returns:
            Figure matplotlib
        """
        # Validasi metrics
        if not metrics or 'train_loss' not in metrics or 'val_loss' not in metrics:
            self.logger.warning("⚠️ Metrics tidak valid untuk plotting")
            fig = plt.figure(figsize=figsize)
            plt.title("Tidak ada data metrik yang valid")
            return fig
        
        # Setup figure and axis
        if include_lr and 'learning_rates' in metrics and metrics['learning_rates']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            has_lr = True
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
            has_lr = False
        
        # Plot training dan validation loss
        epochs = metrics.get('epochs', list(range(1, len(metrics['train_loss']) + 1)))
        
        ax1.plot(epochs, metrics['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
        
        # Highlight best epoch
        best_epoch_idx = np.argmin(metrics['val_loss'])
        best_epoch = epochs[best_epoch_idx]
        best_val_loss = metrics['val_loss'][best_epoch_idx]
        
        ax1.scatter([best_epoch], [best_val_loss], c='gold', s=100, zorder=5, edgecolor='k')
        ax1.annotate(
            f'Best: {best_val_loss:.4f}',
            (best_epoch, best_val_loss),
            xytext=(10, -20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->", color='black')
        )
        
        ax1.set_title(f"{title} - Loss", size=16)
        ax1.set_xlabel('Epoch', size=14)
        ax1.set_ylabel('Loss', size=14)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Plot learning rate jika tersedia
        if has_lr:
            ax2.plot(epochs, metrics['learning_rates'], 'g-', label='Learning Rate')
            ax2.set_title('Learning Rate Schedule', size=16)
            ax2.set_xlabel('Epoch', size=14)
            ax2.set_ylabel('Learning Rate', size=14)
            ax2.set_yscale('log')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()
        
        plt.tight_layout()
        
        # Simpan jika diminta
        if filename:
            output_path = self.output_dir / filename
            VisualizationHelper.save_figure(fig, output_path, logger=self.logger)
        
        return fig
    
    def plot_accuracy_metrics(
        self,
        metrics: Dict[str, List],
        title: str = "Accuracy Metrics",
        filename: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot metrik akurasi (precision, recall, f1, accuracy).
        
        Args:
            metrics: Dictionary dengan metrik
            title: Judul plot
            filename: Nama file untuk menyimpan plot
            figsize: Ukuran figure
            
        Returns:
            Figure matplotlib
        """
        # Identifikasi metrik akurasi yang tersedia
        accuracy_metrics = {}
        for metric in ['precision', 'recall', 'f1', 'accuracy', 'mAP']:
            if metric in metrics and len(metrics[metric]) > 0:
                accuracy_metrics[metric] = metrics[metric]
        
        if not accuracy_metrics:
            self.logger.warning("⚠️ Tidak ada metrik akurasi yang tersedia")
            fig = plt.figure(figsize=figsize)
            plt.title("Tidak ada metrik akurasi tersedia")
            return fig
        
        # Buat figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot setiap metrik
        epochs = metrics.get('epochs', list(range(1, len(next(iter(accuracy_metrics.values()))) + 1)))
        
        for metric_name, values in accuracy_metrics.items():
            ax.plot(epochs, values, 'o-', label=metric_name.capitalize())
        
        ax.set_title(title, size=16)
        ax.set_xlabel('Epoch', size=14)
        ax.set_ylabel('Value', size=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        
        # Simpan jika diminta
        if filename:
            output_path = self.output_dir / filename
            VisualizationHelper.save_figure(fig, output_path, logger=self.logger)
        
        return fig
    
    def plot_model_comparison(
        self,
        comparison_data: pd.DataFrame,
        metric_cols: List[str] = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'mAP'],
        title: str = "Model Comparison",
        filename: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Plot perbandingan metrik berbagai model.
        
        Args:
            comparison_data: DataFrame dengan data perbandingan
            metric_cols: Kolom metrik untuk plot
            title: Judul plot
            filename: Nama file untuk menyimpan plot
            figsize: Ukuran figure
            
        Returns:
            Figure matplotlib
        """
        if comparison_data.empty:
            self.logger.warning("⚠️ DataFrame kosong untuk perbandingan model")
            fig = plt.figure(figsize=figsize)
            plt.title("Tidak ada data perbandingan")
            return fig
        
        # Validasi kolom metrik
        valid_metrics = [col for col in metric_cols if col in comparison_data.columns]
        if not valid_metrics:
            self.logger.warning(f"⚠️ Tidak ada metrik valid dalam {metric_cols}")
            fig = plt.figure(figsize=figsize)
            plt.title("Tidak ada metrik yang valid untuk perbandingan")
            return fig
        
        # Setup figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Reshape data for plotting
        plot_data = pd.melt(
            comparison_data,
            id_vars=['Model'] if 'Model' in comparison_data.columns else comparison_data.index.name,
            value_vars=valid_metrics,
            var_name='Metric',
            value_name='Value'
        )
        
        # Plot perbandingan
        sns.barplot(
            data=plot_data,
            x='Metric',
            y='Value',
            hue='Model' if 'Model' in comparison_data.columns else plot_data.index.name,
            ax=ax
        )
        
        ax.set_title(title, size=16)
        ax.set_xlabel('Metric', size=14)
        ax.set_ylabel('Value', size=14)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add legend with better positioning
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Simpan jika diminta
        if filename:
            output_path = self.output_dir / filename
            VisualizationHelper.save_figure(fig, output_path, logger=self.logger)
        
        return fig
    
    def plot_research_comparison(
        self,
        results_df: pd.DataFrame,
        metric_cols: List[str] = ['Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP'],
        title: str = "Perbandingan Skenario Penelitian",
        filename: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Plot perbandingan hasil skenario penelitian.
        
        Args:
            results_df: DataFrame hasil penelitian
            metric_cols: Kolom metrik untuk plot
            title: Judul plot
            filename: Nama file untuk menyimpan plot
            figsize: Ukuran figure
            
        Returns:
            Figure matplotlib
        """
        if results_df.empty:
            self.logger.warning("⚠️ DataFrame kosong untuk perbandingan skenario")
            fig = plt.figure(figsize=figsize)
            plt.title("Tidak ada data skenario penelitian")
            return fig
        
        # Filter hanya hasil yang sukses jika kolom Status ada
        if 'Status' in results_df.columns:
            success_results = results_df[results_df['Status'] == 'Sukses'].copy()
            if success_results.empty:
                self.logger.warning("⚠️ Tidak ada skenario yang berhasil dievaluasi")
                fig = plt.figure(figsize=figsize)
                plt.title("Tidak ada skenario sukses untuk dibandingkan")
                return fig
        else:
            success_results = results_df.copy()
        
        # Validasi kolom yang diperlukan
        required_cols = ['Skenario']
        if not all(col in success_results.columns for col in required_cols):
            self.logger.warning(f"⚠️ Kolom yang diperlukan tidak lengkap: {required_cols}")
            # Gunakan indeks sebagai skenario jika tidak ada kolom Skenario
            if 'Skenario' not in success_results.columns:
                success_results['Skenario'] = [f"Skenario {i+1}" for i in range(len(success_results))]
        
        # Validasi kolom metrik
        valid_metrics = [col for col in metric_cols if col in success_results.columns]
        if not valid_metrics:
            self.logger.warning(f"⚠️ Tidak ada metrik valid dalam {metric_cols}")
            fig = plt.figure(figsize=figsize)
            plt.title("Tidak ada metrik yang valid untuk perbandingan")
            return fig
        
        # Setup figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Metrik akurasi
        hue_col = 'Backbone' if 'Backbone' in success_results.columns else None
        
        # Reshape data for grouped bar plot
        plot_data = pd.melt(
            success_results,
            id_vars=['Skenario'] + ([hue_col] if hue_col else []),
            value_vars=valid_metrics,
            var_name='Metric',
            value_name='Value'
        )
        
        # Plot grouped bar plot
        sns.barplot(
            data=plot_data,
            x='Skenario',
            y='Value',
            hue=hue_col,
            ax=axes[0]
        )
        
        axes[0].set_title(f"{title} - Metrik Akurasi", size=16)
        axes[0].set_xlabel('Skenario', size=14)
        axes[0].set_ylabel('Value (%)', size=14)
        axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x labels for better readability
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
        
        # Plot 2: Inference time jika tersedia
        time_col = 'Inference Time (ms)' if 'Inference Time (ms)' in success_results.columns else 'Waktu Inferensi (ms)'
        if time_col in success_results.columns:
            # Bar plot untuk waktu inferensi
            sns.barplot(
                data=success_results,
                x='Skenario',
                y=time_col,
                hue=hue_col,
                ax=axes[1]
            )
            
            # Add FPS as text on top of bars
            if not success_results[time_col].isnull().any():
                for i, time_val in enumerate(success_results[time_col]):
                    fps = 1000 / time_val
                    axes[1].text(
                        i,
                        time_val + 0.5,
                        f"{fps:.1f} FPS",
                        ha='center',
                        color='black',
                        fontweight='bold'
                    )
            
            axes[1].set_title(f"Waktu Inferensi per Skenario", size=16)
            axes[1].set_xlabel('Skenario', size=14)
            axes[1].set_ylabel('Waktu (ms)', size=14)
            axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Rotate x labels for better readability
            plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
        else:
            axes[1].set_visible(False)
        
        plt.tight_layout()
        
        # Simpan jika diminta
        if filename:
            output_path = self.output_dir / filename
            VisualizationHelper.save_figure(fig, output_path, logger=self.logger)
        
        return fig


# Fungsi helper untuk plotting cepat tanpa membuat instance
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
    normalize: bool = True
) -> plt.Figure:
    """
    Fungsi helper untuk plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Nama kelas
        output_path: Path untuk menyimpan output
        normalize: Normalisasi nilai dalam matrix
        
    Returns:
        Figure matplotlib
    """
    visualizer = MetricsVisualizer()
    fig = visualizer.plot_confusion_matrix(
        cm, 
        class_names, 
        filename=Path(output_path).name if output_path else None,
        normalize=normalize
    )
    
    return fig