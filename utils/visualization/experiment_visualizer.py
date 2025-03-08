"""
File: smartcash/utils/visualization/experiment_visualizer.py
Author: Alfrida Sabar
Deskripsi: Visualisasi dan analisis hasil eksperimen untuk perbandingan model
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any

from smartcash.utils.visualization.research_base import BaseResearchVisualizer
from smartcash.utils.visualization.research_analysis import ExperimentAnalyzer

class ExperimentVisualizer(BaseResearchVisualizer):
    """Visualisasi dan analisis hasil eksperimen untuk perbandingan model."""
    
    def __init__(
        self, 
        output_dir: str = "results/research/experiments",
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi experiment visualizer.
        
        Args:
            output_dir: Direktori untuk menyimpan hasil
            logger: Logger untuk logging
        """
        super().__init__(output_dir, logger)
        self.analyzer = ExperimentAnalyzer()
    
    def visualize_experiment_comparison(
        self,
        results_df: pd.DataFrame,
        title: str = "Perbandingan Eksperimen",
        filename: Optional[str] = None,
        highlight_best: bool = True,
        figsize: Tuple[int, int] = (15, 12)
    ) -> Dict[str, Any]:
        """
        Visualisasikan perbandingan berbagai eksperimen.
        
        Args:
            results_df: DataFrame hasil eksperimen
            title: Judul visualisasi
            filename: Nama file untuk menyimpan hasil
            highlight_best: Highlight nilai terbaik
            figsize: Ukuran figure
            
        Returns:
            Dict berisi figure dan hasil analisis
        """
        if results_df.empty:
            self.logger.warning("⚠️ DataFrame kosong untuk perbandingan eksperimen")
            return {'status': 'error', 'message': 'Data kosong'}
        
        # Tampilkan tabel hasil
        if highlight_best:
            styled_df = self._create_styled_dataframe(results_df)
        else:
            styled_df = results_df.copy()
        
        # Setup figure untuk visualisasi
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Identifikasi kolom metrik dan performansi
        metric_cols = [col for col in results_df.columns if col in 
                     ['Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP', 'Accuracy']]
        time_col = next((col for col in results_df.columns if 'Time' in col or 'Waktu' in col), None)
        
        # Plot 1: Metrik Akurasi
        self._create_metrics_plot(axes[0], results_df, metric_cols, title)
        
        # Plot 2: Trade-off Akurasi vs Kecepatan
        self._create_tradeoff_plot(axes[1], results_df, metric_cols, time_col)
        
        plt.tight_layout()
        
        # Simpan jika diminta
        if filename:
            self.save_visualization(fig, filename)
        
        # Lakukan analisis
        analysis = self.analyzer.analyze_experiment_results(results_df, metric_cols, time_col)
        
        # Return hasil
        return {
            'status': 'success',
            'figure': fig,
            'styled_df': styled_df,
            'analysis': analysis
        }
    
    def _create_metrics_plot(
        self, 
        ax: plt.Axes, 
        df: pd.DataFrame, 
        metric_cols: List[str],
        title: str
    ) -> None:
        """
        Buat plot metrik akurasi.
        
        Args:
            ax: Axes untuk plot
            df: DataFrame data
            metric_cols: Kolom metrik
            title: Judul plot
        """
        if metric_cols:
            model_col = next((col for col in df.columns if col in 
                           ['Model', 'Backbone', 'Arsitektur']), None)
            
            # Reshape data untuk plotting
            plot_data = pd.melt(
                df,
                id_vars=[model_col] if model_col else df.index.name,
                value_vars=metric_cols,
                var_name='Metric',
                value_name='Value'
            )
            
            # Plot perbandingan
            sns.barplot(
                data=plot_data,
                x='Metric',
                y='Value',
                hue=model_col if model_col else None,
                ax=ax
            )
            
            ax.set_title(f"{title} - Metrik Akurasi", size=16)
            ax.set_xlabel('Metric', size=14)
            ax.set_ylabel('Value (%)', size=14)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Rotate x labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, "Tidak ada data metrik akurasi", 
                  ha='center', va='center', fontsize=14)
            ax.set_title("Metrik Akurasi", size=16)
    
    def _create_tradeoff_plot(
        self, 
        ax: plt.Axes, 
        df: pd.DataFrame, 
        metric_cols: List[str],
        time_col: Optional[str]
    ) -> None:
        """
        Buat plot trade-off akurasi vs kecepatan.
        
        Args:
            ax: Axes untuk plot
            df: DataFrame data
            metric_cols: Kolom metrik
            time_col: Kolom waktu
        """
        if metric_cols and time_col:
            # Pilih metrik representatif (F1 atau mAP)
            repr_metric = 'mAP' if 'mAP' in metric_cols else 'F1-Score' if 'F1-Score' in metric_cols else metric_cols[0]
            model_col = next((col for col in df.columns if col in 
                           ['Model', 'Backbone', 'Arsitektur']), None)
            
            # Scatter plot
            ax.scatter(
                df[time_col], 
                df[repr_metric], 
                s=100, 
                alpha=0.7
            )
            
            # Tambahkan label pada setiap titik
            for i, row in df.iterrows():
                model_name = row[model_col] if model_col and model_col in row else f"Model {i}"
                ax.annotate(
                    model_name,
                    (row[time_col], row[repr_metric]),
                    xytext=(5, 5),
                    textcoords="offset points"
                )
            
            ax.set_title("Trade-off Akurasi vs Kecepatan", size=16)
            ax.set_xlabel(f"{time_col} (semakin kecil semakin baik →)", size=14)
            ax.set_ylabel(f"{repr_metric} (%) (semakin besar semakin baik →)", size=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Tambahkan regions untuk interpretasi
            self._add_tradeoff_regions(ax)
        else:
            ax.text(0.5, 0.5, "Data tidak cukup untuk analisis trade-off", 
                  ha='center', va='center', fontsize=14)
            ax.set_title("Trade-off Akurasi vs Kecepatan", size=16)