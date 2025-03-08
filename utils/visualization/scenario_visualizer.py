"""
File: smartcash/utils/visualization/scenario_visualizer.py
Author: Alfrida Sabar
Deskripsi: Visualisasi dan analisis hasil skenario penelitian dengan berbagai jenis grafik
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any

from smartcash.utils.visualization.research_base import BaseResearchVisualizer
from smartcash.utils.visualization.analysis import ScenarioAnalyzer

class ScenarioVisualizer(BaseResearchVisualizer):
    """Visualisasi dan analisis hasil skenario penelitian."""
    
    def __init__(
        self, 
        output_dir: str = "results/research/scenarios",
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi scenario visualizer.
        
        Args:
            output_dir: Direktori untuk menyimpan hasil
            logger: Logger untuk logging
        """
        super().__init__(output_dir, logger)
        self.analyzer = ScenarioAnalyzer()
    
    def visualize_scenario_comparison(
        self,
        results_df: pd.DataFrame,
        title: str = "Perbandingan Skenario Penelitian",
        filename: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 12)
    ) -> Dict[str, Any]:
        """
        Visualisasikan perbandingan berbagai skenario penelitian.
        
        Args:
            results_df: DataFrame hasil skenario
            title: Judul visualisasi
            filename: Nama file untuk menyimpan hasil
            figsize: Ukuran figure
            
        Returns:
            Dict berisi figure dan hasil analisis
        """
        if results_df.empty:
            self.logger.warning("⚠️ DataFrame kosong untuk perbandingan skenario")
            return {'status': 'error', 'message': 'Data kosong'}
        
        # Filter hanya hasil yang sukses jika kolom Status ada
        results_df = self._filter_successful_scenarios(results_df)
        if results_df.empty:
            return {'status': 'error', 'message': 'Tidak ada skenario sukses'}
            
        # Pastikan kolom Skenario ada
        if 'Skenario' not in results_df.columns:
            results_df = self._add_scenario_column(results_df)
        
        # Setup figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Identifikasi kolom yang dibutuhkan
        metric_cols, time_col, condition_col, backbone_col = self._identify_columns(results_df)
        
        # Buat empat jenis plot
        self._create_accuracy_plot(axes[0, 0], results_df, metric_cols, backbone_col)
        self._create_inference_time_plot(axes[0, 1], results_df, time_col, backbone_col)
        self._create_backbone_comparison_plot(axes[1, 0], results_df, metric_cols, backbone_col)
        self._create_condition_comparison_plot(axes[1, 1], results_df, metric_cols, condition_col)
        
        plt.suptitle(title, fontsize=18, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Berikan ruang untuk suptitle
        
        # Simpan jika diminta
        if filename:
            self.save_visualization(fig, filename)
        
        # Lakukan analisis
        analysis = self.analyzer.analyze_scenario_results(results_df, backbone_col, condition_col)
        
        # Return hasil
        return {
            'status': 'success',
            'figure': fig,
            'styled_df': self._create_styled_dataframe(results_df),
            'analysis': analysis
        }
    
    def _filter_successful_scenarios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter hanya skenario yang sukses.
        
        Args:
            df: DataFrame hasil skenario
            
        Returns:
            DataFrame yang sudah difilter
        """
        if 'Status' in df.columns:
            filtered_df = df[df['Status'] == 'Sukses'].copy()
            if filtered_df.empty:
                self.logger.warning("⚠️ Tidak ada skenario yang berhasil dievaluasi")
            return filtered_df
        return df.copy()
    
    def _add_scenario_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tambahkan kolom Skenario jika tidak ada.
        
        Args:
            df: DataFrame hasil skenario
            
        Returns:
            DataFrame dengan kolom Skenario
        """
        self.logger.warning("⚠️ Kolom Skenario tidak ditemukan, menggunakan penomoran otomatis")
        df_copy = df.copy()
        df_copy['Skenario'] = [f"Skenario {i+1}" for i in range(len(df_copy))]
        return df_copy
    
    def _identify_columns(self, df: pd.DataFrame) -> Tuple[List[str], Optional[str], Optional[str], Optional[str]]:
        """
        Identifikasi kolom-kolom penting dalam DataFrame.
        
        Args:
            df: DataFrame hasil skenario
            
        Returns:
            Tuple berisi (metric_cols, time_col, condition_col, backbone_col)
        """
        # Identifikasi kolom metrik
        metric_cols = [col for col in df.columns if col in 
                     ['Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP', 'Accuracy']]
        
        # Identifikasi kolom waktu
        time_col = next((col for col in df.columns if 'Time' in col or 'Waktu' in col), None)
        
        # Identifikasi kolom kondisi dan backbone
        condition_col = 'Kondisi' if 'Kondisi' in df.columns else None
        backbone_col = 'Backbone' if 'Backbone' in df.columns else None
        
        return metric_cols, time_col, condition_col, backbone_col
    
    def _create_accuracy_plot(
        self, 
        ax: plt.Axes, 
        df: pd.DataFrame, 
        metric_cols: List[str],
        backbone_col: Optional[str]
    ) -> None:
        """
        Buat plot akurasi per skenario.
        
        Args:
            ax: Axes untuk plot
            df: DataFrame data
            metric_cols: Kolom metrik
            backbone_col: Kolom backbone
        """
        if 'Skenario' in df.columns and metric_cols:
            # Pilih metrik representatif
            repr_metric = 'mAP' if 'mAP' in metric_cols else 'F1-Score' if 'F1-Score' in metric_cols else metric_cols[0]
            
            # Bar plot untuk metrik representatif
            sns.barplot(
                data=df,
                x='Skenario',
                y=repr_metric,
                hue=backbone_col,
                ax=ax
            )
            
            ax.set_title(f"Perbandingan {repr_metric} per Skenario", size=14)
            ax.set_xlabel('Skenario', size=12)
            ax.set_ylabel(f"{repr_metric} (%)", size=12)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Rotate x labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, "Data tidak cukup untuk perbandingan skenario", 
                  ha='center', va='center', fontsize=14)
            ax.set_title("Perbandingan Akurasi", size=14)
    
    def _create_inference_time_plot(
        self, 
        ax: plt.Axes, 
        df: pd.DataFrame, 
        time_col: Optional[str],
        backbone_col: Optional[str]
    ) -> None:
        """
        Buat plot waktu inferensi per skenario.
        
        Args:
            ax: Axes untuk plot
            df: DataFrame data
            time_col: Kolom waktu
            backbone_col: Kolom backbone
        """
        if 'Skenario' in df.columns and time_col:
            # Bar plot untuk waktu inferensi
            sns.barplot(
                data=df,
                x='Skenario',
                y=time_col,
                hue=backbone_col,
                ax=ax
            )
            
            # Tambahkan FPS di atas bar
            for i, time_val in enumerate(df[time_col]):
                fps = 1000 / time_val if time_val > 0 else 0
                ax.text(
                    i, 
                    time_val + 0.5, 
                    f"{fps:.1f} FPS",
                    ha='center',
                    color='black',
                    fontweight='bold'
                )
            
            ax.set_title("Waktu Inferensi per Skenario", size=14)
            ax.set_xlabel('Skenario', size=12)
            ax.set_ylabel('Waktu (ms)', size=12)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Rotate x labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, "Data waktu inferensi tidak tersedia", 
                  ha='center', va='center', fontsize=14)
            ax.set_title("Waktu Inferensi", size=14)
    
    def _create_backbone_comparison_plot(
        self, 
        ax: plt.Axes, 
        df: pd.DataFrame, 
        metric_cols: List[str],
        backbone_col: Optional[str]
    ) -> None:
        """
        Buat plot perbandingan backbone.
        
        Args:
            ax: Axes untuk plot
            df: DataFrame data
            metric_cols: Kolom metrik
            backbone_col: Kolom backbone
        """
        if backbone_col and metric_cols:
            # Agregasi per backbone
            backbone_metrics = df.groupby(backbone_col)[metric_cols].mean().reset_index()
            
            # Reshape data untuk plotting
            plot_data = pd.melt(
                backbone_metrics,
                id_vars=[backbone_col],
                value_vars=metric_cols,
                var_name='Metric',
                value_name='Value'
            )
            
            # Plot perbandingan
            sns.barplot(
                data=plot_data,
                x='Metric',
                y='Value',
                hue=backbone_col,
                ax=ax
            )
            
            ax.set_title("Perbandingan Backbone", size=14)
            ax.set_xlabel('Metric', size=12)
            ax.set_ylabel('Value (%)', size=12)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Rotate x labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, "Data backbone tidak tersedia", 
                  ha='center', va='center', fontsize=14)
            ax.set_title("Perbandingan Backbone", size=14)
    
    def _create_condition_comparison_plot(
        self, 
        ax: plt.Axes, 
        df: pd.DataFrame, 
        metric_cols: List[str],
        condition_col: Optional[str]
    ) -> None:
        """
        Buat plot perbandingan kondisi.
        
        Args:
            ax: Axes untuk plot
            df: DataFrame data
            metric_cols: Kolom metrik
            condition_col: Kolom kondisi
        """
        if condition_col and metric_cols:
            # Agregasi per kondisi
            condition_metrics = df.groupby(condition_col)[metric_cols].mean().reset_index()
            
            # Reshape data untuk plotting
            plot_data = pd.melt(
                condition_metrics,
                id_vars=[condition_col],
                value_vars=metric_cols,
                var_name='Metric',
                value_name='Value'
            )
            
            # Plot perbandingan
            sns.barplot(
                data=plot_data,
                x='Metric',
                y='Value',
                hue=condition_col,
                ax=ax
            )
            
            ax.set_title("Perbandingan Kondisi", size=14)
            ax.set_xlabel('Metric', size=12)
            ax.set_ylabel('Value (%)', size=12)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Rotate x labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, "Data kondisi tidak tersedia", 
                  ha='center', va='center', fontsize=14)
            ax.set_title("Perbandingan Kondisi", size=14)