"""
File: smartcash/model/visualization/research/experiment_visualizer.py
Deskripsi: Komponen untuk visualisasi hasil eksperimen model
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

from smartcash.model.visualization.research.base_research_visualizer import BaseResearchVisualizer

class ExperimentVisualizer(BaseResearchVisualizer):
    """
    Komponen untuk visualisasi hasil eksperimen model.
    Digunakan untuk membuat perbandingan visual antara backbone, hyperparameter, dll.
    """
    
    def __init__(
        self,
        output_dir: str = "runs/experiments/visualizations",
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi experiment visualizer.
        
        Args:
            output_dir: Direktori output untuk visualisasi
            logger: Logger untuk logging
        """
        super().__init__(output_dir, logger)
        
    def visualize_experiment_comparison(
        self,
        results_df: pd.DataFrame,
        title: str = "Perbandingan Eksperimen Model",
        filename: Optional[str] = None,
        highlight_best: bool = True,
        figsize: Tuple[int, int] = (14, 10)
    ) -> Dict[str, Any]:
        """
        Visualisasikan perbandingan hasil berbagai eksperimen.
        
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
            
        # Setup figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Ekstrak kolom metrik dan waktu
        metric_cols = [col for col in results_df.columns if col in 
                     ['Akurasi', 'Precision', 'Recall', 'F1-Score', 'mAP', 'Accuracy']]
        time_col = next((col for col in results_df.columns if 'Time' in col or 'Waktu' in col), None)
        
        # Seleksi kolom untuk plot
        if not metric_cols:
            self.logger.warning("⚠️ Tidak ada kolom metrik yang valid")
            return {'status': 'error', 'message': 'Tidak ada metrik valid'}
            
        # Melakukan plotting berdasarkan backbone jika ada
        if 'Backbone' in results_df.columns:
            self._create_backbone_based_plots(axes, results_df, metric_cols, time_col)
        else:
            self._create_general_plots(axes, results_df, metric_cols, time_col)
            
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Simpan jika diminta
        if filename:
            self.save_visualization(fig, filename)
            
        # Buat styled dataframe jika highlight_best
        styled_df = self._create_styled_dataframe(results_df) if highlight_best else results_df
            
        return {
            'status': 'success',
            'figure': fig, 
            'styled_df': styled_df
        }