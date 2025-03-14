"""
File: smartcash/model/visualization/research_visualizer.py
Deskripsi: Komponen untuk visualisasi penelitian model deteksi objek
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from smartcash.model.visualization.base_research_visualizer import BaseResearchVisualizer
from smartcash.model.visualization.experiment_visualizer import ExperimentVisualizer
from smartcash.model.visualization.scenario_visualizer import ScenarioVisualizer
from smartcash.model.utils.research_model_utils import clean_dataframe

class ResearchVisualizer(BaseResearchVisualizer):
    """
    Visualisasi dan analisis hasil penelitian dengan berbagai jenis grafik.
    Mengintegrasikan komponen-komponen visualisasi eksperimen dan skenario.
    """
    
    def __init__(
        self, 
        output_dir: str = "results/research",
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi visualizer penelitian.
        
        Args:
            output_dir: Direktori untuk menyimpan hasil
            logger: Logger untuk logging
        """
        super().__init__(output_dir, logger)
        
        # Inisialisasi sub-visualizer
        self.experiment_visualizer = ExperimentVisualizer(f"{output_dir}/experiments", logger)
        self.scenario_visualizer = ScenarioVisualizer(f"{output_dir}/scenarios", logger)
    
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
        # Bersihkan data
        clean_df = clean_dataframe(results_df)
        
        # Gunakan experiment visualizer
        return self.experiment_visualizer.visualize_experiment_comparison(
            clean_df,
            title=title,
            filename=filename,
            highlight_best=highlight_best,
            figsize=figsize
        )
    
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
        # Bersihkan data
        clean_df = clean_dataframe(results_df)
        
        # Gunakan scenario visualizer
        return self.scenario_visualizer.visualize_scenario_comparison(
            clean_df,
            title=title,
            filename=filename,
            figsize=figsize
        )


# Fungsi helper untuk visualisasi cepat tanpa membuat instance
def visualize_scenario_comparison(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fungsi helper untuk visualisasi perbandingan skenario.
    
    Args:
        results_df: DataFrame hasil skenario
        output_path: Path untuk menyimpan output
        
    Returns:
        Dict berisi hasil visualisasi dan analisis
    """
    visualizer = ResearchVisualizer()
    
    # Bersihkan data
    clean_df = clean_dataframe(results_df)
    
    filename = Path(output_path).name if output_path else None
    return visualizer.visualize_scenario_comparison(clean_df, filename=filename)