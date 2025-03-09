# File: smartcash/handlers/evaluation/integration/visualization_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Adapter untuk integrasi visualisasi hasil evaluasi

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.utils.visualization.evaluation_visualizer import EvaluationVisualizer

class VisualizationAdapter:
    """
    Adapter untuk integrasi visualisasi evaluasi.
    Menghubungkan pipeline evaluasi dengan komponen visualisasi.
    """
    
    def __init__(
        self,
        config: Dict,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi adapter visualisasi.
        
        Args:
            config: Konfigurasi untuk evaluasi
            output_dir: Direktori output untuk visualisasi (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.logger = logger or get_logger("visualization_adapter")
        
        # Output directory
        self.output_dir = output_dir or Path(self.config.get('output_dir', 'results/evaluation/plots'))
        
        # Inisialisasi visualizer
        self.visualizer = EvaluationVisualizer(
            output_dir=self.output_dir,
            logger=self.logger
        )
        
        self.logger.debug(f"🔧 VisualizationAdapter diinisialisasi (output_dir={self.output_dir})")
    
    def generate_batch_plots(
        self,
        batch_results: Dict[str, Any],
        prefix: str = "batch",
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate visualisasi untuk evaluasi batch.
        
        Args:
            batch_results: Hasil evaluasi batch
            prefix: Awalan nama file (opsional)
            **kwargs: Parameter tambahan untuk plotting
            
        Returns:
            Dictionary berisi paths ke plots yang dibuat
        """
        try:
            # Extract metrics table dari summary
            if 'summary' not in batch_results or 'metrics_table' not in batch_results['summary']:
                self.logger.warning("⚠️ Tidak ada metrics_table dalam hasil batch")
                return {}
            
            metrics_df = batch_results['summary']['metrics_table']
            
            # Generate plots
            plots = self.visualizer.create_all_plots(
                metrics_data=metrics_df,
                prefix=prefix,
                **kwargs
            )
            
            self.logger.info(f"📊 Berhasil membuat {len(plots)} visualisasi batch")
            return plots
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat visualisasi batch: {str(e)}")
            return {}
    
    def generate_research_plots(
        self,
        research_results: Dict[str, Any],
        prefix: str = "research",
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate visualisasi untuk evaluasi skenario penelitian.
        
        Args:
            research_results: Hasil evaluasi skenario penelitian
            prefix: Awalan nama file (opsional)
            **kwargs: Parameter tambahan untuk plotting
            
        Returns:
            Dictionary berisi paths ke plots yang dibuat
        """
        try:
            # Extract metrics table dari summary
            if 'summary' not in research_results or 'metrics_table' not in research_results['summary']:
                self.logger.warning("⚠️ Tidak ada metrics_table dalam hasil penelitian")
                return {}
            
            metrics_df = research_results['summary']['metrics_table']
            
            # Generate plots
            plots = self.visualizer.create_all_plots(
                metrics_data=metrics_df,
                prefix=prefix,
                **kwargs
            )
            
            self.logger.info(f"📊 Berhasil membuat {len(plots)} visualisasi penelitian")
            return plots
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat visualisasi penelitian: {str(e)}")
            return {}
    
    def generate_custom_plots(
        self,
        data: Union[pd.DataFrame, List[Dict], Dict[str, Dict]],
        prefix: str = "custom",
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate visualisasi dari data kustom.
        
        Args:
            data: DataFrame, List dict, atau Dict hasil
            prefix: Awalan nama file (opsional)
            **kwargs: Parameter tambahan untuk plotting
            
        Returns:
            Dictionary berisi paths ke plots yang dibuat
        """
        try:
            # Process input data
            if isinstance(data, pd.DataFrame):
                metrics_df = data
            elif isinstance(data, list):
                metrics_df = pd.DataFrame(data)
            elif isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
                # Convert dict of dicts to list of dicts
                metrics_list = []
                for key, value in data.items():
                    metrics_dict = value.copy()
                    metrics_dict['name'] = key
                    metrics_list.append(metrics_dict)
                metrics_df = pd.DataFrame(metrics_list)
            else:
                self.logger.warning(f"⚠️ Format data tidak didukung: {type(data)}")
                return {}
            
            # Generate plots
            plots = self.visualizer.create_all_plots(
                metrics_data=metrics_df,
                prefix=prefix,
                **kwargs
            )
            
            self.logger.info(f"📊 Berhasil membuat {len(plots)} visualisasi kustom")
            return plots
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat visualisasi kustom: {str(e)}")
            return {}