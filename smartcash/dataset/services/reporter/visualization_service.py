"""
File: smartcash/dataset/services/reporter/visualization_service.py
Deskripsi: Layanan untuk membuat visualisasi laporan dataset
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

from smartcash.common.logger import get_logger
from smartcash.dataset.visualization.dashboard.class_visualizer import ClassVisualizer
from smartcash.dataset.visualization.dashboard.layer_visualizer import LayerVisualizer
from smartcash.dataset.visualization.dashboard.bbox_visualizer import BBoxVisualizer
from smartcash.dataset.visualization.dashboard.quality_visualizer import QualityVisualizer
from smartcash.dataset.visualization.dashboard.split_visualizer import SplitVisualizer
from smartcash.dataset.visualization.dashboard.recommendation_visualizer import RecommendationVisualizer
from smartcash.dataset.visualization.dashboard_visualizer import DashboardVisualizer


class VisualizationService:
    """Layanan untuk membuat visualisasi laporan dataset."""
    
    def __init__(self, config: Dict, output_dir: str, logger=None):
        """
        Inisialisasi VisualizationService.
        
        Args:
            config: Konfigurasi aplikasi
            output_dir: Direktori untuk menyimpan visualisasi
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.logger = logger or get_logger("visualization_service")
        
        # Buat direktori output jika belum ada
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi helper visualizer
        self.class_visualizer = ClassVisualizer()
        self.layer_visualizer = LayerVisualizer()
        self.bbox_visualizer = BBoxVisualizer()
        self.quality_visualizer = QualityVisualizer()
        self.split_visualizer = SplitVisualizer()
        self.recommendation_visualizer = RecommendationVisualizer()
        self.dashboard_visualizer = DashboardVisualizer()
        
        self.logger.info(f"📊 VisualizationService diinisialisasi dengan output di: {self.output_dir}")
    
    def _get_save_path(self, save_path: Optional[str], default_name: str) -> str:
        """
        Mendapatkan path untuk menyimpan visualisasi.
        
        Args:
            save_path: Path yang disediakan (opsional)
            default_name: Nama file default
            
        Returns:
            Path lengkap untuk menyimpan visualisasi
        """
        if save_path:
            # Gunakan path yang disediakan
            full_path = Path(save_path)
            # Pastikan direktori ada
            os.makedirs(full_path.parent, exist_ok=True)
        else:
            # Gunakan default path
            full_path = self.output_dir / default_name
        
        return str(full_path)
    
    def create_report_visualizations(
        self, 
        report: Dict[str, Any], 
        output_prefix: str = "report",
        include_dashboard: bool = True
    ) -> Dict[str, str]:
        """
        Buat semua visualisasi yang diperlukan untuk laporan dataset.
        
        Args:
            report: Laporan dataset
            output_prefix: Prefix untuk nama file output
            include_dashboard: Apakah membuat dashboard
            
        Returns:
            Dictionary berisi path visualisasi
        """
        import matplotlib.pyplot as plt
        
        vis_paths = {}
        
        # 1. Visualisasi distribusi kelas
        if 'class_metrics' in report and 'class_percentages' in report['class_metrics']:
            class_data = report['class_metrics']['class_percentages']
            
            fig, ax = plt.subplots(figsize=(12, 8))
            self.class_visualizer.plot_class_distribution(
                ax, class_data, title="Distribusi Kelas"
            )
            
            save_path = self._get_save_path(None, f"{output_prefix}_class_distribution.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            vis_paths['class_distribution'] = save_path
        
        # 2. Visualisasi distribusi layer
        if 'layer_metrics' in report and 'layer_percentages' in report['layer_metrics']:
            layer_data = report['layer_metrics']['layer_percentages']
            
            fig, ax = plt.subplots(figsize=(10, 8))
            self.layer_visualizer.plot_layer_distribution(
                ax, layer_data, title="Distribusi Layer"
            )
            
            save_path = self._get_save_path(None, f"{output_prefix}_layer_distribution.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            vis_paths['layer_distribution'] = save_path
        
        # 3. Visualisasi distribusi bbox
        if 'bbox_metrics' in report and 'size_distribution' in report['bbox_metrics']:
            bbox_data = report['bbox_metrics']['size_distribution']
            
            fig = plt.figure(figsize=(15, 6))
            self.bbox_visualizer.plot_bbox_distribution(
                fig, bbox_data, title="Distribusi Ukuran Bounding Box"
            )
            
            save_path = self._get_save_path(None, f"{output_prefix}_bbox_distribution.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            vis_paths['bbox_distribution'] = save_path
        
        # 4. Visualisasi skor kualitas
        if 'quality_score' in report:
            fig, ax = plt.subplots(figsize=(8, 6))
            self.quality_visualizer.plot_quality_gauge(
                ax, report['quality_score']
            )
            
            save_path = self._get_save_path(None, f"{output_prefix}_quality_score.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            vis_paths['quality_score'] = save_path
            
        # 5. Visualisasi distribusi split
        if 'split_stats' in report:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.split_visualizer.plot_split_distribution(
                ax, report['split_stats'], title="Distribusi Split"
            )
            
            save_path = self._get_save_path(None, f"{output_prefix}_split_distribution.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            vis_paths['split_distribution'] = save_path
        
        # 6. Dashboard
        if include_dashboard:
            try:
                fig = self.dashboard_visualizer.create_dashboard(report)
                
                save_path = self._get_save_path(None, f"{output_prefix}_dashboard.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                vis_paths['dashboard'] = save_path
                
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal membuat dashboard: {str(e)}")
        
        return vis_paths
    
    def create_comparison_visualization(
        self,
        comparison: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Buat visualisasi perbandingan antar dataset.
        
        Args:
            comparison: Data perbandingan
            save_path: Path untuk menyimpan visualisasi (opsional)
            
        Returns:
            Path ke file visualisasi
        """
        import matplotlib.pyplot as plt
        
        # Extract datasets
        datasets = comparison.get('datasets', [])
        
        if not datasets:
            self.logger.warning("⚠️ Tidak ada data perbandingan")
            return ""
        
        # Setup figure
        fig = plt.figure(figsize=(15, 10))
        
        # Bangun comparison dashboard
        try:
            labels = [dataset.get('name', f"Dataset {i+1}") for i, dataset in enumerate(datasets)]
            reports = [dataset.get('report', {}) for dataset in datasets]
            
            # Gunakan dashboard visualizer untuk membuat comparison dashboard
            fig = self.dashboard_visualizer.create_comparison_dashboard(
                reports, labels, title="Perbandingan Dataset"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat perbandingan dataset: {str(e)}")
            return ""
        
        # Simpan visualisasi
        if not save_path:
            save_path = "dataset_comparison.png"
            
        output_path = self.output_dir / save_path
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"📊 Visualisasi perbandingan disimpan ke: {output_path}")
        return str(output_path)
    
    def create_comparison_metrics(
        self,
        reports: List[Dict[str, Any]],
        labels: List[str],
        save_path: Optional[str] = None
    ) -> str:
        """
        Buat visualisasi perbandingan metrik utama dari beberapa laporan dataset.
        
        Args:
            reports: List laporan dataset
            labels: Label untuk setiap laporan
            save_path: Path untuk menyimpan visualisasi (opsional)
            
        Returns:
            Path ke file visualisasi
        """
        import matplotlib.pyplot as plt
        
        # Validasi input
        if len(reports) != len(labels) or not reports:
            self.logger.warning("⚠️ Jumlah laporan dan label tidak sesuai")
            return ""
            
        # Setup figure
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Perbandingan skor kualitas
        self.quality_visualizer.plot_quality_scores_comparison(axs[0, 0], reports, labels)
        
        # 2. Perbandingan distribusi kelas
        self.class_visualizer.plot_class_distribution_comparison(axs[0, 1], reports, labels)
        
        # 3. Perbandingan distribusi layer
        self.layer_visualizer.plot_layer_distribution_comparison(axs[1, 0], reports, labels)
        
        # 4. Perbandingan imbalance scores
        self.quality_visualizer.plot_imbalance_comparison(axs[1, 1], reports, labels)
        
        # Title
        fig.suptitle("Perbandingan Metrik Dataset", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Simpan visualisasi
        if not save_path:
            save_path = "metrics_comparison.png"
            
        output_path = self.output_dir / save_path
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"📊 Perbandingan metrik dataset disimpan ke: {output_path}")
        return str(output_path)