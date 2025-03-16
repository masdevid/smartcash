"""
File: smartcash/dataset/services/explorer/explorer_service.py
Deskripsi: Layanan utama untuk eksplorasi dataset dengan menggunakan interface untuk visualisasi
"""

from pathlib import Path
from typing import Dict, Any, Optional

from smartcash.common.logger import get_logger
from smartcash.common.interfaces.layer_config_interface import ILayerConfigManager
from smartcash.dataset.utils.dataset_utils import DatasetUtils


class ExplorerService:
    """Koordinator utama untuk eksplorasi dan analisis dataset."""
    
    def __init__(
        self, 
        config: Dict, 
        data_dir: str, 
        logger=None, 
        layer_config: Optional[ILayerConfigManager] = None,
        num_workers: int = 4
    ):
        """
        Inisialisasi ExplorerService.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori data
            logger: Logger kustom (opsional)
            layer_config: ILayerConfigManager instance (opsional)
            num_workers: Jumlah worker untuk operasi paralel
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logger or get_logger("explorer_service")
        self.num_workers = num_workers
        
        # Setup utils dengan layer_config
        self.layer_config = layer_config
        self.utils = DatasetUtils(config, data_dir, logger, layer_config)
        
        self.logger.info(f"🔍 ExplorerService diinisialisasi dengan {num_workers} workers")
    
    def analyze_class_distribution(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis distribusi kelas dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Hasil analisis distribusi kelas
        """
        from smartcash.dataset.services.explorer.class_explorer import ClassExplorer
        explorer = ClassExplorer(self.config, str(self.data_dir), self.logger, 
                                self.layer_config, self.num_workers)
        return explorer.analyze_distribution(split, sample_size)
    
    def analyze_layer_distribution(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis distribusi layer dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Hasil analisis distribusi layer
        """
        from smartcash.dataset.services.explorer.layer_explorer import LayerExplorer
        explorer = LayerExplorer(self.config, str(self.data_dir), self.logger, 
                                self.layer_config, self.num_workers)
        return explorer.analyze_distribution(split, sample_size)
    
    def analyze_bbox_statistics(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis statistik bounding box dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Hasil analisis statistik bbox
        """
        from smartcash.dataset.services.explorer.bbox_explorer import BBoxExplorer
        explorer = BBoxExplorer(self.config, str(self.data_dir), self.logger, 
                              self.layer_config, self.num_workers)
        return explorer.analyze_bbox_statistics(split, sample_size)
    
    def analyze_image_sizes(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis ukuran gambar dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Hasil analisis ukuran gambar
        """
        from smartcash.dataset.services.explorer.image_explorer import ImageExplorer
        explorer = ImageExplorer(self.config, str(self.data_dir), self.logger, 
                               self.num_workers)
        return explorer.analyze_image_sizes(split, sample_size)
    
    def visualize_distribution(self, data: Dict[str, Any], output_path: str = None, 
                              title: str = "Distribusi Dataset") -> str:
        """
        Visualisasikan distribusi data dari hasil analisis.
        
        Args:
            data: Data distribusi hasil analisis
            output_path: Path untuk menyimpan visualisasi (opsional)
            title: Judul visualisasi
            
        Returns:
            Path ke file visualisasi
        """
        from smartcash.dataset.visualization.data import DataVisualizationHelper
        
        # Instantiate visualizer dengan direktori output dari hasil analisis
        output_dir = output_path or str(self.data_dir / "visualizations")
        visualizer = DataVisualizationHelper(output_dir, self.logger)
        
        # Identifikasi tipe data yang divisualisasikan
        if 'class_distribution' in data:
            return visualizer.plot_class_distribution(
                data['class_distribution'], 
                title=title or "Distribusi Kelas", 
                save_path=output_path
            )
        elif 'layer_distribution' in data:
            return visualizer.plot_layer_distribution(
                data['layer_distribution'], 
                title=title or "Distribusi Layer", 
                save_path=output_path
            )
        elif 'bbox_statistics' in data:
            # Khusus untuk visualisasi bbox, kita perlu mengonversi data
            # ke dalam format yang dapat divisualisasikan
            bbox_distr = {
                'Small': data['bbox_statistics'].get('small', 0),
                'Medium': data['bbox_statistics'].get('medium', 0),
                'Large': data['bbox_statistics'].get('large', 0)
            }
            return visualizer.plot_size_distribution(
                bbox_distr, 
                title=title or "Distribusi Ukuran Bounding Box", 
                save_path=output_path
            )
        else:
            self.logger.warning("⚠️ Format data tidak dikenali untuk visualisasi")
            return ""
    
    def create_explorer_dashboard(
        self, 
        split: str, 
        sample_size: int = 1000, 
        output_path: str = None
    ) -> str:
        """
        Buat dashboard visualisasi eksplorasi dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            sample_size: Jumlah sampel (0 = semua)
            output_path: Path untuk menyimpan dashboard (opsional)
            
        Returns:
            Path ke file dashboard
        """
        from smartcash.dataset.visualization.report import ReportVisualizer
        
        # Analisis dataset
        class_data = self.analyze_class_distribution(split, sample_size)
        layer_data = self.analyze_layer_distribution(split, sample_size)
        bbox_data = self.analyze_bbox_statistics(split, sample_size)
        
        # Gabungkan data
        report_data = {
            'split': split,
            'sample_size': sample_size,
            'class_distribution': class_data.get('class_distribution', {}),
            'layer_distribution': layer_data.get('layer_distribution', {}),
            'bbox_statistics': bbox_data.get('bbox_statistics', {}),
            'summary': {
                'num_classes': len(class_data.get('class_distribution', {})),
                'num_layers': len(layer_data.get('layer_distribution', {})),
                'avg_bbox_area': bbox_data.get('average_area', 0)
            }
        }
        
        # Instantiate dashboard visualizer
        output_dir = output_path or str(self.data_dir / "reports")
        dashboard = ReportVisualizer(output_dir, self.logger)
        
        # Buat dashboard
        return dashboard.create_dataset_dashboard(
            report_data, 
            save_path=output_path
        )