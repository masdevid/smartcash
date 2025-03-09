# File: smartcash/handlers/model/model_experiments.py
# Author: Alfrida Sabar
# Deskripsi: Kelas untuk melakukan berbagai eksperimen model di SmartCash

from typing import Dict, Optional, Any, List, Union, Tuple
from pathlib import Path
import json
import time

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.exceptions.base import ModelError
from smartcash.utils.visualization import ExperimentVisualizer  # Menggunakan visualizer yang sudah ada

class ModelExperiments:
    """
    Kelas untuk melakukan berbagai eksperimen model di SmartCash.
    Menyediakan antarmuka sederhana untuk berbagai jenis eksperimen.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi model experiments.
        
        Args:
            config: Konfigurasi model dan training
            logger: Custom logger (opsional)
        """
        self.config = config
        self.logger = logger or get_logger("model_experiments")
        
        # Lazy-loaded managers
        self._experiment_manager = None
        self._backbone_comparator = None
        
        # Setup output directory
        self.output_dir = Path(config.get('output_dir', 'runs/train')) / "experiments"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi visualizer dari utils/visualization
        self.visualizer = ExperimentVisualizer(
            output_dir=str(self.output_dir / "visualizations")
        )
        
        self.logger.info(f"🧪 ModelExperiments diinisialisasi, output: {self.output_dir}")
    
    @property
    def experiment_manager(self):
        """Lazy-loaded experiment manager."""
        if self._experiment_manager is None:
            from smartcash.handlers.model.experiments.experiment_manager import ExperimentManager
            self._experiment_manager = ExperimentManager(self.config, self.logger)
        return self._experiment_manager
    
    @property
    def backbone_comparator(self):
        """Lazy-loaded backbone comparator."""
        if self._backbone_comparator is None:
            from smartcash.handlers.model.experiments.backbone_comparator import BackboneComparator
            self._backbone_comparator = BackboneComparator(
                self.config, 
                self.logger, 
                self.experiment_manager
            )
        return self._backbone_comparator
    
    def compare_backbones(
        self,
        backbones: List[str],
        train_loader,
        val_loader,
        test_loader = None,
        parallel: bool = False,
        visualize: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Bandingkan beberapa backbone dengan kondisi yang sama.
        
        Args:
            backbones: List backbone untuk dibandingkan
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            test_loader: DataLoader untuk testing (opsional)
            parallel: Jalankan perbandingan secara paralel
            visualize: Buat visualisasi hasil
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil perbandingan
        """
        # Jalankan eksperimen perbandingan backbone
        results = self.experiment_manager.compare_backbones(
            backbones=backbones,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            parallel=parallel,
            **kwargs
        )
        
        # Buat visualisasi jika diminta menggunakan ExperimentVisualizer
        if visualize and 'results' in results:
            experiment_name = results.get('experiment_name', f"backbone_comparison_{int(time.time())}")
            
            # Visualisasi perbandingan backbone
            viz_paths = self.visualizer.visualize_backbone_comparison(
                results['results'],
                title=f"Perbandingan Backbone - {experiment_name}",
                output_filename=f"{experiment_name}_backbone"
            )
            
            # Tambahkan path visualisasi ke hasil
            if 'visualization_paths' not in results:
                results['visualization_paths'] = {}
                
            results['visualization_paths'].update(viz_paths)
            
            # Visualisasi training curves untuk setiap backbone
            for backbone, backbone_results in results['results'].items():
                if 'error' not in backbone_results and 'metrics_history' in backbone_results:
                    training_viz_paths = self.visualizer.visualize_training_curves(
                        backbone_results['metrics_history'],
                        title=f"Training Progress - {backbone}",
                        output_filename=f"{experiment_name}_{backbone}_training"
                    )
                    
                    if backbone not in results['visualization_paths']:
                        results['visualization_paths'][backbone] = {}
                        
                    results['visualization_paths'][backbone].update(training_viz_paths)
            
            self.logger.info(f"📊 Visualisasi perbandingan backbone dibuat: {len(viz_paths)} plot")
        
        return results
    
    def compare_image_sizes(
        self,
        backbones: List[str],
        image_sizes: List[List[int]],
        train_loader,
        val_loader,
        test_loader = None,
        visualize: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Bandingkan backbone dengan ukuran gambar yang berbeda.
        
        Args:
            backbones: List backbone untuk dibandingkan
            image_sizes: List ukuran gambar untuk dibandingkan
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            test_loader: DataLoader untuk testing (opsional)
            visualize: Buat visualisasi hasil
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil perbandingan
        """
        # Jalankan eksperimen perbandingan ukuran gambar
        results = self.backbone_comparator.compare_with_image_sizes(
            backbones=backbones,
            image_sizes=image_sizes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            **kwargs
        )
        
        # Buat visualisasi jika diminta menggunakan ExperimentVisualizer
        if visualize and 'results' in results:
            experiment_name = results.get('experiment_name', f"image_size_comparison_{int(time.time())}")
            
            # Visualisasi untuk setiap backbone
            for backbone, sizes_results in results['results'].items():
                if 'error' in sizes_results:
                    continue
                    
                # Filter hasil yang tidak error
                valid_results = {
                    size: result for size, result in sizes_results.items() 
                    if 'error' not in result
                }
                
                # Visualisasi parameter comparison
                viz_paths = self.visualizer.visualize_parameter_comparison(
                    {backbone: valid_results},
                    parameter_name="Image Size",
                    title=f"{backbone} - Perbandingan Ukuran Gambar",
                    output_filename=f"{experiment_name}_{backbone}_sizes"
                )
                
                # Tambahkan path visualisasi ke hasil
                if 'visualization_paths' not in results:
                    results['visualization_paths'] = {}
                    
                if backbone not in results['visualization_paths']:
                    results['visualization_paths'][backbone] = {}
                    
                results['visualization_paths'][backbone].update(viz_paths)
            
            self.logger.info(f"📊 Visualisasi perbandingan ukuran gambar dibuat")
        
        return results
    
    def compare_augmentations(
        self,
        backbones: List[str],
        augmentation_types: List[str],
        train_loader,
        val_loader,
        test_loader = None,
        visualize: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Bandingkan backbone dengan tipe augmentasi yang berbeda.
        
        Args:
            backbones: List backbone untuk dibandingkan
            augmentation_types: List tipe augmentasi untuk dibandingkan
            train_loader: DataLoader untuk training
            val_loader: DataLoader untuk validasi
            test_loader: DataLoader untuk testing (opsional)
            visualize: Buat visualisasi hasil
            **kwargs: Parameter tambahan
            
        Returns:
            Dict hasil perbandingan
        """
        # Jalankan eksperimen perbandingan augmentasi
        results = self.backbone_comparator.compare_with_augmentations(
            backbones=backbones,
            augmentation_types=augmentation_types,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            **kwargs
        )
        
        # Buat visualisasi jika diminta menggunakan ExperimentVisualizer
        if visualize and 'results' in results:
            experiment_name = results.get('experiment_name', f"augmentation_comparison_{int(time.time())}")
            
            # Visualisasi untuk setiap backbone
            for backbone, aug_results in results['results'].items():
                if 'error' in aug_results:
                    continue
                    
                # Filter hasil yang tidak error
                valid_results = {
                    aug: result for aug, result in aug_results.items()
                    if 'error' not in result
                }
                
                # Visualisasi parameter comparison
                viz_paths = self.visualizer.visualize_parameter_comparison(
                    {backbone: valid_results},
                    parameter_name="Augmentation Type",
                    title=f"{backbone} - Perbandingan Tipe Augmentasi",
                    output_filename=f"{experiment_name}_{backbone}_augmentations"
                )
                
                # Tambahkan path visualisasi ke hasil
                if 'visualization_paths' not in results:
                    results['visualization_paths'] = {}
                    
                if backbone not in results['visualization_paths']:
                    results['visualization_paths'][backbone] = {}
                    
                results['visualization_paths'][backbone].update(viz_paths)
            
            self.logger.info(f"📊 Visualisasi perbandingan tipe augmentasi dibuat")
        
        return results
    
    def analyze_results(
        self,
        result_path: Union[str, Path],
        output_format: str = 'markdown',
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Analisis hasil eksperimen yang sudah disimpan.
        
        Args:
            result_path: Path ke file hasil eksperimen
            output_format: Format output ('markdown', 'json', 'dict')
            **kwargs: Parameter tambahan
            
        Returns:
            Hasil analisis dalam format yang diminta
        """
        try:
            # Load hasil eksperimen
            path = Path(result_path)
            if not path.exists():
                raise ModelError(f"File hasil eksperimen tidak ditemukan: {path}")
            
            # Baca hasil
            with open(path, 'r') as f:
                results = json.load(f)
            
            # Proses hasil analisis menjadi format yang diminta
            if output_format == 'json':
                return json.dumps(results, indent=2)
            elif output_format == 'dict':
                return results
            else:  # markdown
                return self._create_markdown_analysis(results)
        
        except Exception as e:
            self.logger.error(f"❌ Gagal menganalisis hasil eksperimen: {str(e)}")
            raise ModelError(f"Gagal menganalisis hasil eksperimen: {str(e)}")
    
    def _create_markdown_analysis(self, results: Dict[str, Any]) -> str:
        """
        Buat analisis dalam format Markdown.
        
        Args:
            results: Hasil eksperimen
            
        Returns:
            Markdown analisis
        """
        markdown = f"# Analisis Hasil Eksperimen\n\n"
        
        # Informasi eksperimen
        markdown += "## Informasi Eksperimen\n\n"
        
        for key, value in results.items():
            if key not in ['results', 'summary', 'visualization_paths']:
                markdown += f"- **{key}**: {value}\n"
        
        # Ringkasan
        if 'summary' in results:
            summary = results['summary']
            
            # Backbone atau kombinasi terbaik
            if 'best_backbone' in summary:
                markdown += f"\n## 🏆 Backbone Terbaik: {summary['best_backbone']}\n\n"
                markdown += f"- **Metrik**: {summary.get('best_metric', 'unknown')}\n"
                markdown += f"- **Nilai**: {summary.get('best_value', 0):.4f}\n"
            
            # Perbandingan metrik
            if 'metrics_comparison' in summary:
                markdown += "\n## Perbandingan Metrik\n\n"
                
                for metric, data in summary['metrics_comparison'].items():
                    if not isinstance(data, dict):
                        continue
                        
                    markdown += f"### {metric}\n\n"
                    markdown += f"- **Best**: {data.get('best_backbone', 'N/A')} ({data.get('best_value', 0):.4f})\n"
                    markdown += f"- **Average**: {data.get('average', 0):.4f}\n"
        
        # Visualisasi
        if 'visualization_paths' in results:
            viz_paths = results['visualization_paths']
            
            if isinstance(viz_paths, dict):
                markdown += "\n## Visualisasi\n\n"
                
                # Visualisasi umum
                for name, path in viz_paths.items():
                    if isinstance(path, str):
                        # Dapatkan path relatif jika mungkin
                        rel_path = path
                        markdown += f"- [{name.replace('_', ' ').title()}]({rel_path})\n"
                    elif isinstance(path, dict):
                        # Lewati, ini backbone atau parameter tertentu
                        pass
                
                # Visualisasi per backbone
                for name, paths in viz_paths.items():
                    if isinstance(paths, dict):
                        markdown += f"\n### {name}\n\n"
                        for viz_name, viz_path in paths.items():
                            rel_path = viz_path
                            markdown += f"- [{viz_name.replace('_', ' ').title()}]({rel_path})\n"
        
        return markdown