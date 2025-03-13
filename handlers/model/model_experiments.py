# File: smartcash/handlers/model/model_experiments.py
# Author: Alfrida Sabar
# Deskripsi: Kelas untuk menjalankan dan menganalisis eksperimen model di SmartCash

from typing import Dict, Optional, Any, List, Union
from pathlib import Path
import json
import time

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.exceptions.base import ModelError
from smartcash.utils.visualization import ExperimentVisualizer

class ModelExperiments():
    """
    Kelas untuk melakukan eksperimen model di SmartCash.
    Menyediakan antarmuka untuk membandingkan backbone dan parameter training.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi model experiments."""
        self.config = config
        self.logger = logger or get_logger("model_experiments")
        
        # Lazy-loaded managers
        self._experiment_manager = None
        
        # Setup output directory
        self.output_dir = Path(config.get('output_dir', 'runs/train')) / "experiments"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi visualizer
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
            
        Returns:
            Dict hasil perbandingan
        """
        try:
            # Jalankan eksperimen perbandingan backbone
            results = self.experiment_manager.compare_backbones(
                backbones=backbones,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                parallel=parallel,
                **kwargs
            )
            
            # Buat visualisasi jika diminta
            if visualize and 'results' in results:
                experiment_name = results.get('experiment_name', f"backbone_comparison_{int(time.time())}")
                
                # Visualisasi perbandingan backbone
                viz_paths = self.visualizer.visualize_backbone_comparison(
                    results['results'],
                    title=f"Perbandingan Backbone - {experiment_name}",
                    output_filename=f"{experiment_name}_backbone"
                )
                
                # Tambahkan path visualisasi ke hasil
                results['visualization_paths'] = viz_paths
                
                self.logger.info(f"📊 Visualisasi perbandingan backbone dibuat")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membandingkan backbone: {str(e)}")
            raise ModelError(f"Gagal membandingkan backbone: {str(e)}")
    
    def analyze_results(
        self,
        result_path: Union[str, Path],
        output_format: str = 'markdown'
    ) -> Union[str, Dict[str, Any]]:
        """
        Analisis hasil eksperimen yang sudah disimpan.
        
        Args:
            result_path: Path ke file hasil eksperimen
            output_format: Format output ('markdown', 'json', 'dict')
            
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
        """Buat analisis dalam format Markdown."""
        markdown = f"# Analisis Hasil Eksperimen\n\n"
        
        # Informasi eksperimen
        markdown += "## Informasi Eksperimen\n\n"
        
        for key, value in results.items():
            if key not in ['results', 'summary', 'visualization_paths']:
                markdown += f"- **{key}**: {value}\n"
        
        # Ringkasan
        if 'summary' in results:
            summary = results['summary']
            
            # Backbone terbaik
            if 'best_backbone' in summary:
                markdown += f"\n## 🏆 Backbone Terbaik: {summary['best_backbone']}\n\n"
                markdown += f"- **Metrik**: {summary.get('best_metric', 'unknown')}\n"
                markdown += f"- **Nilai**: {summary.get('best_value', 0):.4f}\n"
        
        # Visualisasi
        if 'visualization_paths' in results:
            viz_paths = results['visualization_paths']
            
            if isinstance(viz_paths, dict):
                markdown += "\n## Visualisasi\n\n"
                
                for name, path in viz_paths.items():
                    if isinstance(path, str):
                        markdown += f"- [{name.replace('_', ' ').title()}]({path})\n"
        
        return markdown