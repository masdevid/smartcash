"""
File: smartcash/dataset/services/reporter/report_service.py
Deskripsi: Layanan untuk membuat laporan komprehensif tentang dataset
"""

import os
import json
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor

from smartcash.common.logger import get_logger
from smartcash.common.layer_config import get_layer_config
from smartcash.dataset.utils.dataset_utils import DatasetUtils
from smartcash.dataset.visualization.data import DataVisualizationHelper

class ReportService:
    """Layanan untuk membuat laporan komprehensif tentang dataset."""
    
    def __init__(self, config: Dict, data_dir: str, logger=None, num_workers: int = 4):
        """
        Inisialisasi ReportService.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori data
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk proses paralel
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logger or get_logger("report_service")
        self.num_workers = num_workers
        
        # Setup utils
        self.utils = DatasetUtils(config, data_dir, logger)
        
        # Setup output direktori
        self.reports_dir = self.data_dir / 'reports'
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Setup visualizer
        self.visualizer = DataVisualizationHelper(str(self.reports_dir / 'visualizations'), logger)
        
        self.logger.info(f"📊 ReportService diinisialisasi dengan output di {self.reports_dir}")
    
    def generate_dataset_report(
        self,
        splits: List[str] = None,
        visualize: bool = True,
        calculate_metrics: bool = True,
        save_report: bool = True,
        export_formats: List[str] = None,
        sample_count: int = 5
    ) -> Dict[str, Any]:
        """
        Buat laporan komprehensif tentang dataset.
        
        Args:
            splits: List split dataset yang akan dimasukkan dalam laporan
            visualize: Apakah membuat visualisasi
            calculate_metrics: Apakah menghitung metrik
            save_report: Apakah menyimpan laporan ke file
            export_formats: Format laporan untuk diekspor ('json', 'yaml', 'csv', 'md', 'html')
            sample_count: Jumlah sampel yang akan divisualisasikan
            
        Returns:
            Laporan dataset
        """
        start_time = time.time()
        
        # Default ke semua split jika tidak disediakan
        if not splits:
            splits = ['train', 'valid', 'test']
        
        # Default format ekspor
        if not export_formats:
            export_formats = ['json', 'html']
            
        # Inisialisasi report
        report = {
            'generated_at': datetime.datetime.now().isoformat(),
            'data_dir': str(self.data_dir),
            'analyses': {},
            'split_stats': {},
            'visualizations': {}
        }
        
        # Statistik split dasar
        split_stats = self.utils.get_split_statistics(splits)
        report['split_stats'] = split_stats
        
        # Proses setiap split secara paralel jika calculate_metrics
        if calculate_metrics:
            analyses_results = self._analyze_splits_parallel(splits)
            report['analyses'] = analyses_results
            
            # Kompilasi metrik dari hasil analisis
            self._compile_metrics(report)
            
        # Buat visualisasi jika diminta
        if visualize:
            vis_paths = self._generate_visualizations(report, sample_count)
            report['visualizations'] = vis_paths
            
        # Hitung skor kualitas
        if calculate_metrics:
            report['quality_score'] = self._calculate_quality_score(report)
            
            # Generate rekomendasi
            report['recommendations'] = self._generate_recommendations(report)
        
        # Hitung durasi
        report['duration'] = time.time() - start_time
        
        # Simpan report jika diminta
        if save_report:
            from smartcash.dataset.services.reporter.export_formatter import ExportFormatter
            
            formatter = ExportFormatter(str(self.reports_dir), self.logger)
            export_paths = formatter.export_report(report, formats=export_formats)
            
            if export_paths:
                report['export_paths'] = export_paths
                self.logger.success(f"📝 Laporan dataset diekspor dalam {len(export_paths)} format")
        
        self.logger.success(
            f"✅ Laporan dataset selesai dalam {report['duration']:.1f} detik:\n"
            f"   • Splits dianalisis: {', '.join(splits)}\n"
            f"   • Total visualisasi: {len(report.get('visualizations', {}))}"
        )
        
        return report
    
    def _analyze_splits_parallel(self, splits: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Analisis semua split secara paralel.
        
        Args:
            splits: List split yang akan dianalisis
            
        Returns:
            Dictionary berisi hasil analisis per split
        """
        self.logger.info(f"🔍 Menganalisis {len(splits)} split secara paralel...")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(self.num_workers, len(splits))) as executor:
            # Submit tasks
            futures = {executor.submit(self._analyze_split, split): split for split in splits}
            
            # Collect results
            for future in futures:
                split = futures[future]
                try:
                    result = future.result()
                    results[split] = result
                except Exception as e:
                    self.logger.error(f"❌ Error saat menganalisis split {split}: {str(e)}")
                    results[split] = {'status': 'error', 'error': str(e)}
        
        return results
    
    def _analyze_split(self, split: str) -> Dict[str, Any]:
        """
        Analisis satu split dataset.
        
        Args:
            split: Split yang akan dianalisis
            
        Returns:
            Hasil analisis
        """
        split_path = self.utils.get_split_path(split)
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        # Cek validitas split
        if not (images_dir.exists() and labels_dir.exists()):
            return {'status': 'invalid', 'message': 'Direktori tidak lengkap'}
        
        # Analisis distribusi kelas
        from smartcash.dataset.services.explorer.class_explorer import ClassExplorer
        class_explorer = ClassExplorer(self.config, str(self.data_dir), self.logger, self.num_workers)
        class_result = class_explorer.analyze_distribution(split)
        
        # Analisis distribusi layer
        from smartcash.dataset.services.explorer.layer_explorer import LayerExplorer
        layer_explorer = LayerExplorer(self.config, str(self.data_dir), self.logger, self.num_workers)
        layer_result = layer_explorer.analyze_distribution(split)
        
        # Analisis statistik bbox
        from smartcash.dataset.services.explorer.bbox_explorer import BBoxExplorer
        bbox_explorer = BBoxExplorer(self.config, str(self.data_dir), self.logger, self.num_workers)
        bbox_result = bbox_explorer.analyze_bbox_statistics(split)
        
        # Analisis ukuran gambar
        from smartcash.dataset.services.explorer.image_explorer import ImageExplorer
        image_explorer = ImageExplorer(self.config, str(self.data_dir), self.logger, self.num_workers)
        image_result = image_explorer.analyze_image_sizes(split)
        
        # Gabungkan hasil
        result = {
            'status': 'success',
            'class_analysis': class_result if class_result['status'] == 'success' else None,
            'layer_analysis': layer_result if layer_result['status'] == 'success' else None,
            'bbox_analysis': bbox_result if bbox_result['status'] == 'success' else None,
            'image_analysis': image_result if image_result['status'] == 'success' else None
        }
        
        return result
    
    def _compile_metrics(self, report: Dict[str, Any]) -> None:
        """
        Kompilasi metrik dari hasil analisis.
        
        Args:
            report: Laporan dataset yang akan diperbarui
        """
        all_class_stats = {}
        all_layer_stats = {}
        
        # Kumpulkan statistik dari semua split
        for split, analysis in report['analyses'].items():
            if analysis['status'] != 'success':
                continue
                
            # Kumpulkan statistik kelas
            if analysis['class_analysis'] and 'counts' in analysis['class_analysis']:
                class_counts = analysis['class_analysis']['counts']
                for cls, count in class_counts.items():
                    if cls not in all_class_stats:
                        all_class_stats[cls] = {}
                    all_class_stats[cls][split] = count
            
            # Kumpulkan statistik layer
            if analysis['layer_analysis'] and 'counts' in analysis['layer_analysis']:
                layer_counts = analysis['layer_analysis']['counts']
                for layer, count in layer_counts.items():
                    if layer not in all_layer_stats:
                        all_layer_stats[layer] = {}
                    all_layer_stats[layer][split] = count
        
        # Kompilasi metrik menggunakan MetricsReporter
        from smartcash.dataset.services.reporter.metrics_reporter import MetricsReporter
        
        metrics_reporter = MetricsReporter(self.config, self.logger)
        
        # Transformasi statistik untuk metrik kelas
        class_totals = {}
        for cls, split_counts in all_class_stats.items():
            class_totals[cls] = sum(split_counts.values())
        
        # Hitung metrik kelas
        report['class_metrics'] = metrics_reporter.calculate_class_metrics(class_totals)
        
        # Hitung metrik layer
        report['layer_metrics'] = metrics_reporter.calculate_layer_metrics(all_layer_stats)
        
        # Kompilasi metrik bbox dari hasil analisis train split
        bbox_stats = {}
        for split, analysis in report['analyses'].items():
            if analysis['status'] == 'success' and analysis['bbox_analysis']:
                bbox_stats = analysis['bbox_analysis']
                break
                
        if bbox_stats:
            report['bbox_metrics'] = metrics_reporter.calculate_bbox_metrics(bbox_stats)
    
    def _generate_visualizations(self, report: Dict[str, Any], sample_count: int = 5) -> Dict[str, str]:
        """
        Generate visualisasi untuk laporan.
        
        Args:
            report: Laporan dataset
            sample_count: Jumlah sampel yang akan divisualisasikan
            
        Returns:
            Dictionary berisi path visualisasi
        """
        vis_paths = {}
        
        # Directory untuk visualisasi
        vis_dir = self.reports_dir / 'visualizations'
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Visualisasi per split
        for split, analysis in report.get('analyses', {}).items():
            if analysis['status'] != 'success':
                continue
                
            # Visualisasi distribusi kelas
            if analysis['class_analysis'] and 'counts' in analysis['class_analysis']:
                class_viz_path = self.visualizer.plot_class_distribution(
                    analysis['class_analysis']['counts'],
                    title=f"Distribusi Kelas - {split}",
                    save_path=str(vis_dir / f"class_distribution_{split}.png")
                )
                vis_paths[f'class_distribution_{split}'] = class_viz_path
            
            # Visualisasi distribusi layer
            if analysis['layer_analysis'] and 'counts' in analysis['layer_analysis']:
                layer_stats = {layer: {'total': count} for layer, count in analysis['layer_analysis']['counts'].items()}
                layer_viz_path = self.visualizer.plot_layer_distribution(
                    layer_stats,
                    title=f"Distribusi Layer - {split}",
                    save_path=str(vis_dir / f"layer_distribution_{split}.png")
                )
                vis_paths[f'layer_distribution_{split}'] = layer_viz_path
            
            # Visualisasi sampel gambar
            sample_viz_path = self.visualizer.plot_sample_images(
                str(self.data_dir / split),
                num_samples=sample_count,
                save_path=str(vis_dir / f"sample_images_{split}.png")
            )
            vis_paths[f'sample_images_{split}'] = sample_viz_path
        
        # 2. Visualisasi dashboard
        try:
            from smartcash.dataset.visualization.report import ReportVisualizer
            
            report_visualizer = ReportVisualizer(str(vis_dir), self.logger)
            
            # Visualisasi distribusi kelas per split
            if 'class_metrics' in report:
                class_viz_path = report_visualizer.create_class_distribution_summary(
                    report.get('class_metrics', {}).get('class_distribution', {}),
                    save_path=str(vis_dir / "class_distribution_summary.png")
                )
                vis_paths['class_distribution_summary'] = class_viz_path
            
            # Visualisasi dashboard
            dashboard_path = report_visualizer.create_dataset_dashboard(
                report,
                save_path=str(vis_dir / "dataset_dashboard.png")
            )
            vis_paths['dashboard'] = dashboard_path
            
        except ImportError:
            self.logger.warning("⚠️ Modul visualisasi report tidak tersedia, dashboard tidak dibuat")
            
        return vis_paths
    
    def _calculate_quality_score(self, report: Dict[str, Any]) -> float:
        """
        Hitung skor kualitas dataset berdasarkan metrik yang tersedia.
        
        Args:
            report: Laporan dataset
            
        Returns:
            Skor kualitas (0-100)
        """
        # Gunakan MetricsReporter untuk menghitung skor
        from smartcash.dataset.services.reporter.metrics_reporter import MetricsReporter
        
        metrics_reporter = MetricsReporter(self.config, self.logger)
        return metrics_reporter.calculate_dataset_quality_score(report)
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """
        Generate rekomendasi berdasarkan metrik dataset.
        
        Args:
            report: Laporan dataset
            
        Returns:
            List rekomendasi
        """
        from smartcash.dataset.services.reporter.metrics_reporter import MetricsReporter
        
        metrics_reporter = MetricsReporter(self.config, self.logger)
        return metrics_reporter._generate_recommendations(report)
        
    def export_report(self, report: Dict[str, Any], formats: List[str] = None) -> Dict[str, str]:
        """
        Ekspor laporan dataset dalam berbagai format.
        
        Args:
            report: Laporan dataset
            formats: Format ekspor ('json', 'yaml', 'csv', 'md', 'html')
            
        Returns:
            Dictionary berisi path hasil ekspor
        """
        from smartcash.dataset.services.reporter.export_formatter import ExportFormatter
        
        formatter = ExportFormatter(str(self.reports_dir), self.logger)
        return formatter.export_report(report, formats=formats or ['json', 'html'])
    
    def generate_comparison_report(
        self, 
        reports: List[Dict[str, Any]], 
        labels: List[str],
        save_path: Optional[str] = None,
        export_formats: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate laporan perbandingan antara beberapa dataset.
        
        Args:
            reports: List laporan dataset
            labels: Label untuk setiap laporan
            save_path: Path untuk menyimpan laporan (opsional)
            export_formats: Format ekspor ('json', 'yaml', 'csv', 'md', 'html')
            
        Returns:
            Laporan perbandingan
        """
        if len(reports) != len(labels):
            raise ValueError("Jumlah laporan dan label harus sama")
            
        # Default format ekspor
        if not export_formats:
            export_formats = ['json', 'html']
            
        start_time = time.time()
        
        # Inisialisasi laporan
        comparison = {
            'generated_at': datetime.datetime.now().isoformat(),
            'datasets': labels,
            'metrics_comparison': {},
            'class_comparison': {},
            'layer_comparison': {},
            'quality_scores': {}
        }
        
        # Bandingkan skor kualitas
        for i, report in enumerate(reports):
            label = labels[i]
            comparison['quality_scores'][label] = report.get('quality_score', 0)
        
        # Bandingkan distribusi kelas
        for i, report in enumerate(reports):
            label = labels[i]
            
            # Dapatkan metrik kelas
            class_metrics = report.get('class_metrics', {})
            if 'class_percentages' in class_metrics:
                comparison['class_comparison'][label] = class_metrics['class_percentages']
        
        # Bandingkan distribusi layer
        for i, report in enumerate(reports):
            label = labels[i]
            
            # Dapatkan metrik layer
            layer_metrics = report.get('layer_metrics', {})
            if 'layer_percentages' in layer_metrics:
                comparison['layer_comparison'][label] = layer_metrics['layer_percentages']
        
        # Bandingkan statistik split
        comparison['split_comparison'] = {}
        for i, report in enumerate(reports):
            label = labels[i]
            comparison['split_comparison'][label] = report.get('split_stats', {})
        
        # Buat visual perbandingan
        try:
            # Buat direktori untuk visualisasi
            vis_dir = self.reports_dir / 'comparisons'
            os.makedirs(vis_dir, exist_ok=True)
            
            # TODO: Implementasi visualisasi perbandingan
            # Untuk sekarang, buat placeholder
            comparison['visualizations'] = {}
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membuat visualisasi perbandingan: {str(e)}")
        
        # Hitung durasi
        comparison['duration'] = time.time() - start_time
        
        # Simpan laporan
        if save_path:
            # Simpan sebagai JSON
            with open(save_path, 'w') as f:
                json.dump(comparison, f, indent=2)
        
        # Ekspor dalam berbagai format
        if export_formats:
            from smartcash.dataset.services.reporter.export_formatter import ExportFormatter
            
            formatter = ExportFormatter(str(self.reports_dir), self.logger)
            export_paths = formatter.export_report(comparison, formats=export_formats)
            
            if export_paths:
                comparison['export_paths'] = export_paths
        
        self.logger.success(
            f"✅ Laporan perbandingan selesai dalam {comparison['duration']:.1f} detik:\n"
            f"   • Dataset dibandingkan: {', '.join(labels)}"
        )
        
        return comparison