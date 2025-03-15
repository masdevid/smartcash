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
        report_format: str = 'json',
        sample_count: int = 5
    ) -> Dict[str, Any]:
        """
        Buat laporan komprehensif tentang dataset.
        
        Args:
            splits: List split dataset yang akan dimasukkan dalam laporan
            visualize: Apakah membuat visualisasi
            calculate_metrics: Apakah menghitung metrik
            save_report: Apakah menyimpan laporan ke file
            report_format: Format laporan ('json', 'html', 'md')
            sample_count: Jumlah sampel yang akan divisualisasikan
            
        Returns:
            Laporan dataset
        """
        start_time = time.time()
        
        # Default ke semua split jika tidak disediakan
        if not splits:
            splits = ['train', 'valid', 'test']
            
        # Inisialisasi report
        report = {
            'generated_at': datetime.datetime.now().isoformat(),
            'data_dir': str(self.data_dir),
            'splits': {},
            'summary': {
                'total_images': 0,
                'total_labels': 0,
                'total_classes': 0,
                'total_layers': 0,
                'class_distribution': {},
                'layer_distribution': {}
            }
        }
        
        # Statistik split dasar
        split_stats = self.utils.get_split_statistics(splits)
        
        # Proses setiap split
        all_class_stats = {}
        all_layer_stats = {}
        
        for split in splits:
            self.logger.info(f"📊 Menganalisis split {split}...")
            
            split_report = {
                'images': split_stats.get(split, {}).get('images', 0),
                'labels': split_stats.get(split, {}).get('labels', 0),
                'status': split_stats.get(split, {}).get('status', 'unknown')
            }
            
            # Skip jika split tidak valid
            if split_report['status'] != 'valid' or split_report['images'] == 0:
                report['splits'][split] = split_report
                continue
                
            # Analisis distribusi kelas
            if calculate_metrics:
                from smartcash.dataset.services.explorer.class_explorer import ClassExplorer
                class_explorer = ClassExplorer(self.config, str(self.data_dir), self.logger, self.num_workers)
                class_result = class_explorer.analyze_distribution(split)
                
                if class_result['status'] == 'success':
                    split_report['class_analysis'] = {
                        'distribution': class_result['counts'],
                        'imbalance_score': class_result['imbalance_score'],
                        'underrepresented_classes': class_result['underrepresented_classes'],
                        'overrepresented_classes': class_result['overrepresented_classes']
                    }
                    
                    # Update all_class_stats
                    for cls, count in class_result['counts'].items():
                        if cls not in all_class_stats:
                            all_class_stats[cls] = {}
                        all_class_stats[cls][split] = count
                    
                # Analisis distribusi layer
                from smartcash.dataset.services.explorer.layer_explorer import LayerExplorer
                layer_explorer = LayerExplorer(self.config, str(self.data_dir), self.logger, self.num_workers)
                layer_result = layer_explorer.analyze_distribution(split)
                
                if layer_result['status'] == 'success':
                    split_report['layer_analysis'] = {
                        'distribution': layer_result['counts'],
                        'imbalance_score': layer_result['imbalance_score'],
                        'layers': layer_result['layers']
                    }
                    
                    # Update all_layer_stats
                    for layer, count in layer_result['counts'].items():
                        if layer not in all_layer_stats:
                            all_layer_stats[layer] = {}
                        all_layer_stats[layer][split] = count
                
                # Analisis bbox
                from smartcash.dataset.services.explorer.bbox_explorer import BBoxExplorer
                bbox_explorer = BBoxExplorer(self.config, str(self.data_dir), self.logger, self.num_workers)
                bbox_result = bbox_explorer.analyze_bbox_statistics(split)
                
                if bbox_result['status'] == 'success':
                    split_report['bbox_analysis'] = {
                        'total_bbox': bbox_result['total_bbox'],
                        'width': bbox_result['width'],
                        'height': bbox_result['height'],
                        'area': bbox_result['area'],
                        'aspect_ratio': bbox_result['aspect_ratio'],
                        'area_categories': bbox_result['area_categories']
                    }
                    
                    # Tambahkan distribusi per layer jika ada
                    if 'by_layer' in bbox_result:
                        split_report['bbox_analysis']['by_layer'] = bbox_result['by_layer']
                        
                    # Tambahkan distribusi per kelas jika ada
                    if 'by_class' in bbox_result:
                        split_report['bbox_analysis']['by_class'] = bbox_result['by_class']
                
                # Analisis gambar
                from smartcash.dataset.services.explorer.image_explorer import ImageExplorer
                image_explorer = ImageExplorer(self.config, str(self.data_dir), self.logger, self.num_workers)
                image_result = image_explorer.analyze_image_sizes(split)
                
                if image_result['status'] == 'success':
                    split_report['image_analysis'] = {
                        'dominant_size': image_result['dominant_size'],
                        'dominant_percentage': image_result['dominant_percentage'],
                        'width_stats': image_result['width_stats'],
                        'height_stats': image_result['height_stats'],
                        'size_categories': image_result['size_categories'],
                        'recommended_size': image_result['recommended_size']
                    }
            
            # Buat visualisasi jika diminta
            if visualize:
                vis_files = []
                
                # Visualisasi distribusi kelas
                if 'class_analysis' in split_report:
                    class_viz_path = self.visualizer.plot_class_distribution(
                        split_report['class_analysis']['distribution'],
                        title=f"Distribusi Kelas - {split}",
                        save_path=f"class_distribution_{split}.png"
                    )
                    vis_files.append(('class_distribution', class_viz_path))
                
                # Visualisasi distribusi layer
                if 'layer_analysis' in split_report:
                    layer_stats = {layer: {'total': count} for layer, count in split_report['layer_analysis']['distribution'].items()}
                    layer_viz_path = self.visualizer.plot_layer_distribution(
                        layer_stats,
                        title=f"Distribusi Layer - {split}",
                        save_path=f"layer_distribution_{split}.png"
                    )
                    vis_files.append(('layer_distribution', layer_viz_path))
                    
                # Visualisasi sampel gambar
                sample_viz_path = self.visualizer.plot_sample_images(
                    str(self.data_dir / split),
                    num_samples=sample_count,
                    save_path=f"sample_images_{split}.png"
                )
                vis_files.append(('sample_images', sample_viz_path))
                
                # Tambahkan path visualisasi ke report
                split_report['visualizations'] = {name: path for name, path in vis_files}
            
            # Tambahkan ke report utama
            report['splits'][split] = split_report
            
            # Update summary
            report['summary']['total_images'] += split_report['images']
            report['summary']['total_labels'] += split_report['labels']
        
        # Kompilasi summary
        report['summary']['class_distribution'] = all_class_stats
        report['summary']['layer_distribution'] = all_layer_stats
        report['summary']['total_classes'] = len(all_class_stats)
        report['summary']['total_layers'] = len(all_layer_stats)
        
        # Hitung durasi
        report['duration'] = time.time() - start_time
        
        # Simpan report jika diminta
        if save_report:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if report_format == 'json':
                report_path = self.reports_dir / f"dataset_report_{timestamp}.json"
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                    
            elif report_format == 'html':
                report_path = self.reports_dir / f"dataset_report_{timestamp}.html"
                self._generate_html_report(report, report_path)
                
            elif report_format == 'md':
                report_path = self.reports_dir / f"dataset_report_{timestamp}.md"
                self._generate_md_report(report, report_path)
                
            else:
                report_path = self.reports_dir / f"dataset_report_{timestamp}.json"
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                    
            self.logger.success(f"📝 Laporan dataset disimpan ke: {report_path}")
        
        self.logger.success(
            f"✅ Analisis dataset selesai ({report['duration']:.1f}s):\n"
            f"   • Total gambar: {report['summary']['total_images']}\n"
            f"   • Total label: {report['summary']['total_labels']}\n"
            f"   • Total kelas: {report['summary']['total_classes']}\n"
            f"   • Total layer: {report['summary']['total_layers']}"
        )
        
        return report
    
    def _generate_html_report(self, report: Dict, output_path: Path) -> None:
        """
        Generate laporan HTML dari data laporan.
        
        Args:
            report: Data laporan
            output_path: Path output untuk laporan HTML
        """
        # Template HTML sederhana
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Dataset Report - {report['generated_at']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; }}
                .metric {{ display: inline-block; margin-right: 20px; margin-bottom: 10px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            </style>
        </head>
        <body>
            <h1>Dataset Report</h1>
            <p>Generated at: {report['generated_at']}</p>
            <p>Data directory: {report['data_dir']}</p>
            
            <div class="card">
                <h2>Summary</h2>
                <div class="metric">
                    <div class="metric-value">{report['summary']['total_images']}</div>
                    <div class="metric-label">Total Images</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report['summary']['total_labels']}</div>
                    <div class="metric-label">Total Labels</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report['summary']['total_classes']}</div>
                    <div class="metric-label">Classes</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report['summary']['total_layers']}</div>
                    <div class="metric-label">Layers</div>
                </div>
            </div>
        """
        
        # Tambahkan bagian untuk setiap split
        for split, split_data in report['splits'].items():
            if split_data['status'] != 'valid':
                html += f"""
                <div class="card">
                    <h2>Split: {split}</h2>
                    <p>Status: {split_data['status']}</p>
                </div>
                """
                continue
                
            html += f"""
            <div class="card">
                <h2>Split: {split}</h2>
                <div class="metric">
                    <div class="metric-value">{split_data['images']}</div>
                    <div class="metric-label">Images</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{split_data['labels']}</div>
                    <div class="metric-label">Labels</div>
                </div>
            """
            
            # Tambahkan visualisasi jika ada
            if 'visualizations' in split_data:
                html += f"""
                <h3>Visualizations</h3>
                <div class="grid">
                """
                
                for name, path in split_data['visualizations'].items():
                    # Konversi path ke path relatif
                    rel_path = os.path.relpath(path, os.path.dirname(output_path))
                    html += f"""
                    <div>
                        <h4>{name.replace('_', ' ').title()}</h4>
                        <img src="{rel_path}" alt="{name}">
                    </div>
                    """
                
                html += "</div>"
                
            # Tambahkan analisis kelas jika ada
            if 'class_analysis' in split_data:
                html += f"""
                <h3>Class Distribution</h3>
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                """
                
                total = sum(split_data['class_analysis']['distribution'].values())
                for cls, count in sorted(split_data['class_analysis']['distribution'].items(), 
                                       key=lambda x: x[1], reverse=True):
                    percentage = (count / total) * 100 if total > 0 else 0
                    html += f"""
                    <tr>
                        <td>{cls}</td>
                        <td>{count}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
                
                # Imbalance score
                html += f"""
                <p>Imbalance Score: {split_data['class_analysis']['imbalance_score']:.2f}/10</p>
                """
            
            html += "</div>"
        
        # Tutup HTML
        html += """
        </body>
        </html>
        """
        
        # Simpan ke file
        with open(output_path, 'w') as f:
            f.write(html)
    
    def _generate_md_report(self, report: Dict, output_path: Path) -> None:
        """
        Generate laporan Markdown dari data laporan.
        
        Args:
            report: Data laporan
            output_path: Path output untuk laporan Markdown
        """
        # Header
        md = f"""# Dataset Report

Generated at: {report['generated_at']}  
Data directory: {report['data_dir']}

## Summary

- **Total Images:** {report['summary']['total_images']}
- **Total Labels:** {report['summary']['total_labels']}
- **Total Classes:** {report['summary']['total_classes']}
- **Total Layers:** {report['summary']['total_layers']}

"""
        
        # Tambahkan bagian untuk setiap split
        for split, split_data in report['splits'].items():
            md += f"## Split: {split}\n\n"
            
            if split_data['status'] != 'valid':
                md += f"Status: {split_data['status']}\n\n"
                continue
                
            md += f"- **Images:** {split_data['images']}\n"
            md += f"- **Labels:** {split_data['labels']}\n\n"
            
            # Tambahkan visualisasi jika ada
            if 'visualizations' in split_data:
                md += "### Visualizations\n\n"
                
                for name, path in split_data['visualizations'].items():
                    # Konversi path ke path relatif
                    rel_path = os.path.relpath(path, os.path.dirname(output_path))
                    md += f"#### {name.replace('_', ' ').title()}\n\n"
                    md += f"![{name}]({rel_path})\n\n"
                
            # Tambahkan analisis kelas jika ada
            if 'class_analysis' in split_data:
                md += "### Class Distribution\n\n"
                md += "| Class | Count | Percentage |\n"
                md += "|-------|-------|------------|\n"
                
                total = sum(split_data['class_analysis']['distribution'].values())
                for cls, count in sorted(split_data['class_analysis']['distribution'].items(), 
                                       key=lambda x: x[1], reverse=True):
                    percentage = (count / total) * 100 if total > 0 else 0
                    md += f"| {cls} | {count} | {percentage:.1f}% |\n"
                    
                md += "\n"
                
                # Imbalance score
                md += f"Imbalance Score: {split_data['class_analysis']['imbalance_score']:.2f}/10\n\n"
            
            # Tambahkan analisis bbox jika ada
            if 'bbox_analysis' in split_data:
                md += "### Bounding Box Analysis\n\n"
                md += f"- Total Bounding Boxes: {split_data['bbox_analysis']['total_bbox']}\n"
                md += f"- Average Width: {split_data['bbox_analysis']['width']['mean']:.4f}\n"
                md += f"- Average Height: {split_data['bbox_analysis']['height']['mean']:.4f}\n"
                md += f"- Average Area: {split_data['bbox_analysis']['area']['mean']:.4f}\n\n"
                
                if 'area_categories' in split_data['bbox_analysis']:
                    cats = split_data['bbox_analysis']['area_categories']
                    md += "#### Size Categories\n\n"
                    md += f"- Small: {cats['small']} ({cats['small_pct']:.1f}%)\n"
                    md += f"- Medium: {cats['medium']} ({cats['medium_pct']:.1f}%)\n"
                    md += f"- Large: {cats['large']} ({cats['large_pct']:.1f}%)\n\n"
        
        # Simpan ke file
        with open(output_path, 'w') as f:
            f.write(md)