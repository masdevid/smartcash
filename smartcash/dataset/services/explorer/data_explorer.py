"""
File: smartcash/dataset/services/explorer/data_explorer.py
Deskripsi: Kelas DataExplorer yang mengintegrasikan semua fitur eksplorasi dataset
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

from smartcash.common.logger import get_logger
from smartcash.common.layer_config import get_layer_config
from smartcash.dataset.services.explorer.class_explorer import ClassExplorer
from smartcash.dataset.services.explorer.layer_explorer import LayerExplorer
from smartcash.dataset.services.explorer.bbox_explorer import BBoxExplorer
from smartcash.dataset.services.explorer.image_explorer import ImageExplorer
from smartcash.dataset.utils.dataset_utils import DatasetUtils


class DataExplorer:
    """Layanan eksplorasi dataset terpadu."""
    
    def __init__(self, config: Dict, data_dir: str, logger=None, num_workers: int = 4):
        """
        Inisialisasi DataExplorer.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori data
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk operasi paralel
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logger or get_logger()
        self.num_workers = num_workers
        
        # Setup utils
        self.utils = DatasetUtils(config, data_dir, logger)
        
        # Setup komponen eksplorasi
        self.class_explorer = ClassExplorer(config, data_dir, logger, num_workers)
        self.layer_explorer = LayerExplorer(config, data_dir, logger, num_workers)
        self.bbox_explorer = BBoxExplorer(config, data_dir, logger, num_workers)
        self.image_explorer = ImageExplorer(config, data_dir, logger, num_workers)
        
        self.logger.info(f"🔍 DataExplorer diinisialisasi dengan {num_workers} workers")
    
    def explore_dataset(self, 
                      split: str = 'train', 
                      sample_size: int = 0,
                      output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Lakukan eksplorasi komprehensif terhadap dataset.
        
        Args:
            split: Split dataset yang akan dieksplorasi
            sample_size: Jumlah sampel (0 = semua)
            output_dir: Direktori output untuk visualisasi (opsional)
            
        Returns:
            Hasil eksplorasi
        """
        start_time = time.time()
        self.logger.info(f"🚀 Memulai eksplorasi dataset {split}...")
        
        output_path = Path(output_dir) if output_dir else self.data_dir / 'exploration'
        os.makedirs(output_path, exist_ok=True)
        
        # Validasi direktori
        split_path = self.utils.get_split_path(split)
        images_dir, labels_dir = split_path / 'images', split_path / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.error(f"❌ Direktori dataset tidak lengkap: {split_path}")
            return {'status': 'error', 'message': f"Direktori tidak lengkap: {split_path}"}
        
        # Analisis paralel
        results = {}
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit tasks
            class_future = executor.submit(
                self.class_explorer.analyze_distribution, split, sample_size)
            layer_future = executor.submit(
                self.layer_explorer.analyze_distribution, split, sample_size)
            bbox_future = executor.submit(
                self.bbox_explorer.analyze_bbox_statistics, split, sample_size)
            image_future = executor.submit(
                self.image_explorer.analyze_image_sizes, split, sample_size)
            
            # Collect results
            results['class_analysis'] = class_future.result()
            results['layer_analysis'] = layer_future.result()
            results['bbox_analysis'] = bbox_future.result()
            results['image_analysis'] = image_future.result()
        
        # Generate visualisasi
        visualizations = self._generate_visualizations(results, split, output_path)
        
        # Kompilasi laporan
        report = {
            'split': split,
            'sample_size': sample_size,
            'status': 'success',
            'duration': time.time() - start_time,
            'analyses': results,
            'visualizations': visualizations
        }
        
        # Generate insight
        report['insights'] = self._generate_insights(results)
        
        self.logger.success(
            f"✅ Eksplorasi dataset {split} selesai dalam {report['duration']:.2f} detik\n"
            f"   • Class: {len(results['class_analysis'].get('counts', {}))} kelas\n"
            f"   • Layer: {len(results['layer_analysis'].get('counts', {}))} layer\n"
            f"   • BBox: {results['bbox_analysis'].get('total_bbox', 0)} bbox\n"
            f"   • Visualisasi: {len(visualizations)} charts"
        )
        
        return report
    
    def _generate_visualizations(self, 
                               results: Dict, 
                               split: str, 
                               output_dir: Path) -> Dict[str, str]:
        """
        Generate visualisasi dari hasil analisis.
        
        Args:
            results: Hasil analisis
            split: Split dataset
            output_dir: Direktori output
            
        Returns:
            Dictionary paths visualisasi
        """
        from smartcash.dataset.visualization.data import DataVisualizationHelper
        
        visualizer = DataVisualizationHelper(str(output_dir), self.logger)
        visualizations = {}
        
        # Class distribution
        if results['class_analysis']['status'] == 'success':
            class_viz_path = visualizer.plot_class_distribution(
                results['class_analysis']['counts'],
                title=f"Distribusi Kelas - {split}",
                save_path=str(output_dir / f"class_distribution_{split}.png")
            )
            visualizations['class_distribution'] = class_viz_path
        
        # Layer distribution
        if results['layer_analysis']['status'] == 'success':
            layer_stats = {layer: {'total': count} 
                         for layer, count in results['layer_analysis']['counts'].items()}
            layer_viz_path = visualizer.plot_layer_distribution(
                layer_stats,
                title=f"Distribusi Layer - {split}",
                save_path=str(output_dir / f"layer_distribution_{split}.png")
            )
            visualizations['layer_distribution'] = layer_viz_path
        
        # Sample images
        sample_viz_path = visualizer.plot_sample_images(
            str(self.data_dir / split),
            num_samples=9,
            save_path=str(output_dir / f"sample_images_{split}.png")
        )
        visualizations['sample_images'] = sample_viz_path
        
        # Bbox size distribution
        if results['bbox_analysis']['status'] == 'success':
            bbox_distr = self._plot_bbox_distribution(
                results['bbox_analysis'],
                split,
                output_dir / f"bbox_distribution_{split}.png"
            )
            visualizations['bbox_distribution'] = bbox_distr
        
        # Advanced visualizations
        if (results['class_analysis']['status'] == 'success' and 
            results['bbox_analysis']['status'] == 'success' and
            'by_class' in results['bbox_analysis']):
            class_vs_bbox = self._plot_class_vs_bbox(
                results['class_analysis']['counts'],
                results['bbox_analysis']['by_class'],
                output_dir / f"class_vs_bbox_{split}.png"
            )
            visualizations['class_vs_bbox'] = class_vs_bbox
        
        return visualizations
    
    def _plot_bbox_distribution(self, 
                              bbox_analysis: Dict, 
                              split: str,
                              output_path: str) -> str:
        """
        Plot distribusi ukuran dan aspek rasio bbox.
        
        Args:
            bbox_analysis: Hasil analisis bbox
            split: Split dataset
            output_path: Path untuk file output
            
        Returns:
            Path ke file visualisasi
        """
        # Setup plot
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Analisis Bounding Box - {split}', fontsize=16)
        
        # Area chart
        area_cats = bbox_analysis['area_categories']
        axs[0, 0].bar(['Kecil', 'Sedang', 'Besar'], 
                     [area_cats['small'], area_cats['medium'], area_cats['large']],
                     color=['#7EB6FF', '#5F7FFF', '#1E47FF'])
        axs[0, 0].set_title('Distribusi Ukuran Bbox')
        axs[0, 0].set_ylabel('Jumlah')
        
        # Aspect ratio histogram
        aspect_data = [bbox_analysis['aspect_ratio']['mean']]
        axs[0, 1].hist(aspect_data, bins=20, alpha=0.7, color='#5F7FFF')
        axs[0, 1].set_title('Distribusi Aspect Ratio')
        axs[0, 1].set_xlabel('Aspect Ratio')
        axs[0, 1].set_ylabel('Frekuensi')
        
        # Width & height stats
        labels = ['Min', 'Max', 'Mean', 'Median']
        width_data = [
            bbox_analysis['width']['min'],
            bbox_analysis['width']['max'],
            bbox_analysis['width']['mean'],
            bbox_analysis['width']['median']
        ]
        height_data = [
            bbox_analysis['height']['min'],
            bbox_analysis['height']['max'],
            bbox_analysis['height']['mean'],
            bbox_analysis['height']['median']
        ]
        
        axs[1, 0].bar(labels, width_data, color='#7EB6FF', alpha=0.7, label='Width')
        axs[1, 0].set_title('Statistik Width')
        axs[1, 0].set_ylabel('Nilai')
        
        axs[1, 1].bar(labels, height_data, color='#5F7FFF', alpha=0.7, label='Height')
        axs[1, 1].set_title('Statistik Height')
        axs[1, 1].set_ylabel('Nilai')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_class_vs_bbox(self,
                          class_counts: Dict[str, int],
                          bbox_by_class: Dict[str, Dict],
                          output_path: str) -> str:
        """
        Plot perbandingan kelas vs ukuran bbox.
        
        Args:
            class_counts: Distribusi kelas
            bbox_by_class: Statistik bbox per kelas
            output_path: Path untuk file output
            
        Returns:
            Path ke file visualisasi
        """
        # Setup plot
        plt.figure(figsize=(12, 8))
        
        # Data preparation
        classes = []
        avg_areas = []
        sizes = []
        
        for cls, stats in bbox_by_class.items():
            if cls in class_counts and 'area' in stats:
                classes.append(cls)
                avg_areas.append(stats['area']['mean'])
                sizes.append(class_counts[cls])
        
        # Penyesuaian ukuran berdasarkan frekuensi
        size_scale = [s * 100 / max(sizes) for s in sizes]
        
        # Plot
        plt.scatter(classes, avg_areas, s=size_scale, alpha=0.6, color='#5F7FFF')
        
        plt.title('Ukuran Rata-rata Bounding Box per Kelas')
        plt.xlabel('Kelas')
        plt.ylabel('Rata-rata Area')
        plt.xticks(rotation=45, ha='right')
        plt.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _generate_insights(self, results: Dict) -> List[str]:
        """
        Generate insight dari hasil analisis.
        
        Args:
            results: Hasil analisis
            
        Returns:
            List insight
        """
        insights = []
        
        # Class imbalance insight
        if (results['class_analysis']['status'] == 'success' and 
            'imbalance_score' in results['class_analysis']):
            
            score = results['class_analysis']['imbalance_score']
            if score > 7:
                insights.append(f"⚠️ Ketidakseimbangan kelas sangat tinggi ({score:.1f}/10). Pertimbangkan untuk melakukan balancing atau augmentasi.")
            elif score > 5:
                insights.append(f"⚠️ Ketidakseimbangan kelas tinggi ({score:.1f}/10). Pertimbangkan untuk melakukan oversampling pada kelas minoritas.")
            
            # Underrepresented classes
            if 'underrepresented_classes' in results['class_analysis']:
                under_classes = results['class_analysis']['underrepresented_classes']
                if under_classes:
                    insights.append(f"📉 Kelas kurang terwakili: {', '.join(under_classes[:3])}{' dan lainnya' if len(under_classes) > 3 else ''}.")
        
        # Bbox size insight
        if (results['bbox_analysis']['status'] == 'success' and 
            'area_categories' in results['bbox_analysis']):
            
            cats = results['bbox_analysis']['area_categories']
            if cats['small_pct'] > 70:
                insights.append(f"📏 Mayoritas ({cats['small_pct']:.1f}%) bounding box berukuran kecil, yang mungkin sulit dideteksi.")
            
            if cats['large_pct'] < 10 and cats['small_pct'] > 50:
                insights.append("📏 Terlalu sedikit bounding box besar. Mungkin perlu augmentasi untuk variasi ukuran yang lebih baik.")
        
        # Image size insight
        if (results['image_analysis']['status'] == 'success' and 
            'recommended_size' in results['image_analysis']):
            
            rec_size = results['image_analysis']['recommended_size']
            insights.append(f"🖼️ Ukuran optimal yang direkomendasikan untuk gambar: {rec_size}.")
        
        # Layer insight
        if (results['layer_analysis']['status'] == 'success' and 
            'imbalance_score' in results['layer_analysis']):
            
            score = results['layer_analysis']['imbalance_score']
            if score > 5:
                insights.append(f"⚠️ Deteksi tidak seimbang antar layer ({score:.1f}/10). Pertimbangkan untuk menyeimbangkan dataset per layer.")
        
        return insights
    
    def export_analysis_report(self, results: Dict, output_format: str = 'markdown') -> str:
        """
        Export hasil analisis ke format yang ditentukan.
        
        Args:
            results: Hasil analisis dari explore_dataset
            output_format: Format output ('markdown', 'html', 'json')
            
        Returns:
            Path ke file output
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        split = results.get('split', 'unknown')
        
        if output_format == 'markdown':
            output_path = self.data_dir / f"exploration_{split}_{timestamp}.md"
            self._export_markdown(results, output_path)
        elif output_format == 'html':
            output_path = self.data_dir / f"exploration_{split}_{timestamp}.html"
            self._export_html(results, output_path)
        else:
            import json
            output_path = self.data_dir / f"exploration_{split}_{timestamp}.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        self.logger.success(f"📝 Laporan eksplorasi disimpan ke: {output_path}")
        return str(output_path)
    
    def _export_markdown(self, results: Dict, output_path: Path) -> None:
        """Export hasil analisis ke format markdown."""
        with open(output_path, 'w') as f:
            f.write(f"# Laporan Eksplorasi Dataset {results['split']}\n\n")
            
            # Metadata
            f.write(f"- **Waktu eksplorasi:** {time.ctime()}\n")
            f.write(f"- **Split:** {results['split']}\n")
            f.write(f"- **Sample size:** {results['sample_size'] or 'Semua'}\n")
            f.write(f"- **Durasi:** {results['duration']:.2f} detik\n\n")
            
            # Insights
            if 'insights' in results and results['insights']:
                f.write("## Insights\n\n")
                for insight in results['insights']:
                    f.write(f"- {insight}\n")
                f.write("\n")
            
            # Class analysis
            if 'class_analysis' in results['analyses']:
                class_analysis = results['analyses']['class_analysis']
                if class_analysis['status'] == 'success':
                    f.write("## Analisis Kelas\n\n")
                    f.write(f"- **Total objek:** {class_analysis['total_objects']}\n")
                    f.write(f"- **Jumlah kelas:** {class_analysis['class_count']}\n")
                    f.write(f"- **Skor ketidakseimbangan:** {class_analysis['imbalance_score']:.2f}/10\n\n")
                    
                    f.write("### Top 5 Kelas\n\n")
                    f.write("| Kelas | Jumlah | Persentase |\n")
                    f.write("|-------|--------|------------|\n")
                    
                    sorted_classes = sorted(class_analysis['counts'].items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
                    
                    for cls, count in sorted_classes:
                        percentage = class_analysis['percentages'].get(cls, 0)
                        f.write(f"| {cls} | {count} | {percentage:.1f}% |\n")
                    
                    f.write("\n")
            
            # Layer analysis
            if 'layer_analysis' in results['analyses']:
                layer_analysis = results['analyses']['layer_analysis']
                if layer_analysis['status'] == 'success':
                    f.write("## Analisis Layer\n\n")
                    f.write(f"- **Total objek:** {layer_analysis['total_objects']}\n")
                    f.write(f"- **Jumlah layer:** {layer_analysis['layer_count']}\n")
                    f.write(f"- **Skor ketidakseimbangan:** {layer_analysis['imbalance_score']:.2f}/10\n\n")
                    
                    f.write("### Distribusi Layer\n\n")
                    f.write("| Layer | Jumlah | Persentase |\n")
                    f.write("|-------|--------|------------|\n")
                    
                    for layer, count in layer_analysis['counts'].items():
                        percentage = layer_analysis['percentages'].get(layer, 0)
                        f.write(f"| {layer} | {count} | {percentage:.1f}% |\n")
                    
                    f.write("\n")
            
            # Bbox analysis
            if 'bbox_analysis' in results['analyses']:
                bbox_analysis = results['analyses']['bbox_analysis']
                if bbox_analysis['status'] == 'success':
                    f.write("## Analisis Bounding Box\n\n")
                    f.write(f"- **Total bounding box:** {bbox_analysis['total_bbox']}\n\n")
                    
                    f.write("### Statistik Ukuran\n\n")
                    f.write("| Dimensi | Min | Max | Mean | Median |\n")
                    f.write("|---------|-----|-----|------|--------|\n")
                    
                    f.write(f"| Width | {bbox_analysis['width']['min']:.4f} | {bbox_analysis['width']['max']:.4f} | {bbox_analysis['width']['mean']:.4f} | {bbox_analysis['width']['median']:.4f} |\n")
                    f.write(f"| Height | {bbox_analysis['height']['min']:.4f} | {bbox_analysis['height']['max']:.4f} | {bbox_analysis['height']['mean']:.4f} | {bbox_analysis['height']['median']:.4f} |\n")
                    f.write(f"| Area | {bbox_analysis['area']['min']:.4f} | {bbox_analysis['area']['max']:.4f} | {bbox_analysis['area']['mean']:.4f} | {bbox_analysis['area']['median']:.4f} |\n")
                    f.write(f"| Aspect Ratio | {bbox_analysis['aspect_ratio']['min']:.4f} | {bbox_analysis['aspect_ratio']['max']:.4f} | {bbox_analysis['aspect_ratio']['mean']:.4f} | {bbox_analysis['aspect_ratio']['median']:.4f} |\n\n")
                    
                    f.write("### Kategori Ukuran\n\n")
                    cats = bbox_analysis['area_categories']
                    f.write(f"- **Kecil:** {cats['small']} ({cats['small_pct']:.1f}%)\n")
                    f.write(f"- **Sedang:** {cats['medium']} ({cats['medium_pct']:.1f}%)\n")
                    f.write(f"- **Besar:** {cats['large']} ({cats['large_pct']:.1f}%)\n\n")
            
            # Image analysis
            if 'image_analysis' in results['analyses']:
                image_analysis = results['analyses']['image_analysis']
                if image_analysis['status'] == 'success':
                    f.write("## Analisis Gambar\n\n")
                    f.write(f"- **Ukuran dominan:** {image_analysis['dominant_size']}\n")
                    f.write(f"- **Persentase ukuran dominan:** {image_analysis['dominant_percentage']:.1f}%\n")
                    f.write(f"- **Ukuran yang direkomendasikan:** {image_analysis['recommended_size']}\n\n")
            
            # Visualizations
            if 'visualizations' in results and results['visualizations']:
                f.write("## Visualisasi\n\n")
                for name, path in results['visualizations'].items():
                    rel_path = os.path.basename(path)
                    f.write(f"### {name.replace('_', ' ').title()}\n\n")
                    f.write(f"![{name}]({rel_path})\n\n")
    
    def _export_html(self, results: Dict, output_path: Path) -> None:
        """Export hasil analisis ke format HTML."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Laporan Eksplorasi Dataset {results['split']}</title>
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
                .insight {{ background-color: #f8f9fa; padding: 10px; border-left: 4px solid #5F7FFF; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <h1>Laporan Eksplorasi Dataset {results['split']}</h1>
            
            <div class="card">
                <h2>Metadata</h2>
                <div class="metric">
                    <div class="metric-label">Waktu Eksplorasi</div>
                    <div>{time.ctime()}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Split</div>
                    <div>{results['split']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Sample Size</div>
                    <div>{results['sample_size'] or 'Semua'}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Durasi</div>
                    <div>{results['duration']:.2f} detik</div>
                </div>
            </div>
        """
        
        # Insights
        if 'insights' in results and results['insights']:
            html += """
            <div class="card">
                <h2>Insights</h2>
            """
            
            for insight in results['insights']:
                html += f"""
                <div class="insight">{insight}</div>
                """
                
            html += """
            </div>
            """
        
        # Class Analysis
        if 'class_analysis' in results['analyses']:
            class_analysis = results['analyses']['class_analysis']
            if class_analysis['status'] == 'success':
                html += f"""
                <div class="card">
                    <h2>Analisis Kelas</h2>
                    <div class="metric">
                        <div class="metric-value">{class_analysis['total_objects']}</div>
                        <div class="metric-label">Total Objek</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{class_analysis['class_count']}</div>
                        <div class="metric-label">Jumlah Kelas</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{class_analysis['imbalance_score']:.2f}/10</div>
                        <div class="metric-label">Skor Ketidakseimbangan</div>
                    </div>
                    
                    <h3>Top 5 Kelas</h3>
                    <table>
                        <tr>
                            <th>Kelas</th>
                            <th>Jumlah</th>
                            <th>Persentase</th>
                        </tr>
                """
                
                sorted_classes = sorted(class_analysis['counts'].items(), 
                                       key=lambda x: x[1], reverse=True)[:5]
                
                for cls, count in sorted_classes:
                    percentage = class_analysis['percentages'].get(cls, 0)
                    html += f"""
                        <tr>
                            <td>{cls}</td>
                            <td>{count}</td>
                            <td>{percentage:.1f}%</td>
                        </tr>
                    """
                    
                html += """
                    </table>
                </div>
                """
        
        # Visualizations
        if 'visualizations' in results and results['visualizations']:
            html += """
            <div class="card">
                <h2>Visualisasi</h2>
                <div class="grid">
            """
            
            for name, path in results['visualizations'].items():
                rel_path = os.path.basename(path)
                html += f"""
                <div>
                    <h3>{name.replace('_', ' ').title()}</h3>
                    <img src="{rel_path}" alt="{name}">
                </div>
                """
                
            html += """
                </div>
            </div>
            """
        
        # Close HTML
        html += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)