"""
File: smartcash/dataset/services/reporter/export_formatter.py
Deskripsi: Komponen untuk memformat dan mengekspor laporan dataset dalam berbagai format
"""

import os
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from smartcash.common.logger import get_logger


class ExportFormatter:
    """Komponen untuk memformat dan mengekspor laporan dataset dalam berbagai format."""
    
    def __init__(self, output_dir: Optional[str] = None, logger=None):
        """
        Inisialisasi ExportFormatter.
        
        Args:
            output_dir: Direktori output untuk ekspor (opsional)
            logger: Logger kustom (opsional)
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.logger = logger or get_logger("export_formatter")
        
        # Buat direktori jika belum ada
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📄 ExportFormatter diinisialisasi dengan output di: {self.output_dir}")
    
    def export_to_json(self, data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Ekspor data ke format JSON.
        
        Args:
            data: Data yang akan diekspor
            filename: Nama file (opsional)
            
        Returns:
            Path ke file yang diekspor
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dataset_report_{timestamp}.json"
            
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"✅ Data berhasil diekspor ke: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekspor data ke JSON: {str(e)}")
            return ""
    
    def export_to_yaml(self, data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Ekspor data ke format YAML.
        
        Args:
            data: Data yang akan diekspor
            filename: Nama file (opsional)
            
        Returns:
            Path ke file yang diekspor
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dataset_report_{timestamp}.yaml"
            
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
                
            self.logger.info(f"✅ Data berhasil diekspor ke: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekspor data ke YAML: {str(e)}")
            return ""
    
    def export_to_csv(self, data: Dict[str, Any], prefix: Optional[str] = None) -> Dict[str, str]:
        """
        Ekspor data ke format CSV.
        
        Args:
            data: Data yang akan diekspor
            prefix: Prefix untuk nama file (opsional)
            
        Returns:
            Dictionary berisi path file yang diekspor
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = prefix or f"dataset_report_{timestamp}"
        exported_files = {}
        
        try:
            # Ekspor data kelas
            if 'class_metrics' in data and 'class_percentages' in data['class_metrics']:
                class_data = []
                for cls, percentage in data['class_metrics']['class_percentages'].items():
                    # Coba dapatkan count dari data terkait
                    counts = {}
                    for split in data.get('split_stats', {}):
                        if 'class_stats' in data.get('analyses', {}).get(split, {}):
                            counts[split] = data['analyses'][split]['class_stats'].get(cls, 0)
                    
                    class_data.append({
                        'class': cls,
                        'percentage': percentage,
                        **counts
                    })
                
                if class_data:
                    df = pd.DataFrame(class_data)
                    output_path = self.output_dir / f"{prefix}_classes.csv"
                    df.to_csv(output_path, index=False)
                    exported_files['classes'] = str(output_path)
            
            # Ekspor data layer
            if 'layer_metrics' in data and 'layer_percentages' in data['layer_metrics']:
                layer_data = []
                for layer, percentage in data['layer_metrics']['layer_percentages'].items():
                    # Coba dapatkan count dari data terkait
                    counts = {}
                    for split in data.get('split_stats', {}):
                        if 'layer_stats' in data.get('analyses', {}).get(split, {}):
                            counts[split] = data['analyses'][split]['layer_stats'].get(layer, 0)
                    
                    layer_data.append({
                        'layer': layer,
                        'percentage': percentage,
                        **counts
                    })
                
                if layer_data:
                    df = pd.DataFrame(layer_data)
                    output_path = self.output_dir / f"{prefix}_layers.csv"
                    df.to_csv(output_path, index=False)
                    exported_files['layers'] = str(output_path)
            
            # Ekspor statistik split
            if 'split_stats' in data:
                split_data = []
                for split, stats in data['split_stats'].items():
                    split_data.append({
                        'split': split,
                        **stats
                    })
                
                if split_data:
                    df = pd.DataFrame(split_data)
                    output_path = self.output_dir / f"{prefix}_splits.csv"
                    df.to_csv(output_path, index=False)
                    exported_files['splits'] = str(output_path)
            
            if exported_files:
                self.logger.info(f"✅ Data berhasil diekspor ke CSV: {', '.join(exported_files.values())}")
            else:
                self.logger.warning("⚠️ Tidak ada data yang diekspor ke CSV")
                
            return exported_files
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekspor data ke CSV: {str(e)}")
            return exported_files
    
    def export_to_markdown(self, data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Ekspor data ke format Markdown.
        
        Args:
            data: Data yang akan diekspor
            filename: Nama file (opsional)
            
        Returns:
            Path ke file yang diekspor
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dataset_report_{timestamp}.md"
            
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Header
                f.write(f"# Laporan Dataset SmartCash\n\n")
                f.write(f"Dibuat pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Skor kualitas
                if 'quality_score' in data:
                    f.write(f"## Skor Kualitas Dataset: {data['quality_score']:.1f}/100\n\n")
                
                # Ringkasan split
                if 'split_stats' in data:
                    f.write("## Statistik Split\n\n")
                    f.write("| Split | Gambar | Label | Status |\n")
                    f.write("|-------|--------|-------|--------|\n")
                    
                    for split, stats in data['split_stats'].items():
                        f.write(f"| {split} | {stats.get('images', 0)} | {stats.get('labels', 0)} | {stats.get('status', '-')} |\n")
                    
                    f.write("\n")
                
                # Metrik kelas
                if 'class_metrics' in data:
                    cm = data['class_metrics']
                    f.write("## Metrik Kelas\n\n")
                    f.write(f"- Total objek: {cm.get('total_objects', 0)}\n")
                    f.write(f"- Jumlah kelas: {cm.get('num_classes', 0)}\n")
                    f.write(f"- Imbalance score: {cm.get('imbalance_score', 0):.2f}/10\n")
                    f.write(f"- Evenness: {cm.get('evenness', 0):.2f}\n\n")
                    
                    if 'most_common' in cm and 'least_common' in cm:
                        f.write(f"- Kelas paling umum: {cm['most_common'].get('class', '-')} ({cm['most_common'].get('count', 0)} sampel)\n")
                        f.write(f"- Kelas paling jarang: {cm['least_common'].get('class', '-')} ({cm['least_common'].get('count', 0)} sampel)\n\n")
                    
                    # Tabel persentase kelas (top 10)
                    if 'class_percentages' in cm and cm['class_percentages']:
                        f.write("### Distribusi Kelas (Top 10)\n\n")
                        f.write("| Kelas | Persentase |\n")
                        f.write("|-------|------------|\n")
                        
                        sorted_classes = sorted(cm['class_percentages'].items(), key=lambda x: x[1], reverse=True)[:10]
                        for cls, percentage in sorted_classes:
                            f.write(f"| {cls} | {percentage:.1f}% |\n")
                        
                        f.write("\n")
                
                # Metrik layer
                if 'layer_metrics' in data:
                    lm = data['layer_metrics']
                    f.write("## Metrik Layer\n\n")
                    f.write(f"- Total objek: {lm.get('total_objects', 0)}\n")
                    f.write(f"- Jumlah layer: {lm.get('num_layers', 0)}\n")
                    f.write(f"- Imbalance score: {lm.get('imbalance_score', 0):.2f}/10\n\n")
                    
                    # Tabel persentase layer
                    if 'layer_percentages' in lm and lm['layer_percentages']:
                        f.write("### Distribusi Layer\n\n")
                        f.write("| Layer | Persentase |\n")
                        f.write("|-------|------------|\n")
                        
                        sorted_layers = sorted(lm['layer_percentages'].items(), key=lambda x: x[1], reverse=True)
                        for layer, percentage in sorted_layers:
                            f.write(f"| {layer} | {percentage:.1f}% |\n")
                        
                        f.write("\n")
                
                # Metrik Bbox
                if 'bbox_metrics' in data:
                    bm = data['bbox_metrics']
                    f.write("## Metrik Bounding Box\n\n")
                    f.write(f"- Total bounding box: {bm.get('total_bbox', 0)}\n")
                    
                    if 'size_distribution' in bm:
                        sd = bm['size_distribution']
                        f.write(f"- Kecil: {sd.get('small', 0)} ({sd.get('small_pct', 0):.1f}%)\n")
                        f.write(f"- Sedang: {sd.get('medium', 0)} ({sd.get('medium_pct', 0):.1f}%)\n")
                        f.write(f"- Besar: {sd.get('large', 0)} ({sd.get('large_pct', 0):.1f}%)\n\n")
                
                # Rekomendasi
                if 'recommendations' in data and data['recommendations']:
                    f.write("## Rekomendasi\n\n")
                    for rec in data['recommendations']:
                        f.write(f"- {rec}\n")
                    
                    f.write("\n")
                
            self.logger.info(f"✅ Data berhasil diekspor ke: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekspor data ke Markdown: {str(e)}")
            return ""
    
    def export_to_html(self, data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Ekspor data ke format HTML.
        
        Args:
            data: Data yang akan diekspor
            filename: Nama file (opsional)
            
        Returns:
            Path ke file yang diekspor
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dataset_report_{timestamp}.html"
            
        output_path = self.output_dir / filename
        
        try:
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Laporan Dataset SmartCash</title>
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
                    .rec {{ background-color: #f8f9fa; padding: 10px; border-left: 4px solid #5F7FFF; margin-bottom: 10px; }}
                    .quality-score {{ font-size: 32px; font-weight: bold; text-align: center; margin: 20px 0; }}
                    .quality-bar {{ height: 20px; background-color: #e0e0e0; border-radius: 10px; margin-bottom: 20px; }}
                    .quality-value {{ height: 100%; border-radius: 10px; }}
                </style>
            </head>
            <body>
                <h1>Laporan Dataset SmartCash</h1>
                <p>Dibuat pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            """
            
            # Skor kualitas
            if 'quality_score' in data:
                score = data['quality_score']
                color = "#ff4d4d"  # Merah untuk skor rendah
                if score >= 70:
                    color = "#4CAF50"  # Hijau untuk skor baik
                elif score >= 50:
                    color = "#FFC107"  # Kuning untuk skor sedang
                    
                html += f"""
                <div class="card">
                    <h2>Skor Kualitas Dataset</h2>
                    <div class="quality-score">{score:.1f}/100</div>
                    <div class="quality-bar">
                        <div class="quality-value" style="width: {score}%; background-color: {color};"></div>
                    </div>
                </div>
                """
            
            # Statistik split
            if 'split_stats' in data:
                html += """
                <div class="card">
                    <h2>Statistik Split</h2>
                    <table>
                        <tr>
                            <th>Split</th>
                            <th>Gambar</th>
                            <th>Label</th>
                            <th>Status</th>
                        </tr>
                """
                
                for split, stats in data['split_stats'].items():
                    html += f"""
                        <tr>
                            <td>{split}</td>
                            <td>{stats.get('images', 0)}</td>
                            <td>{stats.get('labels', 0)}</td>
                            <td>{stats.get('status', '-')}</td>
                        </tr>
                    """
                    
                html += """
                    </table>
                </div>
                """
            
            # Metrik kelas
            if 'class_metrics' in data:
                cm = data['class_metrics']
                html += f"""
                <div class="card">
                    <h2>Metrik Kelas</h2>
                    <div class="metric">
                        <div class="metric-value">{cm.get('total_objects', 0)}</div>
                        <div class="metric-label">Total Objek</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{cm.get('num_classes', 0)}</div>
                        <div class="metric-label">Jumlah Kelas</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{cm.get('imbalance_score', 0):.1f}/10</div>
                        <div class="metric-label">Imbalance Score</div>
                    </div>
                """
                
                if 'most_common' in cm and 'least_common' in cm:
                    html += f"""
                    <div style="clear: both; margin-top: 20px;">
                        <p><strong>Kelas paling umum:</strong> {cm['most_common'].get('class', '-')} ({cm['most_common'].get('count', 0)} sampel)</p>
                        <p><strong>Kelas paling jarang:</strong> {cm['least_common'].get('class', '-')} ({cm['least_common'].get('count', 0)} sampel)</p>
                    </div>
                    """
                
                # Tabel persentase kelas (top 10)
                if 'class_percentages' in cm and cm['class_percentages']:
                    html += """
                    <h3>Distribusi Kelas (Top 10)</h3>
                    <table>
                        <tr>
                            <th>Kelas</th>
                            <th>Persentase</th>
                        </tr>
                    """
                    
                    sorted_classes = sorted(cm['class_percentages'].items(), key=lambda x: x[1], reverse=True)[:10]
                    for cls, percentage in sorted_classes:
                        html += f"""
                        <tr>
                            <td>{cls}</td>
                            <td>{percentage:.1f}%</td>
                        </tr>
                        """
                        
                    html += """
                    </table>
                    """
                    
                html += "</div>"
            
            # Metrik layer
            if 'layer_metrics' in data:
                lm = data['layer_metrics']
                html += f"""
                <div class="card">
                    <h2>Metrik Layer</h2>
                    <div class="metric">
                        <div class="metric-value">{lm.get('total_objects', 0)}</div>
                        <div class="metric-label">Total Objek</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{lm.get('num_layers', 0)}</div>
                        <div class="metric-label">Jumlah Layer</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{lm.get('imbalance_score', 0):.1f}/10</div>
                        <div class="metric-label">Imbalance Score</div>
                    </div>
                """
                
                # Tabel persentase layer
                if 'layer_percentages' in lm and lm['layer_percentages']:
                    html += """
                    <h3>Distribusi Layer</h3>
                    <table>
                        <tr>
                            <th>Layer</th>
                            <th>Persentase</th>
                        </tr>
                    """
                    
                    sorted_layers = sorted(lm['layer_percentages'].items(), key=lambda x: x[1], reverse=True)
                    for layer, percentage in sorted_layers:
                        html += f"""
                        <tr>
                            <td>{layer}</td>
                            <td>{percentage:.1f}%</td>
                        </tr>
                        """
                        
                    html += """
                    </table>
                    """
                    
                html += "</div>"
            
            # Metrik Bbox
            if 'bbox_metrics' in data:
                bm = data['bbox_metrics']
                html += f"""
                <div class="card">
                    <h2>Metrik Bounding Box</h2>
                    <div class="metric">
                        <div class="metric-value">{bm.get('total_bbox', 0)}</div>
                        <div class="metric-label">Total Bounding Box</div>
                    </div>
                """
                
                if 'size_distribution' in bm:
                    sd = bm['size_distribution']
                    html += f"""
                    <div style="clear: both; margin-top: 20px;">
                        <p><strong>Kecil:</strong> {sd.get('small', 0)} ({sd.get('small_pct', 0):.1f}%)</p>
                        <p><strong>Sedang:</strong> {sd.get('medium', 0)} ({sd.get('medium_pct', 0):.1f}%)</p>
                        <p><strong>Besar:</strong> {sd.get('large', 0)} ({sd.get('large_pct', 0):.1f}%)</p>
                    </div>
                    """
                    
                html += "</div>"
            
            # Rekomendasi
            if 'recommendations' in data and data['recommendations']:
                html += """
                <div class="card">
                    <h2>Rekomendasi</h2>
                """
                
                for rec in data['recommendations']:
                    html += f"""
                    <div class="rec">{rec}</div>
                    """
                    
                html += "</div>"
            
            html += """
            </body>
            </html>
            """
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
                
            self.logger.info(f"✅ Data berhasil diekspor ke: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekspor data ke HTML: {str(e)}")
            return ""
    
    def export_report(self, data: Dict[str, Any], formats: List[str] = None) -> Dict[str, str]:
        """
        Ekspor laporan dalam berbagai format.
        
        Args:
            data: Data yang akan diekspor
            formats: List format yang diinginkan ('json', 'yaml', 'csv', 'md', 'html')
            
        Returns:
            Dictionary berisi path file yang diekspor per format
        """
        formats = formats or ['json', 'html']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"dataset_report_{timestamp}"
        
        results = {}
        
        for fmt in formats:
            if fmt.lower() == 'json':
                path = self.export_to_json(data, f"{filename_base}.json")
                if path:
                    results['json'] = path
                    
            elif fmt.lower() == 'yaml':
                path = self.export_to_yaml(data, f"{filename_base}.yaml")
                if path:
                    results['yaml'] = path
                    
            elif fmt.lower() == 'csv':
                paths = self.export_to_csv(data, filename_base)
                if paths:
                    results['csv'] = paths
                    
            elif fmt.lower() in ['md', 'markdown']:
                path = self.export_to_markdown(data, f"{filename_base}.md")
                if path:
                    results['markdown'] = path
                    
            elif fmt.lower() == 'html':
                path = self.export_to_html(data, f"{filename_base}.html")
                if path:
                    results['html'] = path
                    
            else:
                self.logger.warning(f"⚠️ Format tidak didukung: {fmt}")
        
        if results:
            self.logger.success(f"✅ Laporan berhasil diekspor dalam {len(results)} format")
        else:
            self.logger.error("❌ Tidak ada laporan yang berhasil diekspor")
            
        return results