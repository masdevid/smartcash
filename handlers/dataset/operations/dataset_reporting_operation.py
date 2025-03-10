# File: smartcash/handlers/dataset/operations/dataset_reporting_operation.py
# Author: Alfrida Sabar
# Deskripsi: Operasi untuk membuat laporan dataset yang komprehensif namun ringkas

import os, json, time, pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from smartcash.utils.logger import get_logger
from smartcash.utils.layer_config_manager import get_layer_config
from smartcash.handlers.dataset.core.dataset_validator import DatasetValidator
from smartcash.handlers.dataset.explorers.base_explorer import BaseExplorer

# Penggunaan komponen visualisasi dari utils/visualization
from smartcash.utils.visualization.metrics import MetricsVisualizer
from smartcash.utils.visualization.detection import DetectionVisualizer


class DatasetReportingOperation:
    """Operasi untuk membuat laporan komprehensif dataset (versi ringkas)."""
    
    def __init__(self, config: Dict, data_dir: str, output_dir: Optional[str] = None, logger=None):
        """Inisialisasi operasi pelaporan dataset."""
        self.config, self.data_dir = config, Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'reports'
        self.logger = logger or get_logger("dataset_reporting")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi komponen 
        self.layer_config = get_layer_config()
        self.validator = DatasetValidator(config, data_dir, logger=self.logger)
        self.explorers = self._init_explorers()
        
        # Inisialisasi visualizers dari utils/visualization
        self.metrics_visualizer = MetricsVisualizer(
            output_dir=str(self.output_dir), 
            logger=self.logger
        )
        self.detection_visualizer = DetectionVisualizer(
            output_dir=str(self.output_dir),
            logger=self.logger
        )
        
        self.logger.info(f"🧮 DatasetReportingOperation diinisialisasi: {self.data_dir}")
    
    def _init_explorers(self) -> Dict[str, BaseExplorer]:
        """Inisialisasi explorer untuk berbagai jenis analisis."""
        from smartcash.handlers.dataset.explorers.validation_explorer import ValidationExplorer
        from smartcash.handlers.dataset.explorers.distribution_explorer import DistributionExplorer
        from smartcash.handlers.dataset.explorers.bbox_image_explorer import BBoxImageExplorer
        
        return {
            'validation': ValidationExplorer(self.config, str(self.data_dir), self.logger),
            'distribution': DistributionExplorer(self.config, str(self.data_dir), self.logger),
            'bbox_image': BBoxImageExplorer(self.config, str(self.data_dir), self.logger)
        }
    
    def generate_dataset_report(self, splits: List[str] = ['train', 'valid', 'test'], 
                              include_validation: bool = True, include_analysis: bool = True,
                              include_visualizations: bool = True, sample_count: int = 9,
                              save_format: str = 'json') -> Dict:
        """Membuat laporan komprehensif tentang dataset."""
        start_time = time.time()
        self.logger.info(f"📊 Memulai pembuatan laporan dataset untuk: {', '.join(splits)}")
        
        # Struktur laporan dasar
        report = {
            'meta': {
                'timestamp': datetime.now().isoformat(), 'data_dir': str(self.data_dir),
                'splits': splits, 'config': {
                    'img_size': self.config.get('model', {}).get('img_size', [640, 640]),
                    'layers': self.config.get('layers', ['banknote'])
                }
            },
            'summary': {}, 'splits': {}
        }
        
        # Proses setiap split
        for split in splits:
            self.logger.info(f"🔍 Memproses split: {split}")
            if not (self.data_dir / split).exists():
                self.logger.warning(f"⚠️ Split {split} tidak ditemukan, melewati...")
                continue
            
            split_report = {}
            if include_validation: split_report['validation'] = self._process_validation(split)
            if include_analysis: split_report['analysis'] = self._process_analysis(split)
            if include_visualizations: split_report['visualizations'] = self._process_visualizations(split, sample_count)
            report['splits'][split] = split_report
        
        # Buat ringkasan dan simpan laporan
        report['summary'] = self._create_dataset_summary(report)
        self._save_report(report, save_format)
        
        # Tambahkan waktu eksekusi
        elapsed_time = time.time() - start_time
        report['meta']['execution_time'] = elapsed_time
        
        self.logger.success(f"✅ Laporan selesai dalam {elapsed_time:.2f} detik\n"
                           f"   • Tersimpan di: {self.output_dir}\n"
                           f"   • Format: {save_format}")
        return report
    
    def _process_validation(self, split: str) -> Dict[str, Any]:
        """Lakukan validasi dataset dan kembalikan hasilnya."""
        try: return self.explorers['validation'].explore(split)
        except Exception as e:
            self.logger.error(f"❌ Gagal validasi split {split}: {str(e)}")
            return {'error': str(e)}
    
    def _process_analysis(self, split: str) -> Dict[str, Any]:
        """Lakukan analisis mendalam dataset dan kembalikan hasilnya."""
        try:
            return {
                'class_balance': self.explorers['distribution'].analyze_class_distribution(split),
                'layer_balance': self.explorers['distribution'].analyze_layer_distribution(split),
                'image_size_distribution': self.explorers['bbox_image'].analyze_image_sizes(split)
            }
        except Exception as e:
            self.logger.error(f"❌ Gagal analisis split {split}: {str(e)}")
            return {'error': str(e)}
    
    def _process_visualizations(self, split: str, sample_count: int) -> Dict[str, str]:
        """Buat visualisasi dataset dan kembalikan path file hasil menggunakan visualizer dari utils/visualization."""
        visualizations = {}
        try:
            # Visualisasi distribusi kelas menggunakan MetricsVisualizer
            class_distribution = self.explorers['distribution'].analyze_class_distribution(split)
            class_data = {
                'classes': list(class_distribution.get('counts', {}).keys()),
                'counts': list(class_distribution.get('counts', {}).values())
            }
            class_vis_path = str(self.output_dir / f"{split}_class_distribution.png")
            self.metrics_visualizer.plot_bar(
                data=class_data,
                x_key='classes',
                y_key='counts', 
                title=f'Distribusi Kelas ({split})',
                filepath=class_vis_path
            )
            visualizations['class_distribution'] = class_vis_path
            
            # Visualisasi distribusi layer menggunakan MetricsVisualizer
            layer_distribution = self.explorers['distribution'].analyze_layer_distribution(split)
            layer_data = {
                'layers': list(layer_distribution.get('counts', {}).keys()),
                'counts': list(layer_distribution.get('counts', {}).values())
            }
            layer_vis_path = str(self.output_dir / f"{split}_layer_distribution.png")
            self.metrics_visualizer.plot_bar(
                data=layer_data,
                x_key='layers',
                y_key='counts',
                title=f'Distribusi Layer ({split})',
                filepath=layer_vis_path
            )
            visualizations['layer_distribution'] = layer_vis_path
            
            # Visualisasi sampel gambar menggunakan DetectionVisualizer
            samples_path = str(self.output_dir / f"{split}_samples.png")
            # Ambil sampel gambar dari split
            sample_images = self._get_sample_images(split, sample_count)
            
            if sample_images:
                # Kombinasikan gambar sampel menggunakan DetectionVisualizer
                self.detection_visualizer.visualize_multiple_images(
                    images=sample_images,
                    filepath=samples_path,
                    title=f'Sampel Gambar ({split})',
                    grid_size=(3, 3)  # Asumsi 3x3 grid untuk 9 gambar
                )
                visualizations['samples'] = samples_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal visualisasi {split}: {str(e)}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def _get_sample_images(self, split: str, count: int = 9):
        """Ambil sampel gambar dari split tertentu."""
        try:
            # Dapatkan daftar gambar dari direktori split
            split_dir = self.data_dir / split / 'images'
            if not split_dir.exists():
                self.logger.warning(f"⚠️ Direktori gambar untuk split {split} tidak ditemukan")
                return []
            
            import cv2
            import random
            from glob import glob
            
            # Dapatkan daftar file gambar
            image_files = glob(f"{split_dir}/*.jpg") + glob(f"{split_dir}/*.png")
            
            # Batasi jumlah sampel ke count atau jumlah total gambar jika lebih kecil
            count = min(count, len(image_files))
            
            # Pilih sampel secara acak
            if count > 0:
                sample_files = random.sample(image_files, count)
                return [cv2.imread(img_file) for img_file in sample_files]
            return []
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mengambil sampel gambar: {str(e)}")
            return []
    
    def _create_dataset_summary(self, report: Dict) -> Dict:
        """Buat ringkasan dataset dari laporan lengkap."""
        summary = {
            'total_samples': 0, 'total_valid_samples': 0, 'total_invalid_samples': 0,
            'splits_overview': {}, 'class_counts': {}, 'layer_distribution': {}, 'quality_scores': {}
        }
        
        # Iterasi setiap split untuk statistik dasar
        for split_name, split_data in report.get('splits', {}).items():
            validation, analysis = split_data.get('validation', {}), split_data.get('analysis', {})
            if not validation and not analysis: continue
                
            # Hitung total sampel
            total_images = validation.get('total_images', 0)
            valid_images = validation.get('valid_images', 0)
            invalid_images = total_images - valid_images
            
            summary['total_samples'] += total_images
            summary['total_valid_samples'] += valid_images
            summary['total_invalid_samples'] += invalid_images
            
            # Hitung skor kualitas (0-100)
            quality_score = 100
            
            # Penalti untuk ketidakseimbangan dan invalid samples
            if 'class_balance' in analysis: 
                quality_score -= analysis['class_balance'].get('imbalance_score', 0) * 2
            if 'layer_balance' in analysis: 
                quality_score -= analysis['layer_balance'].get('imbalance_score', 0) * 2
            if validation:
                invalid_percentage = invalid_images / max(1, total_images) * 100
                quality_score -= min(30, invalid_percentage)
            
            # Batasi skor antara 0-100
            quality_score = max(0, min(100, quality_score))
            summary['quality_scores'][split_name] = quality_score
            
            # Tambahkan overview split
            summary['splits_overview'][split_name] = {
                'total_images': total_images, 'valid_images': valid_images, 'invalid_images': invalid_images,
                'valid_percentage': (valid_images / max(1, total_images)) * 100
            }
        
        # Hitung skor kualitas keseluruhan
        summary['overall_quality_score'] = sum(summary['quality_scores'].values()) / max(1, len(summary['quality_scores']))
        return summary
    
    def _save_report(self, report: Dict, format: str) -> None:
        """Simpan laporan dataset ke file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Simpan sebagai JSON
        if format in ['json', 'both']:
            json_path = self.output_dir / f"dataset_report_{timestamp}.json"
            with open(json_path, 'w') as f: json.dump(report, f, indent=2)
            self.logger.info(f"💾 Laporan tersimpan sebagai JSON: {json_path}")
        
        # Simpan sebagai Markdown
        if format in ['markdown', 'both']:
            md_path = self.output_dir / f"dataset_report_{timestamp}.md"
            markdown_content = self._generate_markdown(report)
            with open(md_path, 'w') as f: f.write(markdown_content)
            self.logger.info(f"💾 Laporan tersimpan sebagai Markdown: {md_path}")
    
    def _generate_markdown(self, report: Dict) -> str:
        """Konversi laporan ke format markdown."""
        lines = [
            "# Laporan Dataset SmartCash", "",
            f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Direktori Data: {report['meta']['data_dir']}", "",
            "## Ringkasan Dataset", ""
        ]
        
        # Ringkasan dasar
        summary = report['summary']
        valid_pct = summary['total_valid_samples'] / max(1, summary['total_samples']) * 100
        invalid_pct = summary['total_invalid_samples'] / max(1, summary['total_samples']) * 100
        
        lines.extend([
            f"- Total sampel: {summary['total_samples']}",
            f"- Sampel valid: {summary['total_valid_samples']} ({valid_pct:.1f}%)",
            f"- Sampel tidak valid: {summary['total_invalid_samples']} ({invalid_pct:.1f}%)",
            f"- Skor kualitas keseluruhan: {summary['overall_quality_score']:.1f}/100", ""
        ])
        
        # Tambahkan rekomendasi berdasarkan skor kualitas
        lines.append("## Rekomendasi")
        lines.append("")
        
        if summary['overall_quality_score'] < 60:
            lines.append("- 🔴 **Kualitas dataset perlu ditingkatkan** secara signifikan")
        elif summary['overall_quality_score'] < 80:
            lines.append("- 🟠 **Kualitas dataset cukup baik** namun perlu perbaikan")
        else:
            lines.append("- 🟢 **Kualitas dataset sudah baik** dan siap untuk training")
            
        return "\n".join(lines)