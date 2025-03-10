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
from smartcash.handlers.dataset.visualizations.distribution_visualizer import DistributionVisualizer
from smartcash.handlers.dataset.visualizations.sample_visualizer import SampleVisualizer


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
        self.visualizers = self._init_visualizers()
        
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
    
    def _init_visualizers(self) -> Dict[str, Any]:
        """Inisialisasi visualizer untuk berbagai jenis visualisasi."""
        return {
            'distribution': DistributionVisualizer(config=self.config, data_dir=str(self.data_dir), 
                                                  output_dir=str(self.output_dir), logger=self.logger),
            'sample': SampleVisualizer(config=self.config, data_dir=str(self.data_dir), 
                                      output_dir=str(self.output_dir), logger=self.logger)
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
        """Buat visualisasi dataset dan kembalikan path file hasil."""
        visualizations = {}
        try:
            visualizations['class_distribution'] = self.visualizers['distribution'].visualize_class_distribution(
                split=split, save_path=str(self.output_dir / f"{split}_class_distribution.png"))
            
            visualizations['layer_distribution'] = self.visualizers['distribution'].visualize_layer_distribution(
                split=split, save_path=str(self.output_dir / f"{split}_layer_distribution.png"))
            
            visualizations['samples'] = self.visualizers['sample'].visualize_samples(
                split=split, num_samples=sample_count, save_path=str(self.output_dir / f"{split}_samples.png"))
        except Exception as e:
            self.logger.error(f"❌ Gagal visualisasi {split}: {str(e)}")
            visualizations['error'] = str(e)
        return visualizations
    
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