# File: smartcash/handlers/dataset/operations/dataset_reporting_operation.py
# Author: Alfrida Sabar
# Deskripsi: Operasi untuk membuat laporan dataset yang komprehensif

import os
import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

from smartcash.utils.logger import get_logger
from smartcash.utils.layer_config_manager import get_layer_config
from smartcash.handlers.dataset.core.dataset_validator import DatasetValidator
from smartcash.handlers.dataset.core.dataset_explorer import DatasetExplorer
from smartcash.handlers.dataset.visualizations.distribution_visualizer import DistributionVisualizer
from smartcash.handlers.dataset.visualizations.sample_visualizer import SampleVisualizer


class DatasetReportingOperation:
    """
    Operasi untuk membuat laporan dataset yang komprehensif,
    menggabungkan validasi, analisis, dan visualisasi.
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: str,
        output_dir: Optional[str] = None,
        logger=None
    ):
        """
        Inisialisasi operasi pelaporan dataset.
        
        Args:
            config: Konfigurasi dataset
            data_dir: Direktori dataset
            output_dir: Direktori output untuk laporan
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'reports'
        self.logger = logger or get_logger("dataset_reporting")
        
        # Buat direktori output jika belum ada
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi komponen yang dibutuhkan
        self.layer_config = get_layer_config()
        self.validator = DatasetValidator(config, data_dir, logger=self.logger)
        self.explorer = DatasetExplorer(config, data_dir, logger=self.logger)
        
        # Inisialisasi visualizer
        self.distribution_visualizer = DistributionVisualizer(
            config=config,
            data_dir=data_dir,
            output_dir=str(self.output_dir),
            logger=self.logger
        )
        
        self.sample_visualizer = SampleVisualizer(
            config=config,
            data_dir=data_dir,
            output_dir=str(self.output_dir),
            logger=self.logger
        )
        
        self.logger.info(f"🧮 DatasetReportingOperation diinisialisasi: {self.data_dir}")
    
    def generate_dataset_report(
        self,
        splits: List[str] = ['train', 'valid', 'test'],
        include_validation: bool = True,
        include_analysis: bool = True,
        include_visualizations: bool = True,
        sample_count: int = 9,
        save_format: str = 'json'
    ) -> Dict:
        """
        Membuat laporan komprehensif tentang dataset.
        
        Args:
            splits: List split dataset yang akan dianalisis
            include_validation: Apakah menyertakan validasi dataset
            include_analysis: Apakah menyertakan analisis mendalam
            include_visualizations: Apakah menyertakan visualisasi
            sample_count: Jumlah sampel gambar untuk visualisasi
            save_format: Format penyimpanan ('json', 'markdown', 'both')
            
        Returns:
            Dict laporan dataset
        """
        start_time = time.time()
        
        self.logger.info(f"📊 Memulai pembuatan laporan dataset untuk split: {', '.join(splits)}")
        
        # Struktur laporan
        report = {
            'meta': {
                'timestamp': datetime.now().isoformat(),
                'data_dir': str(self.data_dir),
                'splits': splits,
                'config': {
                    'img_size': self.config.get('model', {}).get('img_size', [640, 640]),
                    'layers': self.config.get('layers', ['banknote'])
                }
            },
            'summary': {},
            'splits': {}
        }
        
        # Proses setiap split
        for split in splits:
            self.logger.info(f"🔍 Memproses split: {split}")
            
            split_report = {}
            split_dir = self.data_dir / split
            
            # Skip jika direktori tidak ada
            if not split_dir.exists():
                self.logger.warning(f"⚠️ Split {split} tidak ditemukan, melewati...")
                continue
            
            # 1. Validasi dataset
            if include_validation:
                self.logger.info(f"🛠️ Melakukan validasi split {split}")
                try:
                    validation_results = self.validator.validate_dataset(split)
                    split_report['validation'] = validation_results
                except Exception as e:
                    self.logger.error(f"❌ Gagal melakukan validasi: {str(e)}")
                    split_report['validation'] = {'error': str(e)}
            
            # 2. Analisis dataset
            if include_analysis:
                self.logger.info(f"🧪 Melakukan analisis mendalam split {split}")
                try:
                    analysis_results = self.explorer.analyze_dataset(split)
                    split_report['analysis'] = analysis_results
                except Exception as e:
                    self.logger.error(f"❌ Gagal melakukan analisis: {str(e)}")
                    split_report['analysis'] = {'error': str(e)}
            
            # 3. Visualisasi dataset
            if include_visualizations:
                self.logger.info(f"🎨 Membuat visualisasi untuk split {split}")
                visualizations = {}
                
                try:
                    # Visualisasi distribusi kelas
                    class_dist_path = self.distribution_visualizer.visualize_class_distribution(
                        split=split,
                        save_path=str(self.output_dir / f"{split}_class_distribution.png")
                    )
                    visualizations['class_distribution'] = class_dist_path
                    
                    # Visualisasi distribusi layer
                    layer_dist_path = self.distribution_visualizer.visualize_layer_distribution(
                        split=split,
                        save_path=str(self.output_dir / f"{split}_layer_distribution.png")
                    )
                    visualizations['layer_distribution'] = layer_dist_path
                    
                    # Visualisasi sampel gambar
                    samples_path = self.sample_visualizer.visualize_samples(
                        split=split,
                        num_samples=sample_count,
                        save_path=str(self.output_dir / f"{split}_samples.png")
                    )
                    visualizations['samples'] = samples_path
                    
                    split_report['visualizations'] = visualizations
                except Exception as e:
                    self.logger.error(f"❌ Gagal membuat visualisasi: {str(e)}")
                    split_report['visualizations'] = {'error': str(e)}
            
            # Tambahkan ke laporan lengkap
            report['splits'][split] = split_report
        
        # Buat ringkasan dataset
        summary = self._create_dataset_summary(report)
        report['summary'] = summary
        
        # Simpan laporan
        self._save_report(report, save_format)
        
        # Hitung waktu eksekusi
        elapsed_time = time.time() - start_time
        report['meta']['execution_time'] = elapsed_time
        
        self.logger.success(
            f"✅ Laporan dataset selesai dibuat dalam {elapsed_time:.2f} detik\n"
            f"   • Tersimpan di: {self.output_dir}\n"
            f"   • Format: {save_format}"
        )
        
        return report
    
    def _create_dataset_summary(self, report: Dict) -> Dict:
        """
        Membuat ringkasan dataset dari laporan lengkap.
        
        Args:
            report: Laporan dataset lengkap
            
        Returns:
            Dict berisi ringkasan dataset
        """
        summary = {
            'total_samples': 0,
            'total_valid_samples': 0,
            'total_invalid_samples': 0,
            'splits_overview': {},
            'class_counts': {},
            'layer_distribution': {},
            'quality_scores': {}
        }
        
        # Iterasi setiap split
        for split_name, split_data in report.get('splits', {}).items():
            validation = split_data.get('validation', {})
            analysis = split_data.get('analysis', {})
            
            # Skip jika tidak ada data
            if not validation and not analysis:
                continue
            
            # Hitung total sampel
            total_images = validation.get('total_images', 0)
            valid_images = validation.get('valid_images', 0)
            invalid_images = total_images - valid_images
            
            summary['total_samples'] += total_images
            summary['total_valid_samples'] += valid_images
            summary['total_invalid_samples'] += invalid_images
            
            # Tambahkan ke ringkasan split
            summary['splits_overview'][split_name] = {
                'total_images': total_images,
                'valid_images': valid_images,
                'invalid_images': invalid_images,
                'valid_percentage': (valid_images / max(1, total_images)) * 100
            }
            
            # Persiapkan skor kualitas
            quality_score = 100
            
            # Penalti untuk ketidakseimbangan kelas
            if 'class_balance' in analysis:
                class_imbalance = analysis['class_balance'].get('imbalance_score', 0)
                quality_score -= class_imbalance * 2  # Penalti maksimum 20 poin
                
                # Tambahkan ke ringkasan kelas jika belum ada
                class_dist = analysis['class_balance'].get('class_percentages', {})
                for cls, percentage in class_dist.items():
                    if cls not in summary['class_counts']:
                        summary['class_counts'][cls] = {}
                    summary['class_counts'][cls][split_name] = percentage
            
            # Penalti untuk ketidakseimbangan layer
            if 'layer_balance' in analysis:
                layer_imbalance = analysis['layer_balance'].get('imbalance_score', 0)
                quality_score -= layer_imbalance * 2  # Penalti maksimum 20 poin
                
                # Tambahkan ke ringkasan layer jika belum ada
                layer_dist = analysis['layer_balance'].get('layer_percentages', {})
                for layer, percentage in layer_dist.items():
                    if layer not in summary['layer_distribution']:
                        summary['layer_distribution'][layer] = {}
                    summary['layer_distribution'][layer][split_name] = percentage
            
            # Penalti untuk masalah validasi
            if validation:
                invalid_percentage = invalid_images / max(1, total_images) * 100
                quality_score -= min(30, invalid_percentage)  # Penalti maksimum 30 poin
            
            # Pastikan skor dalam rentang 0-100
            quality_score = max(0, min(100, quality_score))
            
            # Tambahkan ke ringkasan
            summary['quality_scores'][split_name] = quality_score
        
        # Hitung skor kualitas keseluruhan
        if summary['quality_scores']:
            summary['overall_quality_score'] = sum(summary['quality_scores'].values()) / len(summary['quality_scores'])
        else:
            summary['overall_quality_score'] = 0
        
        return summary
    
    def _save_report(self, report: Dict, format: str) -> None:
        """
        Simpan laporan dataset ke file.
        
        Args:
            report: Laporan dataset lengkap
            format: Format penyimpanan ('json', 'markdown', 'both')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Simpan sebagai JSON
        if format in ['json', 'both']:
            json_path = self.output_dir / f"dataset_report_{timestamp}.json"
            try:
                with open(json_path, 'w') as f:
                    json.dump(report, f, indent=2)
                self.logger.info(f"💾 Laporan tersimpan sebagai JSON: {json_path}")
            except Exception as e:
                self.logger.error(f"❌ Gagal menyimpan JSON: {str(e)}")
        
        # Simpan sebagai Markdown
        if format in ['markdown', 'both']:
            md_path = self.output_dir / f"dataset_report_{timestamp}.md"
            try:
                markdown_content = self._generate_markdown(report)
                with open(md_path, 'w') as f:
                    f.write(markdown_content)
                self.logger.info(f"💾 Laporan tersimpan sebagai Markdown: {md_path}")
            except Exception as e:
                self.logger.error(f"❌ Gagal menyimpan Markdown: {str(e)}")
    
    def _generate_markdown(self, report: Dict) -> str:
        """
        Generate laporan dalam format markdown.
        
        Args:
            report: Laporan dataset lengkap
            
        Returns:
            String laporan dalam format markdown
        """
        # Header
        lines = [
            "# Laporan Dataset SmartCash",
            "",
            f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Direktori Data: {report['meta']['data_dir']}",
            "",
            "## Ringkasan Dataset",
            ""
        ]
        
        # Ringkasan
        summary = report['summary']
        lines.extend([
            f"- Total sampel: {summary['total_samples']}",
            f"- Sampel valid: {summary['total_valid_samples']} ({summary['total_valid_samples'] / max(1, summary['total_samples']) * 100:.1f}%)",
            f"- Sampel tidak valid: {summary['total_invalid_samples']} ({summary['total_invalid_samples'] / max(1, summary['total_samples']) * 100:.1f}%)",
            f"- Skor kualitas keseluruhan: {summary['overall_quality_score']:.1f}/100",
            ""
        ])
        
        # Tabel ringkasan per split
        lines.extend([
            "### Ringkasan per Split",
            "",
            "| Split | Total | Valid | Invalid | % Valid | Skor Kualitas |",
            "|-------|-------|-------|---------|---------|---------------|"
        ])
        
        for split, data in summary['splits_overview'].items():
            quality_score = summary['quality_scores'].get(split, 0)
            lines.append(
                f"| {split} | {data['total_images']} | {data['valid_images']} | {data['invalid_images']} | "
                f"{data['valid_percentage']:.1f}% | {quality_score:.1f}/100 |"
            )
        
        lines.append("")
        
        # Detail per split
        for split_name, split_data in report['splits'].items():
            lines.extend([
                f"## Detail Split: {split_name}",
                ""
            ])
            
            # Tambahkan hasil validasi
            if 'validation' in split_data:
                validation = split_data['validation']
                if 'error' in validation:
                    lines.extend([
                        "### Validasi Dataset",
                        "",
                        f"**Error:** {validation['error']}",
                        ""
                    ])
                else:
                    lines.extend([
                        "### Validasi Dataset",
                        "",
                        f"- Total gambar: {validation.get('total_images', 0)}",
                        f"- Gambar valid: {validation.get('valid_images', 0)}",
                        f"- Label valid: {validation.get('valid_labels', 0)}",
                        f"- Label tidak valid: {validation.get('invalid_labels', 0)}",
                        f"- Label hilang: {validation.get('missing_labels', 0)}",
                        ""
                    ])
            
            # Tambahkan hasil analisis
            if 'analysis' in split_data:
                analysis = split_data['analysis']
                if 'error' in analysis:
                    lines.extend([
                        "### Analisis Dataset",
                        "",
                        f"**Error:** {analysis['error']}",
                        ""
                    ])
                else:
                    lines.extend([
                        "### Analisis Dataset",
                        "",
                        "#### Distribusi Kelas",
                        ""
                    ])
                    
                    # Tambahkan tabel distribusi kelas
                    if 'class_balance' in analysis:
                        class_balance = analysis['class_balance']
                        lines.extend([
                            f"Skor ketidakseimbangan: {class_balance.get('imbalance_score', 0):.1f}/10",
                            "",
                            "| Kelas | Persentase |",
                            "|-------|------------|"
                        ])
                        
                        class_percentages = class_balance.get('class_percentages', {})
                        for cls, pct in sorted(class_percentages.items(), key=lambda x: x[1], reverse=True)[:10]:
                            lines.append(f"| {cls} | {pct:.1f}% |")
                        
                        lines.append("")
                    
                    # Tambahkan tabel distribusi layer
                    lines.extend([
                        "#### Distribusi Layer",
                        ""
                    ])
                    
                    if 'layer_balance' in analysis:
                        layer_balance = analysis['layer_balance']
                        lines.extend([
                            f"Skor ketidakseimbangan: {layer_balance.get('imbalance_score', 0):.1f}/10",
                            "",
                            "| Layer | Persentase |",
                            "|-------|------------|"
                        ])
                        
                        layer_percentages = layer_balance.get('layer_percentages', {})
                        for layer, pct in sorted(layer_percentages.items(), key=lambda x: x[1], reverse=True):
                            lines.append(f"| {layer} | {pct:.1f}% |")
                        
                        lines.append("")
            
            # Tambahkan visualisasi
            if 'visualizations' in split_data:
                visualizations = split_data['visualizations']
                if 'error' in visualizations:
                    lines.extend([
                        "### Visualisasi Dataset",
                        "",
                        f"**Error:** {visualizations['error']}",
                        ""
                    ])
                else:
                    lines.extend([
                        "### Visualisasi Dataset",
                        "",
                        "Visualisasi tersedia di direktori berikut:",
                        ""
                    ])
                    
                    for viz_name, viz_path in visualizations.items():
                        if viz_path:
                            lines.append(f"- {viz_name}: [{os.path.basename(viz_path)}]({viz_path})")
                    
                    lines.append("")
        
        # Rekomendasi
        lines.extend([
            "## Rekomendasi",
            ""
        ])
        
        quality_score = summary.get('overall_quality_score', 0)
        if quality_score < 60:
            lines.append("- 🔴 **Kualitas dataset perlu ditingkatkan** secara signifikan sebelum training")
        elif quality_score < 80:
            lines.append("- 🟠 **Kualitas dataset cukup baik** namun masih ada ruang untuk perbaikan")
        else:
            lines.append("- 🟢 **Kualitas dataset sudah baik** dan siap untuk training")
        
        # Tambahkan rekomendasi spesifik
        if 'class_counts' in summary:
            # Cek kelas minoritas/mayoritas
            if any(split_data.get('analysis', {}).get('class_balance', {}).get('imbalance_score', 0) > 5 
                  for split_name, split_data in report['splits'].items()):
                lines.append("- ⚖️ **Ketidakseimbangan kelas terdeteksi**, pertimbangkan untuk melakukan class balancing")
        
        if summary['total_invalid_samples'] > 0.1 * summary['total_samples']:
            lines.append("- 🛠️ **Terlalu banyak sampel tidak valid**, disarankan untuk memperbaiki dataset")
        
        # Selesai
        return "\n".join(lines)