# File: smartcash/handlers/evaluation/core/report_generator.py
# Author: Alfrida Sabar
# Deskripsi: Generator laporan hasil evaluasi model

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import datetime

from smartcash.utils.logger import SmartCashLogger, get_logger

class ReportGenerator:
    """
    Generator laporan hasil evaluasi model.
    Menghasilkan laporan dalam berbagai format (JSON, CSV, Markdown, HTML).
    """
    
    def __init__(
        self, 
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi generator laporan.
        
        Args:
            config: Konfigurasi evaluasi
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.logger = logger or get_logger("report_generator")
        
        # Setup direktori output
        self.output_dir = Path(self.config.get('output_dir', 'results/evaluation/reports'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        results: Dict[str, Any],
        format: str = 'json',
        output_path: Optional[str] = None,
        include_plots: bool = True,
        **kwargs
    ) -> str:
        """
        Generate laporan evaluasi.
        
        Args:
            results: Hasil evaluasi
            format: Format laporan ('json', 'csv', 'markdown', 'html')
            output_path: Path output laporan (opsional)
            include_plots: Sertakan visualisasi (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Path ke laporan yang dihasilkan
        """
        # Tentukan output path jika belum ada
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.output_dir / f"evaluation_report_{timestamp}.{format}")
        
        # Buat path output jika belum ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate laporan sesuai format
        if format.lower() == 'json':
            return self._generate_json(results, output_path, **kwargs)
        elif format.lower() == 'csv':
            return self._generate_csv(results, output_path, **kwargs)
        elif format.lower() == 'markdown' or format.lower() == 'md':
            return self._generate_markdown(results, output_path, include_plots, **kwargs)
        elif format.lower() == 'html':
            return self._generate_html(results, output_path, include_plots, **kwargs)
        else:
            self.logger.warning(f"⚠️ Format laporan tidak didukung: {format}, menggunakan JSON")
            return self._generate_json(results, output_path, **kwargs)
    
    def _generate_json(
        self,
        results: Dict[str, Any],
        output_path: str,
        **kwargs
    ) -> str:
        """
        Generate laporan format JSON.
        
        Args:
            results: Hasil evaluasi
            output_path: Path output laporan
            **kwargs: Parameter tambahan
            
        Returns:
            Path ke laporan JSON
        """
        try:
            # Buat deep copy dari results
            report_data = self._prepare_report_data(results)
            
            # Tulis ke file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.success(f"✅ Laporan JSON berhasil dibuat: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat laporan JSON: {str(e)}")
            raise
    
    def _generate_csv(
        self,
        results: Dict[str, Any],
        output_path: str,
        **kwargs
    ) -> str:
        """
        Generate laporan format CSV.
        
        Args:
            results: Hasil evaluasi
            output_path: Path output laporan
            **kwargs: Parameter tambahan
            
        Returns:
            Path ke laporan CSV
        """
        try:
            # Ekstrak metrics dari hasil
            metrics_data = []
            
            # Deteksi jenis hasil (model tunggal, batch, atau research)
            if 'metrics' in results:
                # Model tunggal
                metrics_data.append({
                    'model': os.path.basename(results.get('model_path', 'unknown')),
                    'mAP': results['metrics'].get('mAP', 0),
                    'F1': results['metrics'].get('f1', 0),
                    'precision': results['metrics'].get('precision', 0),
                    'recall': results['metrics'].get('recall', 0),
                    'accuracy': results['metrics'].get('accuracy', 0),
                    'inference_time_ms': results['metrics'].get('inference_time', 0) * 1000
                })
            elif 'model_results' in results:
                # Batch evaluation
                for model_name, model_result in results['model_results'].items():
                    if 'error' in model_result:
                        continue
                    
                    if 'metrics' in model_result:
                        metrics = model_result['metrics']
                        metrics_data.append({
                            'model': model_name,
                            'mAP': metrics.get('mAP', 0),
                            'F1': metrics.get('f1', 0),
                            'precision': metrics.get('precision', 0),
                            'recall': metrics.get('recall', 0),
                            'accuracy': metrics.get('accuracy', 0),
                            'inference_time_ms': metrics.get('inference_time', 0) * 1000
                        })
            elif 'scenario_results' in results:
                # Research evaluation
                for scenario_name, scenario_result in results['scenario_results'].items():
                    if 'error' in scenario_result:
                        continue
                    
                    if 'results' in scenario_result and 'avg_metrics' in scenario_result['results']:
                        metrics = scenario_result['results']['avg_metrics']
                        metrics_data.append({
                            'scenario': scenario_name,
                            'description': scenario_result.get('config', {}).get('desc', ''),
                            'mAP': metrics.get('mAP', 0),
                            'F1': metrics.get('f1', 0),
                            'precision': metrics.get('precision', 0),
                            'recall': metrics.get('recall', 0),
                            'accuracy': metrics.get('accuracy', 0),
                            'inference_time_ms': metrics.get('inference_time', 0) * 1000
                        })
            
            # Buat DataFrame
            if not metrics_data:
                self.logger.warning("⚠️ Tidak ada data metrik untuk CSV")
                # Buat file kosong
                with open(output_path, 'w') as f:
                    f.write("# Tidak ada data metrik\n")
                return output_path
            
            df = pd.DataFrame(metrics_data)
            
            # Tulis ke CSV
            df.to_csv(output_path, index=False)
            
            self.logger.success(f"✅ Laporan CSV berhasil dibuat: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat laporan CSV: {str(e)}")
            raise
    
    def _generate_markdown(
        self,
        results: Dict[str, Any],
        output_path: str,
        include_plots: bool = True,
        **kwargs
    ) -> str:
        """
        Generate laporan format Markdown.
        
        Args:
            results: Hasil evaluasi
            output_path: Path output laporan
            include_plots: Sertakan visualisasi
            **kwargs: Parameter tambahan
            
        Returns:
            Path ke laporan Markdown
        """
        try:
            # Deteksi jenis hasil (model tunggal, batch, atau research)
            if 'metrics' in results:
                md_content = self._generate_single_model_markdown(results, include_plots)
            elif 'model_results' in results:
                md_content = self._generate_batch_markdown(results, include_plots)
            elif 'scenario_results' in results:
                md_content = self._generate_research_markdown(results, include_plots)
            else:
                md_content = "# Laporan Evaluasi\n\nTidak ada data yang valid untuk dilaporkan.\n"
            
            # Tulis ke file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            self.logger.success(f"✅ Laporan Markdown berhasil dibuat: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat laporan Markdown: {str(e)}")
            raise
    
    def _generate_html(
        self,
        results: Dict[str, Any],
        output_path: str,
        include_plots: bool = True,
        **kwargs
    ) -> str:
        """
        Generate laporan format HTML.
        
        Args:
            results: Hasil evaluasi
            output_path: Path output laporan
            include_plots: Sertakan visualisasi
            **kwargs: Parameter tambahan
            
        Returns:
            Path ke laporan HTML
        """
        try:
            # Generate Markdown dulu
            md_content = ""
            if 'metrics' in results:
                md_content = self._generate_single_model_markdown(results, include_plots)
            elif 'model_results' in results:
                md_content = self._generate_batch_markdown(results, include_plots)
            elif 'scenario_results' in results:
                md_content = self._generate_research_markdown(results, include_plots)
            else:
                md_content = "# Laporan Evaluasi\n\nTidak ada data yang valid untuk dilaporkan.\n"
            
            # Konversi Markdown ke HTML menggunakan library markdown
            try:
                import markdown
                html_body = markdown.markdown(md_content, extensions=['tables'])
            except ImportError:
                # Fallback jika markdown tidak tersedia
                html_body = f"<pre>{md_content}</pre>"
            
            # Buat HTML lengkap dengan styling
            html_content = f"""
            <!DOCTYPE html>
            <html lang="id">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Laporan Evaluasi SmartCash</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 10px 0; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .metrics {{ font-weight: bold; color: #2980b9; }}
                    .timestamp {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px; }}
                    .error {{ color: #e74c3c; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="timestamp">Dihasilkan pada: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                    {html_body}
                </div>
            </body>
            </html>
            """
            
            # Tulis ke file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.success(f"✅ Laporan HTML berhasil dibuat: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat laporan HTML: {str(e)}")
            raise
    
    def _generate_single_model_markdown(
        self,
        results: Dict[str, Any],
        include_plots: bool = True
    ) -> str:
        """
        Generate markdown untuk evaluasi model tunggal.
        
        Args:
            results: Hasil evaluasi model tunggal
            include_plots: Sertakan visualisasi
            
        Returns:
            String konten markdown
        """
        model_path = results.get('model_path', 'unknown')
        model_name = os.path.basename(model_path)
        dataset_path = results.get('dataset_path', 'unknown')
        dataset_name = os.path.basename(dataset_path)
        metrics = results.get('metrics', {})
        
        # Format metrics
        map_value = metrics.get('mAP', 0)
        f1_value = metrics.get('f1', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        accuracy = metrics.get('accuracy', 0)
        inference_time = metrics.get('inference_time', 0) * 1000  # Convert to ms
        
        # Buat markdown
        md = f"""# Laporan Evaluasi Model SmartCash

## Informasi Model

- **Nama Model**: {model_name}
- **Path**: {model_path}
- **Dataset**: {dataset_name}
- **Tanggal Evaluasi**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Metrik Performa

| Metrik | Nilai |
|--------|-------|
| mAP | {map_value:.4f} |
| F1-Score | {f1_value:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| Accuracy | {accuracy:.4f} |
| Waktu Inferensi | {inference_time:.2f} ms |

"""
        
        # Tambahkan class metrics jika ada
        class_metrics = metrics.get('class_metrics', {})
        if class_metrics:
            md += "## Metrik per Kelas\n\n"
            md += "| Kelas | Precision | Recall | F1-Score | mAP |\n"
            md += "|-------|-----------|--------|----------|-----|\n"
            
            for class_id, class_metric in class_metrics.items():
                md += f"| {class_id} | {class_metric.get('precision', 0):.4f} | {class_metric.get('recall', 0):.4f} | {class_metric.get('f1', 0):.4f} | {class_metric.get('ap', 0):.4f} |\n"
            
            md += "\n"
        
        # Tambahkan confusion matrix jika ada
        if 'confusion_matrix_path' in results and include_plots:
            md += f"## Confusion Matrix\n\n"
            md += f"![Confusion Matrix]({results['confusion_matrix_path']})\n\n"
        
        # Tambahkan roc curve jika ada
        if 'roc_curve_path' in results and include_plots:
            md += f"## ROC Curve\n\n"
            md += f"![ROC Curve]({results['roc_curve_path']})\n\n"
        
        # Tambahkan precision-recall curve jika ada
        if 'pr_curve_path' in results and include_plots:
            md += f"## Precision-Recall Curve\n\n"
            md += f"![Precision-Recall Curve]({results['pr_curve_path']})\n\n"
        
        # Tambahkan kesimpulan
        md += "## Kesimpulan\n\n"
        
        # Buat kesimpulan berdasarkan mAP
        if map_value >= 0.9:
            md += "Model menunjukkan performa yang sangat baik dengan mAP yang tinggi. Cocok untuk digunakan dalam aplikasi produksi.\n"
        elif map_value >= 0.7:
            md += "Model menunjukkan performa yang baik, tetapi masih ada ruang untuk peningkatan terutama dalam beberapa kelas.\n"
        elif map_value >= 0.5:
            md += "Model menunjukkan performa yang cukup, tetapi perlu perbaikan signifikan untuk penggunaan produksi.\n"
        else:
            md += "Model menunjukkan performa yang kurang baik dan perlu perbaikan major sebelum digunakan.\n"
        
        # Tambahkan rekomendasi berdasarkan inference time
        if inference_time <= 20:  # kurang dari 20ms
            md += "Waktu inferensi sangat cepat, cocok untuk aplikasi real-time.\n"
        elif inference_time <= 50:  # kurang dari 50ms
            md += "Waktu inferensi cukup cepat untuk kebanyakan aplikasi praktis.\n"
        else:
            md += "Waktu inferensi relatif lambat, perlu dipertimbangkan untuk optimasi jika digunakan dalam aplikasi real-time.\n"
        
        return md
    
    def _generate_batch_markdown(
        self,
        results: Dict[str, Any],
        include_plots: bool = True
    ) -> str:
        """
        Generate markdown untuk evaluasi batch model.
        
        Args:
            results: Hasil evaluasi batch model
            include_plots: Sertakan visualisasi
            
        Returns:
            String konten markdown
        """
        model_results = results.get('model_results', {})
        dataset_path = results.get('dataset_path', 'unknown')
        dataset_name = os.path.basename(dataset_path)
        summary = results.get('summary', {})
        
        # Buat markdown
        md = f"""# Laporan Evaluasi Batch Model SmartCash

## Informasi Umum

- **Jumlah Model**: {summary.get('num_models', len(model_results))}
- **Model Sukses**: {summary.get('successful_models', 0)}
- **Model Gagal**: {summary.get('failed_models', 0)}
- **Dataset**: {dataset_name}
- **Tanggal Evaluasi**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

"""
        
        # Tambahkan best model jika ada
        if 'best_model' in summary and summary['best_model'] is not None:
            md += f"""## Model Terbaik

- **Nama Model**: {summary['best_model']}
- **mAP**: {summary.get('best_map', 0):.4f}
- **F1-Score**: {summary.get('best_f1', 0):.4f}

"""
        
        # Tambahkan tabel perbandingan
        if 'metrics_table' in summary and summary['metrics_table'] is not None:
            metrics_df = summary['metrics_table']
            
            md += "## Perbandingan Model\n\n"
            md += "| Model | mAP | F1-Score | Precision | Recall | Waktu Inferensi (ms) |\n"
            md += "|-------|-----|----------|-----------|--------|----------------------|\n"
            
            for _, row in metrics_df.iterrows():
                md += f"| {row['model']} | {row['mAP']:.4f} | {row['F1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['inference_time']:.2f} |\n"
            
            md += "\n"
        
        # Tambahkan performance comparison jika ada
        if 'performance_comparison' in summary and summary['performance_comparison']:
            md += "## Analisis Performa\n\n"
            
            for metric, metric_data in summary['performance_comparison'].items():
                if metric == 'mAP':
                    metric_name = "mAP"
                elif metric == 'F1':
                    metric_name = "F1-Score"
                elif metric == 'precision':
                    metric_name = "Precision"
                elif metric == 'recall':
                    metric_name = "Recall"
                elif metric == 'inference_time':
                    metric_name = "Waktu Inferensi (ms)"
                else:
                    metric_name = metric
                
                best_model = metric_data.get('best_model', 'Unknown')
                best_value = metric_data.get('best_value', 0)
                average = metric_data.get('average', 0)
                
                md += f"### {metric_name}\n\n"
                md += f"- **Model Terbaik**: {best_model}\n"
                md += f"- **Nilai Terbaik**: {best_value:.4f}\n"
                md += f"- **Rata-rata**: {average:.4f}\n\n"
        
        # Tambahkan kesimpulan
        md += "## Kesimpulan\n\n"
        
        # Buat kesimpulan berdasarkan perbandingan model
        avg_map = summary.get('average_map', 0)
        
        if avg_map >= 0.8:
            md += "Secara umum, semua model menunjukkan performa yang baik dengan mAP rata-rata yang tinggi.\n"
        elif avg_map >= 0.6:
            md += "Performa rata-rata model cukup baik, dengan beberapa model yang menunjukkan hasil sangat baik.\n"
        else:
            md += "Performa rata-rata model masih kurang optimal dan perlu peningkatan lebih lanjut.\n"
        
        # Rekomendasi model terbaik
        if 'best_model' in summary and summary['best_model'] is not None:
            md += f"\nModel **{summary['best_model']}** adalah model terbaik berdasarkan mAP dan direkomendasikan untuk digunakan dalam aplikasi produksi.\n"
        
        return md
    
    def _generate_research_markdown(
        self,
        results: Dict[str, Any],
        include_plots: bool = True
    ) -> str:
        """
        Generate markdown untuk evaluasi skenario penelitian.
        
        Args:
            results: Hasil evaluasi skenario penelitian
            include_plots: Sertakan visualisasi
            
        Returns:
            String konten markdown
        """
        scenario_results = results.get('scenario_results', {})
        summary = results.get('summary', {})
        plots = results.get('plots', {})
        
        # Buat markdown
        md = f"""# Laporan Evaluasi Skenario Penelitian SmartCash

## Informasi Umum

- **Jumlah Skenario**: {summary.get('num_scenarios', len(scenario_results))}
- **Skenario Sukses**: {summary.get('successful_scenarios', 0)}
- **Skenario Gagal**: {summary.get('failed_scenarios', 0)}
- **Tanggal Evaluasi**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

"""
        
        # Tambahkan best scenario jika ada
        if 'best_scenario' in summary and summary['best_scenario'] is not None:
            best_scenario = summary['best_scenario']
            best_scenario_config = scenario_results.get(best_scenario, {}).get('config', {})
            
            md += f"""## Skenario Terbaik

- **Nama Skenario**: {best_scenario}
- **Deskripsi**: {best_scenario_config.get('desc', 'Tidak ada deskripsi')}
- **mAP**: {summary.get('best_map', 0):.4f}

"""
        
        # Tambahkan tabel perbandingan
        if 'metrics_table' in summary and summary['metrics_table'] is not None:
            metrics_df = summary['metrics_table']
            
            md += "## Perbandingan Skenario\n\n"
            md += "| Skenario | Deskripsi | mAP | F1-Score | Precision | Recall | Waktu Inferensi (ms) |\n"
            md += "|----------|-----------|-----|----------|-----------|--------|----------------------|\n"
            
            for _, row in metrics_df.iterrows():
                md += f"| {row['scenario']} | {row['description']} | {row['mAP']:.4f} | {row['F1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['inference_time']:.2f} |\n"
            
            md += "\n"
        
        # Tambahkan backbone comparison jika ada
        if 'backbone_comparison' in summary and summary['backbone_comparison']:
            md += "## Perbandingan Backbone\n\n"
            md += "| Backbone | mAP | F1-Score | Waktu Inferensi (ms) |\n"
            md += "|----------|-----|----------|----------------------|\n"
            
            for backbone, metrics in summary['backbone_comparison'].items():
                md += f"| {backbone} | {metrics.get('mAP', 0):.4f} | {metrics.get('F1', 0):.4f} | {metrics.get('inference_time', 0):.2f} |\n"
            
            md += "\n"
        
        # Tambahkan condition comparison jika ada
        if 'condition_comparison' in summary and summary['condition_comparison']:
            md += "## Perbandingan Kondisi Pengujian\n\n"
            md += "| Kondisi | mAP | F1-Score |\n"
            md += "|---------|-----|----------|\n"
            
            for condition, metrics in summary['condition_comparison'].items():
                md += f"| {condition} | {metrics.get('mAP', 0):.4f} | {metrics.get('F1', 0):.4f} |\n"
            
            md += "\n"
        
        # Tambahkan visualisasi jika ada dan diminta
        if include_plots and plots:
            md += "## Visualisasi\n\n"
            
            # Urutkan plot berdasarkan kepentingan
            plot_order = [
                'backbone_comparison', 
                'condition_comparison', 
                'map_f1_comparison', 
                'inference_time', 
                'combined_heatmap'
            ]
            
            for plot_key in plot_order:
                if plot_key in plots:
                    plot_path = plots[plot_key]
                    plot_title = self._get_plot_title(plot_key)
                    
                    md += f"### {plot_title}\n\n"
                    md += f"![{plot_title}]({plot_path})\n\n"
        
        # Tambahkan kesimpulan
        md += "## Kesimpulan dan Rekomendasi\n\n"
        
        # Backbone comparison
        if 'backbone_comparison' in summary and summary['backbone_comparison']:
            # Bandingkan EfficientNet vs CSPDarknet
            if 'EfficientNet' in summary['backbone_comparison'] and 'CSPDarknet' in summary['backbone_comparison']:
                eff_map = summary['backbone_comparison']['EfficientNet'].get('mAP', 0)
                csp_map = summary['backbone_comparison']['CSPDarknet'].get('mAP', 0)
                
                eff_inference = summary['backbone_comparison']['EfficientNet'].get('inference_time', 0)
                csp_inference = summary['backbone_comparison']['CSPDarknet'].get('inference_time', 0)
                
                md += "### Perbandingan Backbone\n\n"
                
                if eff_map > csp_map:
                    map_diff = (eff_map - csp_map) / csp_map * 100
                    md += f"- EfficientNet menunjukkan performa mAP **{map_diff:.1f}%** lebih baik dibandingkan CSPDarknet.\n"
                elif csp_map > eff_map:
                    map_diff = (csp_map - eff_map) / eff_map * 100
                    md += f"- CSPDarknet menunjukkan performa mAP **{map_diff:.1f}%** lebih baik dibandingkan EfficientNet.\n"
                else:
                    md += "- EfficientNet dan CSPDarknet menunjukkan performa mAP yang setara.\n"
                
                if eff_inference < csp_inference:
                    time_diff = (csp_inference - eff_inference) / csp_inference * 100
                    md += f"- EfficientNet **{time_diff:.1f}%** lebih cepat dalam inferensi dibandingkan CSPDarknet.\n"
                elif csp_inference < eff_inference:
                    time_diff = (eff_inference - csp_inference) / eff_inference * 100
                    md += f"- CSPDarknet **{time_diff:.1f}%** lebih cepat dalam inferensi dibandingkan EfficientNet.\n"
                else:
                    md += "- EfficientNet dan CSPDarknet menunjukkan waktu inferensi yang setara.\n"
                
                md += "\n"
        
        # Condition comparison
        if 'condition_comparison' in summary and summary['condition_comparison']:
            # Bandingkan Posisi vs Pencahayaan
            if 'Posisi Bervariasi' in summary['condition_comparison'] and 'Pencahayaan Bervariasi' in summary['condition_comparison']:
                posisi_map = summary['condition_comparison']['Posisi Bervariasi'].get('mAP', 0)
                cahaya_map = summary['condition_comparison']['Pencahayaan Bervariasi'].get('mAP', 0)
                
                md += "### Perbandingan Kondisi Pengujian\n\n"
                
                if posisi_map > cahaya_map:
                    map_diff = (posisi_map - cahaya_map) / cahaya_map * 100
                    md += f"- Model menunjukkan performa mAP **{map_diff:.1f}%** lebih baik pada kondisi posisi bervariasi dibandingkan dengan pencahayaan bervariasi.\n"
                elif cahaya_map > posisi_map:
                    map_diff = (cahaya_map - posisi_map) / posisi_map * 100
                    md += f"- Model menunjukkan performa mAP **{map_diff:.1f}%** lebih baik pada kondisi pencahayaan bervariasi dibandingkan dengan posisi bervariasi.\n"
                else:
                    md += "- Model menunjukkan performa yang setara pada kondisi posisi bervariasi dan pencahayaan bervariasi.\n"
                
                md += "\n"
        
        # Rekomendasi akhir
        md += "### Rekomendasi\n\n"
        
        # Tentukan backbone terbaik
        best_backbone = None
        best_backbone_map = 0
        
        if 'backbone_comparison' in summary and summary['backbone_comparison']:
            for backbone, metrics in summary['backbone_comparison'].items():
                if metrics.get('mAP', 0) > best_backbone_map:
                    best_backbone_map = metrics.get('mAP', 0)
                    best_backbone = backbone
        
        # Rekomendasi berdasarkan best scenario
        if 'best_scenario' in summary and summary['best_scenario'] is not None:
            best_scenario = summary['best_scenario']
            best_scenario_config = scenario_results.get(best_scenario, {}).get('config', {})
            
            md += f"Berdasarkan hasil evaluasi, kami merekomendasikan:\n\n"
            
            # Rekomendasi backbone
            if best_backbone:
                md += f"1. Menggunakan backbone **{best_backbone}** yang menunjukkan performa terbaik dengan mAP {best_backbone_map:.4f}.\n"
            
            # Rekomendasi skenario
            md += f"2. Mengadopsi konfigurasi dari skenario **{best_scenario}** ({best_scenario_config.get('desc', '')}) yang menghasilkan mAP tertinggi.\n"
            
            # Rekomendasi kondisi optimum
            if 'condition_comparison' in summary and summary['condition_comparison']:
                best_condition = max(summary['condition_comparison'].items(), key=lambda x: x[1].get('mAP', 0))[0]
                md += f"3. Memperhatikan bahwa model bekerja paling optimal pada kondisi **{best_condition}**.\n"
        
        return md
    
    def _prepare_report_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for report, clean non-serializable objects.
        
        Args:
            results: Result dictionary
            
        Returns:
            Clean copy for report
        """
        # Helper function to check if object is JSON serializable
        def is_json_serializable(obj):
            try:
                json.dumps(obj)
                return True
            except (TypeError, OverflowError):
                return False
        
        # Helper function to clean dictionary
        def clean_dict(d):
            if not isinstance(d, dict):
                return d
                
            clean = {}
            for k, v in d.items():
                # Skip pandas DataFrames
                if str(type(v)).startswith("<class 'pandas."):
                    continue
                
                # Recursively clean nested dicts
                if isinstance(v, dict):
                    clean[k] = clean_dict(v)
                # Clean lists
                elif isinstance(v, list):
                    clean[k] = [clean_dict(item) if isinstance(item, dict) else item for item in v if is_json_serializable(item)]
                # Include only serializable values
                elif is_json_serializable(v):
                    clean[k] = v
            return clean
        
        # Clean the results
        return clean_dict(results)
    
    def _get_plot_title(self, plot_key: str) -> str:
        """
        Get human-readable title for plot.
        
        Args:
            plot_key: Plot key
            
        Returns:
            Human-readable title
        """
        plot_titles = {
            'backbone_comparison': 'Perbandingan Backbone',
            'condition_comparison': 'Perbandingan Kondisi Pengujian',
            'map_f1_comparison': 'Perbandingan mAP dan F1 Score',
            'inference_time': 'Waktu Inferensi',
            'combined_heatmap': 'Heatmap Kombinasi Backbone dan Kondisi',
            'confusion_matrix': 'Confusion Matrix',
            'roc_curve': 'ROC Curve',
            'pr_curve': 'Precision-Recall Curve'
        }
        
        return plot_titles.get(plot_key, plot_key.replace('_', ' ').title())