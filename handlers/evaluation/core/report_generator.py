# File: smartcash/handlers/evaluation/core/report_generator.py
# Author: Alfrida Sabar
# Deskripsi: Generator laporan yang disederhanakan untuk hasil evaluasi model

import os
import json
import pandas as pd
import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.utils.observer.event_dispatcher import EventDispatcher

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
        # Notifikasi start
        EventDispatcher.notify("evaluation.report.start", self, {
            'format': format,
            'output_path': output_path
        })
        
        self.logger.info(f"📊 Membuat laporan evaluasi format {format}")
        
        # Tentukan output path jika belum ada
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.output_dir / f"evaluation_report_{timestamp}.{format}")
        
        # Buat path output jika belum ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Generate laporan sesuai format
            if format.lower() == 'json':
                result_path = self._generate_json(results, output_path, **kwargs)
            elif format.lower() == 'csv':
                result_path = self._generate_csv(results, output_path, **kwargs)
            elif format.lower() in ['markdown', 'md']:
                result_path = self._generate_markdown(results, output_path, include_plots, **kwargs)
            elif format.lower() == 'html':
                result_path = self._generate_html(results, output_path, include_plots, **kwargs)
            else:
                self.logger.warning(f"⚠️ Format laporan tidak didukung: {format}, menggunakan JSON")
                result_path = self._generate_json(results, output_path, **kwargs)
            
            # Notifikasi complete
            EventDispatcher.notify("evaluation.report.complete", self, {
                'format': format,
                'output_path': result_path
            })
            
            return result_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat laporan: {str(e)}")
            
            # Notifikasi error
            EventDispatcher.notify("evaluation.report.error", self, {
                'error': str(e),
                'format': format
            })
            
            raise
    
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
        # Persiapkan data (buang objek non-serializable)
        report_data = self._prepare_report_data(results)
        
        # Tulis ke file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.success(f"✅ Laporan JSON berhasil dibuat: {output_path}")
        return output_path
    
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
        # Ekstrak metrik
        metrics_data = self._extract_metrics_data(results)
        
        if not metrics_data:
            self.logger.warning("⚠️ Tidak ada data metrik untuk CSV")
            # Buat file kosong
            with open(output_path, 'w') as f:
                f.write("# Tidak ada data metrik\n")
            return output_path
        
        # Buat DataFrame
        df = pd.DataFrame(metrics_data)
        
        # Tulis ke CSV
        df.to_csv(output_path, index=False)
        
        self.logger.success(f"✅ Laporan CSV berhasil dibuat: {output_path}")
        return output_path
    
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
        # Deteksi tipe hasil
        md_content = "# Laporan Evaluasi Model SmartCash\n\n"
        
        if 'metrics' in results:
            md_content += self._generate_single_model_md(results, include_plots)
        elif 'model_results' in results:
            md_content += self._generate_batch_md(results, include_plots)
        elif 'scenario_results' in results:
            md_content += self._generate_scenarios_md(results, include_plots)
        else:
            md_content += "Tidak ada data yang valid untuk dilaporkan.\n"
        
        # Tambahkan timestamp
        md_content += f"\n\n---\nLaporan dibuat pada: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # Tulis ke file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        self.logger.success(f"✅ Laporan Markdown berhasil dibuat: {output_path}")
        return output_path
    
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
        # Generate markdown dulu
        md_content = ""
        if 'metrics' in results:
            md_content = self._generate_single_model_md(results, include_plots)
        elif 'model_results' in results:
            md_content = self._generate_batch_md(results, include_plots)
        elif 'scenario_results' in results:
            md_content = self._generate_scenarios_md(results, include_plots)
        else:
            md_content = "Tidak ada data yang valid untuk dilaporkan.\n"
        
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
                <div class="timestamp">Dibuat pada: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
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
    
    def _extract_metrics_data(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Ekstrak data metrik dari hasil evaluasi.
        
        Args:
            results: Hasil evaluasi
            
        Returns:
            List data metrik untuk CSV
        """
        metrics_data = []
        
        # Deteksi jenis hasil
        if 'metrics' in results:
            # Model tunggal
            metrics_data.append({
                'model': os.path.basename(results.get('model_path', 'unknown')),
                'mAP': results['metrics'].get('mAP', 0),
                'F1': results['metrics'].get('f1', 0),
                'precision': results['metrics'].get('precision', 0),
                'recall': results['metrics'].get('recall', 0),
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
                        'inference_time_ms': metrics.get('inference_time', 0) * 1000
                    })
                    
        return metrics_data
    
    def _generate_single_model_md(self, results: Dict[str, Any], include_plots: bool) -> str:
        """
        Generate markdown untuk model tunggal.
        
        Args:
            results: Hasil evaluasi
            include_plots: Sertakan visualisasi
            
        Returns:
            Markdown content
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
        inference_time = metrics.get('inference_time', 0) * 1000  # Convert to ms
        
        # Markdown untuk model tunggal
        md = f"""## Informasi Model

- **Nama Model**: {model_name}
- **Path**: {model_path}
- **Dataset**: {dataset_name}

## Metrik Performa

| Metrik | Nilai |
|--------|-------|
| mAP | {map_value:.4f} |
| F1-Score | {f1_value:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
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
        
        # Tambahkan visualisasi jika ada dan diinginkan
        if include_plots:
            if 'confusion_matrix_path' in results:
                md += f"## Confusion Matrix\n\n"
                md += f"![Confusion Matrix]({results['confusion_matrix_path']})\n\n"
                
            if 'pr_curve_path' in results:
                md += f"## Precision-Recall Curve\n\n"
                md += f"![Precision-Recall Curve]({results['pr_curve_path']})\n\n"
        
        return md
    
    def _generate_batch_md(self, results: Dict[str, Any], include_plots: bool) -> str:
        """
        Generate markdown untuk batch evaluation.
        
        Args:
            results: Hasil evaluasi
            include_plots: Sertakan visualisasi
            
        Returns:
            Markdown content
        """
        model_results = results.get('model_results', {})
        dataset_path = results.get('dataset_path', 'unknown')
        dataset_name = os.path.basename(dataset_path)
        summary = results.get('summary', {})
        
        # Markdown untuk batch evaluation
        md = f"""## Informasi Batch

- **Jumlah Model**: {summary.get('num_models', len(model_results))}
- **Model Sukses**: {summary.get('successful_models', 0)}
- **Model Gagal**: {summary.get('failed_models', 0)}
- **Dataset**: {dataset_name}

"""
        # Tambahkan best model jika ada
        if 'best_model' in summary and summary['best_model'] is not None:
            md += f"""## Model Terbaik

- **Nama Model**: {summary['best_model']}
- **mAP**: {summary.get('best_map', 0):.4f}
- **F1-Score**: {summary.get('best_f1', 0):.4f}

"""
        # Tambahkan tabel perbandingan
        md += "## Perbandingan Model\n\n"
        md += "| Model | mAP | F1-Score | Precision | Recall | Waktu Inferensi (ms) |\n"
        md += "|-------|-----|----------|-----------|--------|----------------------|\n"
        
        for model_name, model_result in model_results.items():
            if 'error' in model_result:
                md += f"| {model_name} | - | - | - | - | - |\n"
                continue
                
            if 'metrics' in model_result:
                metrics = model_result['metrics']
                md += f"| {model_name} | {metrics.get('mAP', 0):.4f} | {metrics.get('f1', 0):.4f} | {metrics.get('precision', 0):.4f} | {metrics.get('recall', 0):.4f} | {metrics.get('inference_time', 0)*1000:.2f} |\n"
        
        # Tambahkan visualisasi jika ada dan diinginkan
        if include_plots and 'plots' in results:
            md += "## Visualisasi\n\n"
            
            for plot_name, plot_path in results['plots'].items():
                plot_title = plot_name.replace('_', ' ').title()
                md += f"### {plot_title}\n\n"
                md += f"![{plot_title}]({plot_path})\n\n"
        
        return md
    
    def _generate_scenarios_md(self, results: Dict[str, Any], include_plots: bool) -> str:
        """
        Generate markdown untuk skenario penelitian.
        
        Args:
            results: Hasil evaluasi
            include_plots: Sertakan visualisasi
            
        Returns:
            Markdown content
        """
        scenario_results = results.get('scenario_results', {})
        summary = results.get('summary', {})
        
        # Markdown untuk skenario penelitian
        md = f"""## Informasi Skenario Penelitian

- **Jumlah Skenario**: {summary.get('num_scenarios', len(scenario_results))}
- **Skenario Sukses**: {summary.get('successful_scenarios', 0)}
- **Skenario Gagal**: {summary.get('failed_scenarios', 0)}

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
        md += "## Perbandingan Skenario\n\n"
        md += "| Skenario | Deskripsi | mAP | F1-Score | Precision | Recall | Waktu Inferensi (ms) |\n"
        md += "|----------|-----------|-----|----------|-----------|--------|----------------------|\n"
        
        for scenario_name, scenario_result in scenario_results.items():
            if 'error' in scenario_result:
                md += f"| {scenario_name} | {scenario_result.get('config', {}).get('desc', '-')} | - | - | - | - | - |\n"
                continue
                
            if 'results' in scenario_result and 'avg_metrics' in scenario_result['results']:
                metrics = scenario_result['results']['avg_metrics']
                md += f"| {scenario_name} | {scenario_result.get('config', {}).get('desc', '-')} | {metrics.get('mAP', 0):.4f} | {metrics.get('f1', 0):.4f} | {metrics.get('precision', 0):.4f} | {metrics.get('recall', 0):.4f} | {metrics.get('inference_time', 0)*1000:.2f} |\n"
        
        # Tambahkan backbone comparison jika ada
        if 'backbone_comparison' in summary and summary['backbone_comparison']:
            md += "\n## Perbandingan Backbone\n\n"
            md += "| Backbone | mAP | F1-Score | Waktu Inferensi (ms) |\n"
            md += "|----------|-----|----------|----------------------|\n"
            
            for backbone, metrics in summary['backbone_comparison'].items():
                md += f"| {backbone} | {metrics.get('mAP', 0):.4f} | {metrics.get('F1', 0):.4f} | {metrics.get('inference_time', 0):.2f} |\n"
            
            md += "\n"
            
        # Tambahkan visualisasi jika ada dan diinginkan
        if include_plots and 'plots' in results:
            md += "## Visualisasi\n\n"
            
            for plot_name, plot_path in results['plots'].items():
                plot_title = plot_name.replace('_', ' ').title()
                md += f"### {plot_title}\n\n"
                md += f"![{plot_title}]({plot_path})\n\n"
        
        return md
    
    def _prepare_report_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Persiapkan data laporan, bersihkan objek non-serializable.
        
        Args:
            results: Dictionary hasil
            
        Returns:
            Clean dictionary untuk JSON
        """
        # Helper function untuk cek serializable
        def is_json_serializable(obj):
            try:
                json.dumps(obj)
                return True
            except (TypeError, OverflowError):
                return False
        
        # Helper function untuk bersihkan dict
        def clean_dict(d):
            if not isinstance(d, dict):
                return d
                
            clean = {}
            for k, v in d.items():
                # Skip pandas DataFrames dan objek non-serializable lainnya
                if str(type(v)).startswith("<class 'pandas.") or not is_json_serializable(v):
                    continue
                
                # Rekursif untuk nested dicts
                if isinstance(v, dict):
                    clean[k] = clean_dict(v)
                # Clean lists
                elif isinstance(v, list):
                    clean[k] = [clean_dict(item) if isinstance(item, dict) else item for item in v if is_json_serializable(item)]
                # Include serializable values
                else:
                    clean[k] = v
            return clean
        
        # Bersihkan hasil
        return clean_dict(results)