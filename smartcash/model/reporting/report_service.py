"""
File: smartcash/model/reporting/report_service.py
Deskripsi: Service untuk generating comprehensive reports dari analysis results
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import csv
from datetime import datetime
from smartcash.common.logger import get_logger
from smartcash.model.reporting.generators.summary_generator import SummaryGenerator
from smartcash.model.reporting.generators.comparison_generator import ComparisonGenerator
from smartcash.model.reporting.generators.research_generator import ResearchGenerator

class ReportService:
    """Service untuk generating comprehensive analysis reports"""
    
    def __init__(self, config: Dict[str, Any] = None, output_dir: str = 'data/analysis/reports', logger=None):
        self.config = config or {}
        self.logger = logger or get_logger('report_service')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Report configuration
        report_config = self.config.get('reporting', {})
        self.formats = report_config.get('formats', {'markdown': True, 'json': True})
        self.sections = report_config.get('sections', {})
        self.include_visualizations = report_config.get('export', {}).get('include_visualizations', True)
        self.timestamp_files = report_config.get('export', {}).get('timestamp_files', True)
        
        # Initialize generators
        self.summary_generator = SummaryGenerator(self.config, self.logger)
        self.comparison_generator = ComparisonGenerator(self.config, self.logger)
        self.research_generator = ResearchGenerator(self.config, self.logger)
    
    def generate_comprehensive_report(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate comprehensive report dalam berbagai format"""
        try:
            self.logger.info("📋 Starting comprehensive report generation...")
            
            report_paths = {}
            timestamp = self._get_timestamp() if self.timestamp_files else ""
            
            # Generate markdown report
            if self.formats.get('markdown', True):
                md_path = self._generate_markdown_report(analysis_results, timestamp)
                if md_path:
                    report_paths['markdown'] = md_path
            
            # Generate JSON report
            if self.formats.get('json', True):
                json_path = self._generate_json_report(analysis_results, timestamp)
                if json_path:
                    report_paths['json'] = json_path
            
            # Generate CSV summary
            if self.formats.get('csv', False):
                csv_path = self._generate_csv_summary(analysis_results, timestamp)
                if csv_path:
                    report_paths['csv'] = csv_path
            
            # Generate HTML report
            if self.formats.get('html', False):
                html_path = self._generate_html_report(analysis_results, timestamp)
                if html_path:
                    report_paths['html'] = html_path
            
            self.logger.info(f"✅ Generated {len(report_paths)} report formats")
            return report_paths
            
        except Exception as e:
            self.logger.error(f"❌ Error generating reports: {str(e)}")
            return {}
    
    def generate_quick_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate quick text summary untuk immediate review"""
        try:
            return self.summary_generator.generate_quick_summary(analysis_results)
        except Exception as e:
            self.logger.error(f"❌ Error generating quick summary: {str(e)}")
            return "⚠️ Error generating quick summary."
    
    def _generate_markdown_report(self, analysis_results: Dict[str, Any], timestamp: str = "") -> Optional[str]:
        """Generate comprehensive markdown report"""
        try:
            # Prepare filename
            filename = f"smartcash_analysis_report_{timestamp}.md" if timestamp else "smartcash_analysis_report.md"
            output_path = self.output_dir / filename
            
            # Generate report content
            content = []
            
            # Header
            content.append(self._generate_md_header(analysis_results))
            
            # Executive Summary
            if self.sections.get('executive_summary', True):
                content.append(self.summary_generator.generate_executive_summary(analysis_results))
            
            # Methodology
            if self.sections.get('methodology', True):
                content.append(self.research_generator.generate_methodology_report(analysis_results))
            
            # Results Overview
            if self.sections.get('results_overview', True):
                content.append(self.summary_generator.generate_results_overview(analysis_results))
            
            # Currency Analysis
            if self.sections.get('currency_analysis', True):
                content.append(self.research_generator.generate_currency_analysis_report(analysis_results))
            
            # Performance Analysis
            if self.sections.get('performance_analysis', True):
                content.append(self.research_generator.generate_performance_analysis_report(analysis_results))
            
            # Backbone Comparison
            if self.sections.get('backbone_comparison', True):
                content.append(self.comparison_generator.generate_backbone_comparison(analysis_results))
            
            # Scenario Comparison
            if self.sections.get('scenario_comparison', True):
                content.append(self.comparison_generator.generate_scenario_comparison(analysis_results))
            
            # Recommendations
            if self.sections.get('recommendations', True):
                content.append(self._generate_md_recommendations(analysis_results))
            
            # Technical Details
            if self.sections.get('technical_details', True):
                content.append(self._generate_md_technical_details(analysis_results))
            
            # Write file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(filter(None, content)))
            
            self.logger.info(f"📄 Markdown report saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating markdown report: {str(e)}")
            return None
    
    def _generate_json_report(self, analysis_results: Dict[str, Any], timestamp: str = "") -> Optional[str]:
        """Generate structured JSON report"""
        try:
            filename = f"smartcash_analysis_data_{timestamp}.json" if timestamp else "smartcash_analysis_data.json"
            output_path = self.output_dir / filename
            
            # Structure JSON report
            json_report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_version': '1.0',
                    'generator': 'SmartCash ReportService'
                },
                'analysis_summary': analysis_results.get('analysis_summary', {}),
                'summary_metrics': analysis_results.get('summary_metrics', {}),
                'currency_analysis': analysis_results.get('currency_analysis', {}),
                'layer_analysis': analysis_results.get('layer_analysis', {}),
                'class_analysis': analysis_results.get('class_analysis', {}),
                'comparison_analysis': analysis_results.get('comparison_analysis', {}),
                'key_findings': analysis_results.get('key_findings', []),
                'recommendations': analysis_results.get('recommendations', []),
                'model_info': analysis_results.get('model_info', {}),
                'dataset_info': analysis_results.get('dataset_info', {})
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📊 JSON report saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating JSON report: {str(e)}")
            return None
    
    def _generate_csv_summary(self, analysis_results: Dict[str, Any], timestamp: str = "") -> Optional[str]:
        """Generate CSV summary dengan key metrics"""
        try:
            filename = f"smartcash_metrics_summary_{timestamp}.csv" if timestamp else "smartcash_metrics_summary.csv"
            output_path = self.output_dir / filename
            
            # Extract key metrics untuk CSV
            metrics_data = []
            
            # Summary metrics
            summary_metrics = analysis_results.get('summary_metrics', {})
            if summary_metrics:
                metrics_data.append({
                    'Category': 'Overall',
                    'Metric': 'mAP',
                    'Value': summary_metrics.get('overall_map', 0.0),
                    'Unit': 'percentage'
                })
                metrics_data.append({
                    'Category': 'Overall',
                    'Metric': 'Accuracy',
                    'Value': summary_metrics.get('overall_accuracy', 0.0),
                    'Unit': 'percentage'
                })
                metrics_data.append({
                    'Category': 'Overall',
                    'Metric': 'Inference Time',
                    'Value': summary_metrics.get('avg_inference_time', 0.0),
                    'Unit': 'milliseconds'
                })
            
            # Per-class metrics
            class_analysis = analysis_results.get('class_analysis', {})
            per_class_metrics = class_analysis.get('per_class_metrics', {})
            
            denominations = {
                '0': 'Rp 1.000', '1': 'Rp 2.000', '2': 'Rp 5.000', '3': 'Rp 10.000',
                '4': 'Rp 20.000', '5': 'Rp 50.000', '6': 'Rp 100.000'
            }
            
            for class_id, metrics in per_class_metrics.items():
                denom_name = denominations.get(class_id, f'Class {class_id}')
                metrics_data.extend([
                    {'Category': denom_name, 'Metric': 'Precision', 'Value': metrics.get('precision', 0.0), 'Unit': 'percentage'},
                    {'Category': denom_name, 'Metric': 'Recall', 'Value': metrics.get('recall', 0.0), 'Unit': 'percentage'},
                    {'Category': denom_name, 'Metric': 'F1-Score', 'Value': metrics.get('f1_score', 0.0), 'Unit': 'percentage'}
                ])
            
            # Write CSV
            if metrics_data:
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['Category', 'Metric', 'Value', 'Unit']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(metrics_data)
                
                self.logger.info(f"📈 CSV summary saved: {output_path}")
                return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating CSV summary: {str(e)}")
            return None
    
    def _generate_html_report(self, analysis_results: Dict[str, Any], timestamp: str = "") -> Optional[str]:
        """Generate interactive HTML report"""
        try:
            filename = f"smartcash_analysis_report_{timestamp}.html" if timestamp else "smartcash_analysis_report.html"
            output_path = self.output_dir / filename
            
            # Generate HTML content
            html_content = f"""
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartCash Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1280px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric-card {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; display: inline-block; min-width: 200px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
        .metric-label {{ color: #7f8c8d; font-size: 14px; }}
        .insights {{ background: #e8f6ff; padding: 20px; border-left: 4px solid #3498db; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; }}
        .timestamp {{ color: #7f8c8d; font-size: 12px; text-align: right; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 SmartCash Analysis Report</h1>
        
        <div class="insights">
            <h3>📋 Executive Summary</h3>
            <p>{self._extract_html_summary(analysis_results)}</p>
        </div>
        
        <h2>📊 Key Metrics</h2>
        {self._generate_html_metrics_cards(analysis_results)}
        
        <h2>💰 Currency Analysis Results</h2>
        {self._generate_html_currency_table(analysis_results)}
        
        <h2>🔍 Key Findings</h2>
        {self._generate_html_findings_list(analysis_results)}
        
        <div class="timestamp">
            Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"🌐 HTML report saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating HTML report: {str(e)}")
            return None
    
    def _generate_md_header(self, analysis_results: Dict[str, Any]) -> str:
        """Generate markdown header"""
        model_info = analysis_results.get('model_info', {})
        backbone = model_info.get('backbone', 'EfficientNet-B4')
        
        return f"""# 🔬 SmartCash YOLOv5-{backbone} Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model**: YOLOv5 dengan {backbone} backbone  
**Analysis Version**: {analysis_results.get('version', '1.0')}

---"""
    
    def _generate_md_recommendations(self, analysis_results: Dict[str, Any]) -> str:
        """Generate markdown recommendations section"""
        recommendations = analysis_results.get('recommendations', [])
        
        if not recommendations:
            return "## 💡 Recommendations\n\n*No specific recommendations available.*"
        
        content = ["## 💡 Recommendations", ""]
        for i, rec in enumerate(recommendations, 1):
            content.append(f"{i}. {rec}")
        
        return "\n".join(content)
    
    def _generate_md_technical_details(self, analysis_results: Dict[str, Any]) -> str:
        """Generate markdown technical details section"""
        content = ["## 🔧 Technical Details", ""]
        
        # Model configuration
        model_info = analysis_results.get('model_info', {})
        if model_info:
            content.extend([
                "### Model Configuration",
                f"- **Backbone**: {model_info.get('backbone', 'EfficientNet-B4')}",
                f"- **Model Size**: {model_info.get('model_size', 0.0):.1f}MB",
                f"- **Parameters**: {model_info.get('parameters', 0):,}",
                ""
            ])
        
        # Analysis metadata
        analysis_summary = analysis_results.get('analysis_summary', {})
        if analysis_summary:
            content.extend([
                "### Analysis Metadata",
                f"- **Images Analyzed**: {analysis_summary.get('total_images_analyzed', 0):,}",
                f"- **Analysis Duration**: {analysis_summary.get('analysis_duration', 0.0):.1f}s",
                f"- **Components**: {', '.join(analysis_summary.get('analysis_components', []))}",
                ""
            ])
        
        return "\n".join(content)
    
    def _extract_html_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Extract summary untuk HTML report"""
        summary_metrics = analysis_results.get('summary_metrics', {})
        overall_map = summary_metrics.get('overall_map', 0.0)
        overall_accuracy = summary_metrics.get('overall_accuracy', 0.0)
        
        return f"Model achieved {overall_map:.1%} mAP dan {overall_accuracy:.1%} accuracy pada currency detection tasks dengan multi-layer architecture."
    
    def _generate_html_metrics_cards(self, analysis_results: Dict[str, Any]) -> str:
        """Generate HTML metrics cards"""
        summary_metrics = analysis_results.get('summary_metrics', {})
        
        cards = []
        metrics = [
            ('Overall mAP', summary_metrics.get('overall_map', 0.0), '%'),
            ('Overall Accuracy', summary_metrics.get('overall_accuracy', 0.0), '%'),
            ('Avg Inference Time', summary_metrics.get('avg_inference_time', 0.0), 'ms'),
            ('Total Detections', summary_metrics.get('total_detections', 0), '')
        ]
        
        for label, value, unit in metrics:
            if unit == '%':
                display_value = f"{value:.1%}"
            elif label == 'Total Detections':
                display_value = f"{value:,}"
            else:
                display_value = f"{value:.1f}{unit}"
            
            cards.append(f"""
            <div class="metric-card">
                <div class="metric-value">{display_value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """)
        
        return "".join(cards)
    
    def _generate_html_currency_table(self, analysis_results: Dict[str, Any]) -> str:
        """Generate HTML currency analysis table"""
        class_analysis = analysis_results.get('class_analysis', {})
        per_class_metrics = class_analysis.get('per_class_metrics', {})
        
        if not per_class_metrics:
            return "<p><em>Currency analysis data not available.</em></p>"
        
        denominations = {
            '0': 'Rp 1.000', '1': 'Rp 2.000', '2': 'Rp 5.000', '3': 'Rp 10.000',
            '4': 'Rp 20.000', '5': 'Rp 50.000', '6': 'Rp 100.000'
        }
        
        table_rows = []
        for class_id, metrics in per_class_metrics.items():
            denom_name = denominations.get(class_id, f'Class {class_id}')
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            f1 = metrics.get('f1_score', 0.0)
            
            table_rows.append(f"""
            <tr>
                <td>{denom_name}</td>
                <td>{precision:.1%}</td>
                <td>{recall:.1%}</td>
                <td>{f1:.1%}</td>
            </tr>
            """)
        
        return f"""
        <table>
            <thead>
                <tr>
                    <th>Denomination</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                </tr>
            </thead>
            <tbody>
                {"".join(table_rows)}
            </tbody>
        </table>
        """
    
    def _generate_html_findings_list(self, analysis_results: Dict[str, Any]) -> str:
        """Generate HTML findings list"""
        key_findings = analysis_results.get('key_findings', [])
        
        if not key_findings:
            return "<p><em>No specific findings available.</em></p>"
        
        findings_html = ["<ul>"]
        for finding in key_findings[:10]:  # Limit to top 10
            findings_html.append(f"<li>{finding}</li>")
        findings_html.append("</ul>")
        
        return "".join(findings_html)
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp untuk file naming"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")