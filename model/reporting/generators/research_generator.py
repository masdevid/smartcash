"""
File: smartcash/model/reporting/generators/research_generator.py
Deskripsi: Base generator untuk research-specific reports dengan delegation ke specialized modules
"""

from typing import Dict, Any, List, Optional
from smartcash.common.logger import get_logger
from .research import (
    MethodologyReportGenerator,
    CurrencyResearchGenerator, 
    PerformanceResearchGenerator,
    RecommendationGenerator
)

class ResearchGenerator:
    """Main research report generator yang koordinasi specialized generators"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger('research_generator')
        
        # Initialize specialized generators
        self.methodology_gen = MethodologyReportGenerator(self.config, self.logger)
        self.currency_gen = CurrencyResearchGenerator(self.config, self.logger) 
        self.performance_gen = PerformanceResearchGenerator(self.config, self.logger)
        self.recommendation_gen = RecommendationGenerator(self.config, self.logger)
    
    def generate_comprehensive_research_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive research report dengan all sections"""
        try:
            self.logger.info("📝 Generating comprehensive research report...")
            
            report_sections = []
            
            # Main header
            report_sections.append("# 🔬 SmartCash YOLOv5-EfficientNet Research Report")
            report_sections.append("")
            report_sections.append(f"**Generated**: {self._get_timestamp()}")
            report_sections.append(f"**Analysis Version**: {analysis_results.get('version', '1.0')}")
            report_sections.append("")
            
            # Executive abstract
            abstract = self._generate_research_abstract(analysis_results)
            if abstract:
                report_sections.append(abstract)
                report_sections.append("")
            
            # Methodology section
            methodology_section = self.methodology_gen.generate_methodology_section(analysis_results)
            report_sections.append(methodology_section)
            report_sections.append("")
            
            # Currency research analysis
            currency_section = self.currency_gen.generate_currency_research_analysis(analysis_results)
            report_sections.append(currency_section)
            report_sections.append("")
            
            # Performance research analysis
            performance_section = self.performance_gen.generate_performance_research_analysis(analysis_results)
            report_sections.append(performance_section)
            report_sections.append("")
            
            # Recommendations & future work
            recommendations_section = self.recommendation_gen.generate_research_recommendations(analysis_results)
            report_sections.append(recommendations_section)
            report_sections.append("")
            
            # Research conclusions
            conclusions = self._generate_research_conclusions(analysis_results)
            if conclusions:
                report_sections.append(conclusions)
            
            self.logger.info("✅ Research report generated successfully")
            return "\n".join(report_sections)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating research report: {str(e)}")
            return "# 🔬 SmartCash Research Report\n\n⚠️ Error generating research report."
    
    def generate_methodology_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate methodology-focused report"""
        return self.methodology_gen.generate_methodology_section(analysis_results)
    
    def generate_currency_analysis_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate currency analysis research report"""
        return self.currency_gen.generate_currency_research_analysis(analysis_results)
    
    def generate_performance_analysis_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate performance analysis research report"""
        return self.performance_gen.generate_performance_research_analysis(analysis_results)
    
    def generate_quick_research_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate quick research summary untuk immediate review"""
        try:
            summary_parts = []
            
            # Header
            summary_parts.append("## 🔬 Research Summary")
            summary_parts.append("")
            
            # Key research findings
            key_findings = analysis_results.get('research_findings', {})
            if key_findings:
                summary_parts.append("### 🎯 Key Research Findings")
                summary_parts.append("")
                
                # Currency findings
                currency_findings = key_findings.get('currency_analysis', [])
                if currency_findings:
                    summary_parts.append("**Currency Detection Analysis:**")
                    for finding in currency_findings[:3]:
                        summary_parts.append(f"- {finding}")
                    summary_parts.append("")
                
                # Performance findings  
                performance_findings = key_findings.get('performance_analysis', [])
                if performance_findings:
                    summary_parts.append("**Performance Analysis:**")
                    for finding in performance_findings[:3]:
                        summary_parts.append(f"- {finding}")
                    summary_parts.append("")
            
            # Research metrics overview
            metrics_overview = self._generate_research_metrics_overview(analysis_results)
            if metrics_overview:
                summary_parts.extend(metrics_overview)
            
            # Top recommendations
            recommendations = analysis_results.get('recommendations', [])
            if recommendations:
                summary_parts.append("### 💡 Key Recommendations")
                summary_parts.append("")
                for rec in recommendations[:3]:
                    summary_parts.append(f"- {rec}")
                summary_parts.append("")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating research summary: {str(e)}")
            return "## 🔬 Research Summary\n\n⚠️ Error generating summary."
    
    def _generate_research_abstract(self, analysis_results: Dict[str, Any]) -> str:
        """Generate research abstract section"""
        try:
            abstract_parts = []
            
            abstract_parts.append("## 📋 Abstract")
            abstract_parts.append("")
            
            # Research objective
            objective = self.config.get('research', {}).get('objective', 
                'Evaluation of YOLOv5 with EfficientNet-B4 backbone untuk currency denomination detection')
            abstract_parts.append(f"**Objective**: {objective}")
            abstract_parts.append("")
            
            # Key metrics summary
            metrics_summary = analysis_results.get('summary_metrics', {})
            if metrics_summary:
                map_score = metrics_summary.get('overall_map', 0.0)
                accuracy = metrics_summary.get('overall_accuracy', 0.0)
                
                abstract_parts.append(f"**Results**: Model achieved {map_score:.1%} mAP dengan {accuracy:.1%} accuracy pada currency detection tasks.")
                abstract_parts.append("")
            
            # Research scope
            scope_items = [
                "Multi-layer currency denomination analysis",
                "Backbone architecture comparison",
                "Evaluation scenario robustness testing",
                "Performance efficiency analysis"
            ]
            abstract_parts.append(f"**Scope**: {', '.join(scope_items)}")
            abstract_parts.append("")
            
            return "\n".join(abstract_parts)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating abstract: {str(e)}")
            return ""
    
    def _generate_research_conclusions(self, analysis_results: Dict[str, Any]) -> str:
        """Generate research conclusions section"""
        try:
            conclusion_parts = []
            
            conclusion_parts.append("## 🎯 Research Conclusions")
            conclusion_parts.append("")
            
            # Main conclusions
            conclusions = analysis_results.get('conclusions', [])
            if conclusions:
                conclusion_parts.append("### 📊 Key Conclusions")
                conclusion_parts.append("")
                for idx, conclusion in enumerate(conclusions, 1):
                    conclusion_parts.append(f"{idx}. {conclusion}")
                conclusion_parts.append("")
            
            # Research limitations
            limitations = analysis_results.get('limitations', [])
            if limitations:
                conclusion_parts.append("### ⚠️ Research Limitations")
                conclusion_parts.append("")
                for limitation in limitations:
                    conclusion_parts.append(f"- {limitation}")
                conclusion_parts.append("")
            
            # Future research directions
            future_work = analysis_results.get('future_work', [])
            if future_work:
                conclusion_parts.append("### 🚀 Future Research Directions")
                conclusion_parts.append("")
                for work in future_work:
                    conclusion_parts.append(f"- {work}")
                conclusion_parts.append("")
            
            return "\n".join(conclusion_parts)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating conclusions: {str(e)}")
            return ""
    
    def _generate_research_metrics_overview(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate research metrics overview section"""
        try:
            overview_parts = []
            
            overview_parts.append("### 📈 Research Metrics Overview")
            overview_parts.append("")
            
            # Performance metrics
            metrics = analysis_results.get('summary_metrics', {})
            if metrics:
                overview_parts.append("**Detection Performance:**")
                overview_parts.append(f"- mAP@0.5: {metrics.get('overall_map', 0.0):.1%}")
                overview_parts.append(f"- Accuracy: {metrics.get('overall_accuracy', 0.0):.1%}")
                overview_parts.append(f"- Precision: {metrics.get('overall_precision', 0.0):.1%}")
                overview_parts.append(f"- Recall: {metrics.get('overall_recall', 0.0):.1%}")
                overview_parts.append("")
            
            # Analysis coverage
            analysis_summary = analysis_results.get('analysis_summary', {})
            if analysis_summary:
                total_images = analysis_summary.get('total_images_analyzed', 0)
                components = len(analysis_summary.get('analysis_components', []))
                
                overview_parts.append("**Analysis Coverage:**")
                overview_parts.append(f"- Images Analyzed: {total_images:,}")
                overview_parts.append(f"- Analysis Components: {components}")
                overview_parts.append("")
            
            return overview_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating metrics overview: {str(e)}")
            return []
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp untuk reports"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")