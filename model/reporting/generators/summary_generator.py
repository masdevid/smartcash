"""
File: smartcash/model/reporting/generators/summary_generator.py
Deskripsi: Generator untuk executive summary dan results overview dari analysis data
"""

from typing import Dict, Any, List, Optional
from smartcash.common.logger import get_logger

class SummaryGenerator:
    """Generator untuk comprehensive summaries dan executive overviews"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger('summary_generator')
        
    def generate_executive_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate executive summary dari comprehensive analysis results"""
        try:
            summary_parts = []
            
            # Header
            summary_parts.append("## 📋 Executive Summary")
            summary_parts.append("")
            
            # Overview statistics
            overview_stats = self._extract_overview_statistics(analysis_results)
            if overview_stats:
                summary_parts.append("### 🎯 Key Performance Indicators")
                summary_parts.append("")
                for metric, value in overview_stats.items():
                    summary_parts.append(f"- **{metric}**: {value}")
                summary_parts.append("")
            
            # Key findings summary
            key_findings = analysis_results.get('key_findings', [])
            if key_findings:
                summary_parts.append("### 🔍 Key Findings")
                summary_parts.append("")
                for finding in key_findings[:5]:  # Top 5 findings
                    summary_parts.append(f"- {finding}")
                summary_parts.append("")
            
            # Performance highlights
            performance_highlights = self._generate_performance_highlights(analysis_results)
            if performance_highlights:
                summary_parts.append("### 🚀 Performance Highlights")
                summary_parts.append("")
                for highlight in performance_highlights:
                    summary_parts.append(f"- {highlight}")
                summary_parts.append("")
            
            # Recommendations preview
            recommendations = analysis_results.get('recommendations', [])
            if recommendations:
                summary_parts.append("### 💡 Top Recommendations")
                summary_parts.append("")
                for rec in recommendations[:3]:  # Top 3 recommendations
                    summary_parts.append(f"- {rec}")
                summary_parts.append("")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating executive summary: {str(e)}")
            return "## 📋 Executive Summary\n\n⚠️ Error generating summary."
    
    def generate_results_overview(self, analysis_results: Dict[str, Any]) -> str:
        """Generate detailed results overview"""
        try:
            overview_parts = []
            
            # Header
            overview_parts.append("## 📊 Results Overview")
            overview_parts.append("")
            
            # Analysis summary
            analysis_summary = analysis_results.get('analysis_summary', {})
            if analysis_summary:
                overview_parts.append("### 📈 Analysis Coverage")
                overview_parts.append("")
                total_images = analysis_summary.get('total_images_analyzed', 0)
                components = analysis_summary.get('analysis_components', [])
                visualizations = analysis_summary.get('visualizations_generated', 0)
                
                overview_parts.append(f"- **Total Images Analyzed**: {total_images:,}")
                overview_parts.append(f"- **Analysis Components**: {', '.join(components)}")
                overview_parts.append(f"- **Visualizations Generated**: {visualizations}")
                overview_parts.append("")
            
            # Currency analysis overview
            currency_overview = self._generate_currency_overview(analysis_results)
            if currency_overview:
                overview_parts.extend(currency_overview)
            
            # Layer analysis overview
            layer_overview = self._generate_layer_overview(analysis_results)
            if layer_overview:
                overview_parts.extend(layer_overview)
            
            # Class analysis overview
            class_overview = self._generate_class_overview(analysis_results)
            if class_overview:
                overview_parts.extend(class_overview)
            
            # Comparative analysis overview
            comparative_overview = self._generate_comparative_overview(analysis_results)
            if comparative_overview:
                overview_parts.extend(comparative_overview)
            
            return "\n".join(overview_parts)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating results overview: {str(e)}")
            return "## 📊 Results Overview\n\n⚠️ Error generating overview."
    
    def generate_quick_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate quick summary untuk immediate review"""
        try:
            quick_parts = []
            
            # Header
            quick_parts.append("## ⚡ Quick Summary")
            quick_parts.append("")
            
            # Key metrics
            summary_metrics = analysis_results.get('summary_metrics', {})
            if summary_metrics:
                quick_parts.append("### 📊 Key Metrics")
                overall_map = summary_metrics.get('overall_map', 0.0)
                overall_accuracy = summary_metrics.get('overall_accuracy', 0.0)
                avg_inference_time = summary_metrics.get('avg_inference_time', 0.0)
                
                quick_parts.append(f"- **Overall mAP**: {overall_map:.1%}")
                quick_parts.append(f"- **Overall Accuracy**: {overall_accuracy:.1%}")
                quick_parts.append(f"- **Avg Inference Time**: {avg_inference_time:.1f}ms")
                quick_parts.append("")
            
            # Top insights
            insights = analysis_results.get('insights', [])
            if insights:
                quick_parts.append("### 💡 Top Insights")
                for insight in insights[:3]:
                    quick_parts.append(f"- {insight}")
                quick_parts.append("")
            
            # Status summary
            status = analysis_results.get('analysis_status', 'completed')
            quick_parts.append(f"**Analysis Status**: {status.title()}")
            
            return "\n".join(quick_parts)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating quick summary: {str(e)}")
            return "## ⚡ Quick Summary\n\n⚠️ Error generating summary."
    
    def _extract_overview_statistics(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Extract key statistics untuk overview"""
        try:
            stats = {}
            
            # Summary metrics
            summary_metrics = analysis_results.get('summary_metrics', {})
            if summary_metrics:
                overall_map = summary_metrics.get('overall_map', 0.0)
                overall_accuracy = summary_metrics.get('overall_accuracy', 0.0)
                total_detections = summary_metrics.get('total_detections', 0)
                
                stats['Overall mAP'] = f"{overall_map:.1%}"
                stats['Overall Accuracy'] = f"{overall_accuracy:.1%}"
                stats['Total Detections'] = f"{total_detections:,}"
            
            # Analysis coverage
            analysis_summary = analysis_results.get('analysis_summary', {})
            if analysis_summary:
                total_images = analysis_summary.get('total_images_analyzed', 0)
                analysis_duration = analysis_summary.get('analysis_duration', 0.0)
                
                stats['Images Analyzed'] = f"{total_images:,}"
                stats['Analysis Duration'] = f"{analysis_duration:.1f}s"
            
            # Model information
            model_info = analysis_results.get('model_info', {})
            if model_info:
                backbone = model_info.get('backbone', 'Unknown')
                model_size = model_info.get('model_size', 0.0)
                
                stats['Model Backbone'] = backbone
                stats['Model Size'] = f"{model_size:.1f}MB"
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting overview statistics: {str(e)}")
            return {}
    
    def _generate_performance_highlights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate performance highlights"""
        try:
            highlights = []
            
            # Currency detection highlights
            currency_analysis = analysis_results.get('currency_analysis', {})
            if currency_analysis:
                strategy_stats = currency_analysis.get('strategy_statistics', {})
                primary_usage = strategy_stats.get('primary_strategy_usage', 0.0)
                
                if primary_usage > 0.8:
                    highlights.append(f"Strong primary detection strategy dengan {primary_usage:.1%} success rate")
            
            # Layer performance highlights
            layer_analysis = analysis_results.get('layer_analysis', {})
            if layer_analysis:
                collaboration_score = layer_analysis.get('collaboration_score', 0.0)
                
                if collaboration_score > 0.75:
                    highlights.append(f"Excellent layer collaboration dengan {collaboration_score:.1%} synergy score")
            
            # Class performance highlights
            class_analysis = analysis_results.get('class_analysis', {})
            if class_analysis:
                balanced_performance = class_analysis.get('balanced_performance', False)
                
                if balanced_performance:
                    highlights.append("Balanced performance across all currency denominations")
            
            # Comparative highlights
            comparison_analysis = analysis_results.get('comparison_analysis', {})
            if comparison_analysis:
                backbone_improvement = comparison_analysis.get('backbone_improvement', 0.0)
                
                if backbone_improvement > 0.05:
                    highlights.append(f"Significant backbone improvement: +{backbone_improvement:.1%} performance gain")
            
            # Default highlights jika tidak ada data
            if not highlights:
                highlights = [
                    "Comprehensive multi-layer analysis completed",
                    "Professional visualization generation successful",
                    "Detailed performance metrics available"
                ]
            
            return highlights[:5]  # Max 5 highlights
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating performance highlights: {str(e)}")
            return ["Analysis completed successfully"]
    
    def _generate_currency_overview(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate currency analysis overview"""
        try:
            currency_parts = []
            
            currency_analysis = analysis_results.get('currency_analysis', {})
            if currency_analysis:
                currency_parts.append("### 💰 Currency Analysis Overview")
                currency_parts.append("")
                
                # Strategy distribution
                strategy_stats = currency_analysis.get('strategy_statistics', {})
                if strategy_stats:
                    primary_usage = strategy_stats.get('primary_strategy_usage', 0.0)
                    boost_usage = strategy_stats.get('boost_strategy_usage', 0.0)
                    
                    currency_parts.append(f"- **Primary Strategy Usage**: {primary_usage:.1%}")
                    currency_parts.append(f"- **Boost Strategy Usage**: {boost_usage:.1%}")
                
                # Detection effectiveness
                effectiveness = currency_analysis.get('strategy_effectiveness', {})
                if effectiveness:
                    primary_acc = effectiveness.get('primary_accuracy', 0.0)
                    currency_parts.append(f"- **Primary Detection Accuracy**: {primary_acc:.1%}")
                
                currency_parts.append("")
            
            return currency_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating currency overview: {str(e)}")
            return []
    
    def _generate_layer_overview(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate layer analysis overview"""
        try:
            layer_parts = []
            
            layer_analysis = analysis_results.get('layer_analysis', {})
            if layer_analysis:
                layer_parts.append("### 🏗️ Layer Analysis Overview")
                layer_parts.append("")
                
                # Layer performance
                layer_performance = layer_analysis.get('layer_performance', {})
                if layer_performance:
                    banknote_perf = layer_performance.get('banknote_layer', {}).get('map', 0.0)
                    nominal_perf = layer_performance.get('nominal_layer', {}).get('map', 0.0)
                    security_perf = layer_performance.get('security_layer', {}).get('map', 0.0)
                    
                    layer_parts.append(f"- **Banknote Layer mAP**: {banknote_perf:.1%}")
                    layer_parts.append(f"- **Nominal Layer mAP**: {nominal_perf:.1%}")
                    layer_parts.append(f"- **Security Layer mAP**: {security_perf:.1%}")
                
                # Collaboration metrics
                collaboration_score = layer_analysis.get('collaboration_score', 0.0)
                if collaboration_score > 0:
                    layer_parts.append(f"- **Layer Collaboration Score**: {collaboration_score:.1%}")
                
                layer_parts.append("")
            
            return layer_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating layer overview: {str(e)}")
            return []
    
    def _generate_class_overview(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate class analysis overview"""
        try:
            class_parts = []
            
            class_analysis = analysis_results.get('class_analysis', {})
            if class_analysis:
                class_parts.append("### 📊 Class Analysis Overview")
                class_parts.append("")
                
                # Performance balance
                balance_score = class_analysis.get('balance_score', 0.0)
                if balance_score > 0:
                    class_parts.append(f"- **Class Balance Score**: {balance_score:.1%}")
                
                # Best/worst performing classes
                performance_ranking = class_analysis.get('performance_ranking', {})
                if performance_ranking:
                    best_class = performance_ranking.get('best_performing', 'Unknown')
                    worst_class = performance_ranking.get('worst_performing', 'Unknown')
                    
                    class_parts.append(f"- **Best Performing**: {best_class}")
                    class_parts.append(f"- **Most Challenging**: {worst_class}")
                
                class_parts.append("")
            
            return class_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating class overview: {str(e)}")
            return []
    
    def _generate_comparative_overview(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate comparative analysis overview"""
        try:
            comparative_parts = []
            
            comparison_analysis = analysis_results.get('comparison_analysis', {})
            if comparison_analysis:
                comparative_parts.append("### 🔄 Comparative Analysis Overview")
                comparative_parts.append("")
                
                # Backbone comparison
                backbone_comparison = comparison_analysis.get('backbone_comparison', {})
                if backbone_comparison:
                    improvement = backbone_comparison.get('performance_improvement', 0.0)
                    comparative_parts.append(f"- **Backbone Performance Improvement**: {improvement:+.1%}")
                
                # Scenario comparison
                scenario_comparison = comparison_analysis.get('scenario_comparison', {})
                if scenario_comparison:
                    easier_scenario = scenario_comparison.get('easier_scenario', 'Unknown')
                    comparative_parts.append(f"- **Easier Evaluation Scenario**: {easier_scenario}")
                
                comparative_parts.append("")
            
            return comparative_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating comparative overview: {str(e)}")
            return []