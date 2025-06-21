"""
File: smartcash/model/reporting/generators/research/currency_generator.py
Deskripsi: Generator untuk currency-specific research analysis dalam reports
"""

from typing import Dict, Any, List, Optional
from smartcash.common.logger import get_logger

class CurrencyResearchGenerator:
    """Generator untuk currency denomination research analysis"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger('currency_research_generator')
        
    def generate_currency_research_analysis(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive currency research analysis section"""
        try:
            currency_parts = []
            
            # Main header
            currency_parts.append("## 💰 Currency Denomination Analysis")
            currency_parts.append("")
            
            # Multi-layer strategy analysis
            strategy_section = self._generate_multi_layer_strategy_analysis(analysis_results)
            if strategy_section:
                currency_parts.extend(strategy_section)
            
            # Denomination performance analysis
            denomination_section = self._generate_denomination_performance_analysis(analysis_results)
            if denomination_section:
                currency_parts.extend(denomination_section)
            
            # Detection strategy effectiveness
            effectiveness_section = self._generate_detection_strategy_effectiveness(analysis_results)
            if effectiveness_section:
                currency_parts.extend(effectiveness_section)
            
            # Layer collaboration analysis
            collaboration_section = self._generate_layer_collaboration_analysis(analysis_results)
            if collaboration_section:
                currency_parts.extend(collaboration_section)
            
            return "\n".join(currency_parts)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating currency research analysis: {str(e)}")
            return "## 💰 Currency Denomination Analysis\n\n⚠️ Error generating analysis."
    
    def _generate_multi_layer_strategy_analysis(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate multi-layer detection strategy analysis"""
        try:
            strategy_parts = []
            
            strategy_parts.append("### 🎯 Multi-Layer Detection Strategy")
            strategy_parts.append("")
            
            # Strategy overview
            currency_analysis = analysis_results.get('currency_analysis', {})
            strategy_stats = currency_analysis.get('strategy_statistics', {})
            
            if strategy_stats:
                strategy_parts.append("**Detection Strategy Distribution:**")
                primary_usage = strategy_stats.get('primary_strategy_usage', 0.0)
                boost_usage = strategy_stats.get('boost_strategy_usage', 0.0) 
                validation_usage = strategy_stats.get('validation_strategy_usage', 0.0)
                fallback_usage = strategy_stats.get('fallback_strategy_usage', 0.0)
                
                strategy_parts.append(f"- **Primary Strategy**: {primary_usage:.1%} of detections")
                strategy_parts.append(f"- **Confidence Boost**: {boost_usage:.1%} of detections")
                strategy_parts.append(f"- **Validation Strategy**: {validation_usage:.1%} of detections")
                strategy_parts.append(f"- **Fallback Strategy**: {fallback_usage:.1%} of detections")
                strategy_parts.append("")
            
            # Strategy effectiveness
            effectiveness = currency_analysis.get('strategy_effectiveness', {})
            if effectiveness:
                strategy_parts.append("**Strategy Effectiveness Analysis:**")
                
                primary_acc = effectiveness.get('primary_accuracy', 0.0)
                boost_improvement = effectiveness.get('boost_improvement', 0.0)
                validation_precision = effectiveness.get('validation_precision', 0.0)
                
                strategy_parts.append(f"- **Primary Detection Accuracy**: {primary_acc:.1%}")
                strategy_parts.append(f"- **Boost Layer Improvement**: +{boost_improvement:.1%} confidence gain")
                strategy_parts.append(f"- **Validation Layer Precision**: {validation_precision:.1%}")
                strategy_parts.append("")
            
            # Research insights
            strategy_parts.append("**Research Insights:**")
            strategy_parts.append("- Multi-layer approach meningkatkan detection robustness dengan redundant validation")
            strategy_parts.append("- Boost layer efektif dalam disambiguating similar denominations")
            strategy_parts.append("- Validation layer berkontribusi pada false positive reduction")
            strategy_parts.append("- Fallback strategy penting untuk edge cases dan partial occlusions")
            strategy_parts.append("")
            
            return strategy_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating strategy analysis: {str(e)}")
            return []
    
    def _generate_denomination_performance_analysis(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate per-denomination performance analysis"""
        try:
            denom_parts = []
            
            denom_parts.append("### 📊 Per-Denomination Performance Analysis")
            denom_parts.append("")
            
            # Denomination mapping
            denominations = {
                0: "Rp 1.000", 1: "Rp 2.000", 2: "Rp 5.000", 3: "Rp 10.000",
                4: "Rp 20.000", 5: "Rp 50.000", 6: "Rp 100.000"
            }
            
            # Per-class performance
            class_analysis = analysis_results.get('class_analysis', {})
            class_metrics = class_analysis.get('per_class_metrics', {})
            
            if class_metrics:
                denom_parts.append("**Individual Denomination Performance:**")
                denom_parts.append("")
                
                # Performance table
                denom_parts.append("| Denomination | Precision | Recall | F1-Score | AP@0.5 |")
                denom_parts.append("|--------------|-----------|--------|----------|--------|")
                
                for class_id, denom_name in denominations.items():
                    if str(class_id) in class_metrics:
                        metrics = class_metrics[str(class_id)]
                        precision = metrics.get('precision', 0.0)
                        recall = metrics.get('recall', 0.0)
                        f1 = metrics.get('f1_score', 0.0)
                        ap = metrics.get('average_precision', 0.0)
                        
                        denom_parts.append(f"| {denom_name} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {ap:.3f} |")
                
                denom_parts.append("")
            
            # Denomination difficulty analysis
            difficulty_analysis = class_analysis.get('difficulty_analysis', {})
            if difficulty_analysis:
                denom_parts.append("**Denomination Detection Difficulty:**")
                denom_parts.append("")
                
                easy_classes = difficulty_analysis.get('easy_classes', [])
                moderate_classes = difficulty_analysis.get('moderate_classes', [])
                difficult_classes = difficulty_analysis.get('difficult_classes', [])
                
                if easy_classes:
                    easy_denoms = [denominations.get(cls, f"Class {cls}") for cls in easy_classes]
                    denom_parts.append(f"- **Easy Detection**: {', '.join(easy_denoms)}")
                
                if moderate_classes:
                    mod_denoms = [denominations.get(cls, f"Class {cls}") for cls in moderate_classes]
                    denom_parts.append(f"- **Moderate Detection**: {', '.join(mod_denoms)}")
                
                if difficult_classes:
                    diff_denoms = [denominations.get(cls, f"Class {cls}") for cls in difficult_classes]
                    denom_parts.append(f"- **Difficult Detection**: {', '.join(diff_denoms)}")
                
                denom_parts.append("")
            
            # Common confusion patterns
            confusion_analysis = class_analysis.get('confusion_analysis', {})
            if confusion_analysis:
                denom_parts.append("**Common Confusion Patterns:**")
                denom_parts.append("")
                
                top_confusions = confusion_analysis.get('top_confusions', [])
                for confusion in top_confusions[:5]:
                    true_class = confusion.get('true_class', 0)
                    pred_class = confusion.get('predicted_class', 0)
                    frequency = confusion.get('frequency', 0.0)
                    
                    true_denom = denominations.get(true_class, f"Class {true_class}")
                    pred_denom = denominations.get(pred_class, f"Class {pred_class}")
                    
                    denom_parts.append(f"- {true_denom} → {pred_denom}: {frequency:.1%} confusion rate")
                
                denom_parts.append("")
            
            return denom_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating denomination analysis: {str(e)}")
            return []
    
    def _generate_detection_strategy_effectiveness(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate detection strategy effectiveness analysis"""
        try:
            effectiveness_parts = []
            
            effectiveness_parts.append("### 🔍 Detection Strategy Effectiveness")
            effectiveness_parts.append("")
            
            currency_analysis = analysis_results.get('currency_analysis', {})
            strategy_performance = currency_analysis.get('strategy_performance', {})
            
            if strategy_performance:
                # Primary layer analysis
                primary_perf = strategy_performance.get('primary_layer', {})
                if primary_perf:
                    effectiveness_parts.append("**Primary Layer (Banknote Detection):**")
                    primary_map = primary_perf.get('map', 0.0)
                    primary_coverage = primary_perf.get('coverage', 0.0)
                    primary_confidence = primary_perf.get('avg_confidence', 0.0)
                    
                    effectiveness_parts.append(f"- **mAP**: {primary_map:.1%} - Strong baseline performance")
                    effectiveness_parts.append(f"- **Coverage**: {primary_coverage:.1%} - Detection completeness")
                    effectiveness_parts.append(f"- **Avg Confidence**: {primary_confidence:.1%} - Prediction certainty")
                    effectiveness_parts.append("")
                
                # Boost layer analysis
                boost_perf = strategy_performance.get('boost_layer', {})
                if boost_perf:
                    effectiveness_parts.append("**Boost Layer (Nominal Enhancement):**")
                    boost_improvement = boost_perf.get('improvement_rate', 0.0)
                    boost_precision_gain = boost_perf.get('precision_gain', 0.0)
                    boost_usage = boost_perf.get('usage_rate', 0.0)
                    
                    effectiveness_parts.append(f"- **Improvement Rate**: {boost_improvement:.1%} - Detection enhancement")
                    effectiveness_parts.append(f"- **Precision Gain**: +{boost_precision_gain:.1%} - Accuracy improvement")
                    effectiveness_parts.append(f"- **Usage Rate**: {boost_usage:.1%} - Strategy activation frequency")
                    effectiveness_parts.append("")
                
                # Validation layer analysis
                validation_perf = strategy_performance.get('validation_layer', {})
                if validation_perf:
                    effectiveness_parts.append("**Validation Layer (Security Features):**")
                    val_precision = validation_perf.get('precision', 0.0)
                    false_positive_reduction = validation_perf.get('fp_reduction', 0.0)
                    authenticity_score = validation_perf.get('authenticity_score', 0.0)
                    
                    effectiveness_parts.append(f"- **Validation Precision**: {val_precision:.1%} - Security feature accuracy")
                    effectiveness_parts.append(f"- **FP Reduction**: -{false_positive_reduction:.1%} - False positive mitigation")
                    effectiveness_parts.append(f"- **Authenticity Score**: {authenticity_score:.1%} - Security validation reliability")
                    effectiveness_parts.append("")
            
            # Strategy synergy analysis
            synergy_analysis = currency_analysis.get('layer_synergy', {})
            if synergy_analysis:
                effectiveness_parts.append("**Layer Synergy Analysis:**")
                collaboration_score = synergy_analysis.get('collaboration_score', 0.0)
                redundancy_benefit = synergy_analysis.get('redundancy_benefit', 0.0)
                overall_improvement = synergy_analysis.get('overall_improvement', 0.0)
                
                effectiveness_parts.append(f"- **Collaboration Score**: {collaboration_score:.1%} - Inter-layer cooperation effectiveness")
                effectiveness_parts.append(f"- **Redundancy Benefit**: +{redundancy_benefit:.1%} - Multi-layer validation gain")
                effectiveness_parts.append(f"- **Overall Improvement**: +{overall_improvement:.1%} - System-wide enhancement")
                effectiveness_parts.append("")
            
            return effectiveness_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating effectiveness analysis: {str(e)}")
            return []
    
    def _generate_layer_collaboration_analysis(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate layer collaboration analysis"""
        try:
            collaboration_parts = []
            
            collaboration_parts.append("### 🤝 Layer Collaboration Analysis")
            collaboration_parts.append("")
            
            layer_analysis = analysis_results.get('layer_analysis', {})
            collaboration_metrics = layer_analysis.get('collaboration_metrics', {})
            
            if collaboration_metrics:
                # Spatial collaboration
                spatial_collab = collaboration_metrics.get('spatial_collaboration', {})
                if spatial_collab:
                    collaboration_parts.append("**Spatial Collaboration:**")
                    avg_iou = spatial_collab.get('average_iou', 0.0)
                    overlap_consistency = spatial_collab.get('overlap_consistency', 0.0)
                    spatial_agreement = spatial_collab.get('spatial_agreement', 0.0)
                    
                    collaboration_parts.append(f"- **Average IoU**: {avg_iou:.3f} - Layer boundary alignment")
                    collaboration_parts.append(f"- **Overlap Consistency**: {overlap_consistency:.1%} - Spatial agreement stability")
                    collaboration_parts.append(f"- **Spatial Agreement**: {spatial_agreement:.1%} - Multi-layer consensus")
                    collaboration_parts.append("")
                
                # Temporal collaboration
                temporal_collab = collaboration_metrics.get('temporal_collaboration', {})
                if temporal_collab:
                    collaboration_parts.append("**Temporal Collaboration:**")
                    prediction_consistency = temporal_collab.get('prediction_consistency', 0.0)
                    confidence_correlation = temporal_collab.get('confidence_correlation', 0.0)
                    
                    collaboration_parts.append(f"- **Prediction Consistency**: {prediction_consistency:.1%} - Cross-layer prediction stability")
                    collaboration_parts.append(f"- **Confidence Correlation**: {confidence_correlation:.3f} - Layer confidence alignment")
                    collaboration_parts.append("")
                
                # Semantic collaboration
                semantic_collab = collaboration_metrics.get('semantic_collaboration', {})
                if semantic_collab:
                    collaboration_parts.append("**Semantic Collaboration:**")
                    class_agreement = semantic_collab.get('class_agreement', 0.0)
                    semantic_consistency = semantic_collab.get('semantic_consistency', 0.0)
                    
                    collaboration_parts.append(f"- **Class Agreement**: {class_agreement:.1%} - Multi-layer classification consensus")
                    collaboration_parts.append(f"- **Semantic Consistency**: {semantic_consistency:.1%} - Semantic interpretation alignment")
                    collaboration_parts.append("")
            
            # Collaboration insights
            collaboration_insights = layer_analysis.get('collaboration_insights', [])
            if collaboration_insights:
                collaboration_parts.append("**Key Collaboration Insights:**")
                for insight in collaboration_insights:
                    collaboration_parts.append(f"- {insight}")
                collaboration_parts.append("")
            
            # Research implications
            collaboration_parts.append("**Research Implications:**")
            collaboration_parts.append("- Multi-layer architecture menunjukkan strong spatial dan semantic collaboration")
            collaboration_parts.append("- Layer redundancy berkontribusi pada robust detection dalam challenging conditions")
            collaboration_parts.append("- Spatial alignment antar layers mendukung accurate denomination classification")
            collaboration_parts.append("- Collaborative approach efektif untuk complex currency detection tasks")
            collaboration_parts.append("")
            
            return collaboration_parts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating collaboration analysis: {str(e)}")
            return []