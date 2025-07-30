#!/usr/bin/env python3
"""
Research-focused metrics system for SmartCash multi-layer banknote detection.

This module provides clear, research-aligned metrics that directly support
the hypothesis testing of multi-layer hierarchical banknote detection.

Research Goals:
- Phase 1: Test if Layer 1 (denomination detection) can be trained effectively
- Phase 2: Test if multi-layer approach (Layer 1+2+3) improves denomination accuracy

Key Research Metrics:
- Denomination Accuracy: Core metric for banknote classification research
- Layer Performance: Individual layer effectiveness measurement  
- Hierarchical Benefit: Improvement from multi-layer vs single-layer approach
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class ResearchMetricsManager:
    """Manages research-focused metrics with clear naming and phase-appropriate selection."""
    
    def __init__(self):
        """Initialize research metrics manager."""
        self.phase_1_focus_metrics = [
            'denomination_accuracy',  # Primary research metric: can we detect denominations?
            'denomination_precision', 
            'denomination_recall',
            'denomination_f1'
        ]
        
        self.phase_2_focus_metrics = [
            'denomination_accuracy',      # Still primary: did multi-layer improve denomination detection?
            'hierarchical_accuracy',      # Combined multi-layer performance
            'layer_1_contribution',       # Individual layer contributions
            'layer_2_contribution',
            'layer_3_contribution',
            'multi_layer_benefit',        # Improvement over single-layer
            'detection_map50'             # Additional detection performance information
        ]
    
    def standardize_metric_names(self, raw_metrics: Dict[str, float], phase_num: int, 
                                is_validation: bool = False) -> Dict[str, float]:
        """
        Convert raw metrics to standardized research-focused naming.
        
        Args:
            raw_metrics: Raw metrics from training/validation
            phase_num: Current phase (1 or 2)
            is_validation: True if these are validation metrics
            
        Returns:
            Dictionary with standardized metric names
        """
        standardized = {}
        prefix = "val_" if is_validation else "train_"
        
        # Phase 1: Focus on denomination detection (Layer 1 only)
        if phase_num == 1:
            # Map layer_1 metrics to denomination metrics (clearer research meaning)
            layer_1_accuracy = raw_metrics.get('layer_1_accuracy', raw_metrics.get('accuracy', 0.0))
            layer_1_precision = raw_metrics.get('layer_1_precision', raw_metrics.get('precision', 0.0))
            layer_1_recall = raw_metrics.get('layer_1_recall', raw_metrics.get('recall', 0.0))
            layer_1_f1 = raw_metrics.get('layer_1_f1', raw_metrics.get('f1', 0.0))
            
            standardized.update({
                f"{prefix}denomination_accuracy": layer_1_accuracy,
                f"{prefix}denomination_precision": layer_1_precision,
                f"{prefix}denomination_recall": layer_1_recall,
                f"{prefix}denomination_f1": layer_1_f1,
                f"{prefix}loss": raw_metrics.get('loss', 0.0)
            })
            
            # Research interpretation metrics
            standardized[f"{prefix}research_primary_metric"] = layer_1_accuracy
            standardized[f"{prefix}research_interpretation"] = "single_layer_denomination_detection"
            
        # Phase 2: Focus on multi-layer hierarchical benefit
        elif phase_num == 2:
            # Individual layer contributions
            layer_1_acc = raw_metrics.get('layer_1_accuracy', 0.0)
            layer_2_acc = raw_metrics.get('layer_2_accuracy', 0.0) 
            layer_3_acc = raw_metrics.get('layer_3_accuracy', 0.0)
            
            # Calculate hierarchical metrics
            hierarchical_accuracy = self._calculate_hierarchical_accuracy(raw_metrics)
            multi_layer_benefit = self._calculate_multi_layer_benefit(raw_metrics)
            
            standardized.update({
                # Core research metrics
                f"{prefix}denomination_accuracy": layer_1_acc,  # Still primary for comparison
                f"{prefix}hierarchical_accuracy": hierarchical_accuracy,
                f"{prefix}multi_layer_benefit": multi_layer_benefit,
                
                # Individual layer contributions  
                f"{prefix}layer_1_contribution": layer_1_acc,
                f"{prefix}layer_2_contribution": layer_2_acc,
                f"{prefix}layer_3_contribution": layer_3_acc,
                
                # Supporting metrics
                f"{prefix}loss": raw_metrics.get('loss', 0.0),
                f"{prefix}denomination_precision": raw_metrics.get('layer_1_precision', 0.0),
                f"{prefix}denomination_recall": raw_metrics.get('layer_1_recall', 0.0),
                f"{prefix}denomination_f1": raw_metrics.get('layer_1_f1', 0.0),
                
                # Additional detection information (Phase 2 only)
                f"{prefix}detection_map50": raw_metrics.get('detection_map50', raw_metrics.get('map50', 0.0))
            })
            
            # Ensure detection_map50 is included in Phase 2
            if f"{prefix}detection_map50" not in standardized:
                standardized[f"{prefix}detection_map50"] = 0.0
            
            # Research interpretation metrics
            standardized[f"{prefix}research_primary_metric"] = hierarchical_accuracy
            standardized[f"{prefix}research_interpretation"] = "multi_layer_hierarchical_detection"
        
        # Add backward compatibility metrics for UI and progress tracking
        if is_validation:
            # Provide legacy metric names for backward compatibility
            if f"{prefix}denomination_accuracy" in standardized:
                standardized["val_accuracy"] = standardized[f"{prefix}denomination_accuracy"]
            if f"{prefix}denomination_precision" in standardized:
                standardized["val_precision"] = standardized[f"{prefix}denomination_precision"]
            if f"{prefix}denomination_recall" in standardized:
                standardized["val_recall"] = standardized[f"{prefix}denomination_recall"]
            if f"{prefix}denomination_f1" in standardized:
                standardized["val_f1"] = standardized[f"{prefix}denomination_f1"]
        
        # Remove confusing metrics that don't align with research goals
        return self._filter_research_relevant_metrics(standardized, phase_num)
    
    def _calculate_hierarchical_accuracy(self, raw_metrics: Dict[str, float]) -> float:
        """Calculate combined hierarchical accuracy across all active layers."""
        layer_1_acc = raw_metrics.get('layer_1_accuracy', 0.0)
        layer_2_acc = raw_metrics.get('layer_2_accuracy', 0.0)
        layer_3_acc = raw_metrics.get('layer_3_accuracy', 0.0)
        
        # Weighted average: Layer 1 most important (denomination), others supportive
        hierarchical_acc = (0.6 * layer_1_acc + 0.25 * layer_2_acc + 0.15 * layer_3_acc)
        return hierarchical_acc
    
    def _calculate_multi_layer_benefit(self, raw_metrics: Dict[str, float]) -> float:
        """Calculate improvement benefit from using multiple layers vs single layer."""
        layer_1_acc = raw_metrics.get('layer_1_accuracy', 0.0)
        hierarchical_acc = self._calculate_hierarchical_accuracy(raw_metrics)
        
        # Benefit = hierarchical performance - single layer performance
        benefit = max(0.0, hierarchical_acc - layer_1_acc)
        return benefit
    
    def _filter_research_relevant_metrics(self, metrics: Dict[str, float], phase_num: int) -> Dict[str, float]:
        """Remove metrics that don't support the research goals."""
        # Metrics to exclude (confusing/irrelevant for denomination research)
        excluded_patterns = [
            'map50_95',  # Not relevant for classification research
            'map75',     # Object detection metric, not classification  
            'box_loss',  # Low-level loss, not research-relevant
            'obj_loss',  # Low-level loss, not research-relevant
            'cls_loss'   # Low-level loss, not research-relevant
        ]
        
        # In Phase 2, allow map50 and detection_map50 as additional information
        if phase_num == 2:
            # Remove 'map50' from exclusions for Phase 2
            excluded_patterns = [p for p in excluded_patterns if p != 'map50']
        
        filtered = {}
        for key, value in metrics.items():
            # Skip if key contains any excluded pattern
            if not any(pattern in key.lower() for pattern in excluded_patterns):
                filtered[key] = value
            else:
                logger.debug(f"Filtered out non-research metric: {key}")
        
        return filtered
    
    def get_best_model_criteria(self, phase_num: int) -> Dict[str, Any]:
        """
        Get research-focused criteria for selecting the best model in each phase.
        
        Args:
            phase_num: Current training phase
            
        Returns:
            Dictionary with metric name, mode, and research justification
        """
        if phase_num == 1:
            return {
                'metric': 'val_denomination_accuracy',
                'mode': 'max',
                'research_justification': 'Phase 1 tests single-layer denomination detection capability',
                'fallback_metric': 'val_denomination_f1',
                'fallback_mode': 'max'
            }
        elif phase_num == 2:
            return {
                'metric': 'val_hierarchical_accuracy', 
                'mode': 'max',
                'research_justification': 'Phase 2 tests multi-layer hierarchical detection improvement',
                'fallback_metric': 'val_multi_layer_benefit',
                'fallback_mode': 'max'
            }
        else:
            # Default fallback
            return {
                'metric': 'val_denomination_accuracy',
                'mode': 'max', 
                'research_justification': 'Default denomination detection focus',
                'fallback_metric': 'val_loss',
                'fallback_mode': 'min'
            }
    
    def generate_research_summary(self, phase_num: int, final_metrics: Dict[str, float]) -> str:
        """Generate research-focused summary of training results."""
        if phase_num == 1:
            denom_acc = final_metrics.get('val_denomination_accuracy', 0.0)
            summary = f"""
🔬 PHASE 1 RESEARCH RESULTS - Single-Layer Denomination Detection:
• Denomination Accuracy: {denom_acc:.4f} ({denom_acc*100:.2f}%)
• Research Question: Can Layer 1 alone detect Indonesian Rupiah denominations?
• Result: {'SUCCESS' if denom_acc > 0.7 else 'MODERATE' if denom_acc > 0.5 else 'NEEDS_IMPROVEMENT'}
• Interpretation: {'Strong single-layer performance' if denom_acc > 0.7 else 'Baseline established for multi-layer comparison'}
            """.strip()
        
        elif phase_num == 2:
            denom_acc = final_metrics.get('val_denomination_accuracy', 0.0)
            hierarchical_acc = final_metrics.get('val_hierarchical_accuracy', 0.0)
            benefit = final_metrics.get('val_multi_layer_benefit', 0.0)
            detection_map50 = final_metrics.get('val_detection_map50', 0.0)
            
            summary = f"""
🔬 PHASE 2 RESEARCH RESULTS - Multi-Layer Hierarchical Detection:
• Denomination Accuracy: {denom_acc:.4f} ({denom_acc*100:.2f}%)
• Hierarchical Accuracy: {hierarchical_acc:.4f} ({hierarchical_acc*100:.2f}%)
• Multi-Layer Benefit: +{benefit:.4f} (+{benefit*100:.2f}% improvement)
• Additional Detection Info: mAP@0.5 = {detection_map50:.4f} ({detection_map50*100:.2f}%)
• Research Question: Does multi-layer approach improve denomination detection?
• Result: {'SIGNIFICANT_IMPROVEMENT' if benefit > 0.05 else 'MODERATE_IMPROVEMENT' if benefit > 0.02 else 'MINIMAL_BENEFIT'}
• Interpretation: {'Multi-layer architecture provides substantial benefit' if benefit > 0.05 else 'Limited improvement from hierarchical approach'}
            """.strip()
        
        return summary
    
    def log_phase_appropriate_metrics(self, phase_num: int, metrics: Dict[str, float]):
        """Log only the metrics relevant to the current phase and research goals."""
        if phase_num == 1:
            logger.info("📊 PHASE 1 - Single-Layer Denomination Detection Metrics:")
            for metric in self.phase_1_focus_metrics:
                for prefix in ['train_', 'val_']:
                    key = f"{prefix}{metric}"
                    if key in metrics:
                        logger.info(f"   • {key}: {metrics[key]:.6f}")
        
        elif phase_num == 2:
            logger.info("📊 PHASE 2 - Multi-Layer Hierarchical Detection Metrics:")
            for metric in self.phase_2_focus_metrics:
                for prefix in ['train_', 'val_']:
                    key = f"{prefix}{metric}"
                    if key in metrics:
                        logger.info(f"   • {key}: {metrics[key]:.6f}")

# Global instance
_research_metrics_manager = None

def get_research_metrics_manager() -> ResearchMetricsManager:
    """Get global research metrics manager instance."""
    global _research_metrics_manager
    if _research_metrics_manager is None:
        _research_metrics_manager = ResearchMetricsManager()
    return _research_metrics_manager