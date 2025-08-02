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
        """Initialize simplified metrics manager focused on standard YOLO metrics."""
        # Simplified metrics - focus on standard YOLO training metrics
        self.phase_1_focus_metrics = [
            'accuracy',  # Standard classification accuracy
            'precision', 
            'recall',
            'f1'
            # 'map50'      # Object detection mAP - disabled for performance
        ]
        
        self.phase_2_focus_metrics = [
            'accuracy',       # Standard classification accuracy
            'precision', 
            'recall',
            'f1'
            # 'map50'          # Object detection mAP - disabled for performance
        ]
    
    def standardize_metric_names(self, raw_metrics: Dict[str, float], phase_num: int, 
                                is_validation: bool = False) -> Dict[str, float]:
        """
        Convert raw metrics to standardized simple naming focused on YOLO metrics.
        
        Args:
            raw_metrics: Raw metrics from training/validation
            phase_num: Current phase (1 or 2)
            is_validation: True if these are validation metrics
            
        Returns:
            Dictionary with standardized metric names
        """
        standardized = {}
        prefix = "val_" if is_validation else "train_"
        
        # Simple approach: Just use the standard metrics from raw_metrics
        # Focus on core YOLO metrics: loss, accuracy, precision, recall, f1, map50
        
        # Loss (always included)
        standardized[f"{prefix}loss"] = raw_metrics.get('loss', raw_metrics.get('train_loss', raw_metrics.get('val_loss', 0.0)))
        
        # Classification metrics - use layer_1 if available, otherwise use generic
        if 'layer_1_accuracy' in raw_metrics:
            # Use layer_1 metrics (preferred for both Phase 1 and 2)
            standardized[f"{prefix}accuracy"] = raw_metrics.get('layer_1_accuracy', 0.0)
            standardized[f"{prefix}precision"] = raw_metrics.get('layer_1_precision', 0.0)
            standardized[f"{prefix}recall"] = raw_metrics.get('layer_1_recall', 0.0)
            standardized[f"{prefix}f1"] = raw_metrics.get('layer_1_f1', 0.0)
        else:
            # Use generic metrics as fallback
            standardized[f"{prefix}accuracy"] = raw_metrics.get('accuracy', 0.0)
            standardized[f"{prefix}precision"] = raw_metrics.get('precision', 0.0)
            standardized[f"{prefix}recall"] = raw_metrics.get('recall', 0.0)
            standardized[f"{prefix}f1"] = raw_metrics.get('f1', 0.0)
        
        # mAP metrics from YOLOv5 calculator
        if 'map50' in raw_metrics:
            standardized[f"{prefix}map50"] = raw_metrics.get('map50', 0.0)
        if 'map50_95' in raw_metrics:
            standardized[f"{prefix}map50_95"] = raw_metrics.get('map50_95', 0.0)
        
        # mAP-specific precision, recall, F1 (from mAP computation)
        if 'map_precision' in raw_metrics:
            standardized[f"{prefix}map50_precision"] = raw_metrics.get('map_precision', 0.0)
        if 'map_recall' in raw_metrics:
            standardized[f"{prefix}map50_recall"] = raw_metrics.get('map_recall', 0.0)
        if 'map_f1' in raw_metrics:
            standardized[f"{prefix}map50_f1"] = raw_metrics.get('map_f1', 0.0)
        
        # For validation metrics, also provide backward compatibility names
        if is_validation:
            # Ensure we have the expected validation metric names
            if f"{prefix}accuracy" in standardized:
                standardized["val_accuracy"] = standardized[f"{prefix}accuracy"]
            if f"{prefix}precision" in standardized:
                standardized["val_precision"] = standardized[f"{prefix}precision"]
            if f"{prefix}recall" in standardized:
                standardized["val_recall"] = standardized[f"{prefix}recall"]
            if f"{prefix}f1" in standardized:
                standardized["val_f1"] = standardized[f"{prefix}f1"]
            # mAP metrics for validation
            if f"{prefix}map50" in standardized:
                standardized["val_map50"] = standardized[f"{prefix}map50"]
            if f"{prefix}map50_95" in standardized:
                standardized["val_map50_95"] = standardized[f"{prefix}map50_95"]
            # mAP-specific metrics for validation
            if f"{prefix}map50_precision" in standardized:
                standardized["val_map50_precision"] = standardized[f"{prefix}map50_precision"]
            if f"{prefix}map50_recall" in standardized:
                standardized["val_map50_recall"] = standardized[f"{prefix}map50_recall"]
            if f"{prefix}map50_f1" in standardized:
                standardized["val_map50_f1"] = standardized[f"{prefix}map50_f1"]
        
        return standardized
    
    def get_best_model_criteria(self, phase_num: int) -> Dict[str, Any]:
        """
        Get simplified criteria for selecting the best model.
        
        Args:
            phase_num: Current training phase
            
        Returns:
            Dictionary with metric name and mode
        """
        # Use accuracy as primary metric for all phases
        return {
            'metric': 'val_accuracy',
            'mode': 'max',
            'fallback_metric': 'val_f1',
            'fallback_mode': 'max'
        }
    
    def generate_research_summary(self, phase_num: int, final_metrics: Dict[str, float]) -> str:
        """Generate simple summary of training results (classification metrics only)."""
        accuracy = final_metrics.get('val_accuracy', 0.0)
        f1 = final_metrics.get('val_f1', 0.0)
        precision = final_metrics.get('val_precision', 0.0)
        recall = final_metrics.get('val_recall', 0.0)
        
        summary = f"""
📊 PHASE {phase_num} TRAINING RESULTS:
• Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
• Validation F1: {f1:.4f} ({f1*100:.2f}%)
• Validation Precision: {precision:.4f} ({precision*100:.2f}%)
• Validation Recall: {recall:.4f} ({recall*100:.2f}%)
        """.strip()
        
        return summary
    
    def identify_metrics_source(self, metrics: Dict[str, float]) -> str:
        """
        Identify which metrics computation method was used based on available metrics.
        
        Args:
            metrics: Dictionary of computed metrics
            
        Returns:
            String describing the metrics source
        """
        has_layer_metrics = any(key.startswith('layer_') for key in metrics.keys())
        has_yolo_style = 'val_map50' in metrics and 'val_accuracy' in metrics
        
        if has_layer_metrics and has_yolo_style:
            return "YOLOv5 + Hierarchical"
        elif has_layer_metrics:
            return "Hierarchical Only"  
        elif has_yolo_style:
            return "YOLOv5 Only"
        else:
            return "Unknown"

    def log_phase_appropriate_metrics(self, phase_num: int, metrics: Dict[str, float]):
        """Log essential metrics only - detailed metrics are now stored in JSON."""
        # Log only core metrics summary - detailed data is in JSON files
        val_loss = metrics.get('val_loss', 0.0)
        val_map50 = metrics.get('val_map50', 0.0)
        val_accuracy = metrics.get('val_accuracy', 0.0)
        
        if val_map50 > 0:
            logger.info(f"📊 Phase {phase_num}: Val Loss={val_loss:.4f}, mAP@0.5={val_map50:.4f}, Accuracy={val_accuracy:.4f}")
        else:
            logger.info(f"📊 Phase {phase_num}: Val Loss={val_loss:.4f}, Accuracy={val_accuracy:.4f}")
        
        # Note: Detailed metrics available in JSON history files

# Global instance
_research_metrics_manager = None

def get_research_metrics_manager() -> ResearchMetricsManager:
    """Get global research metrics manager instance."""
    global _research_metrics_manager
    if _research_metrics_manager is None:
        _research_metrics_manager = ResearchMetricsManager()
    return _research_metrics_manager