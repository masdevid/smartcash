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
            'train_loss',
            'val_loss',
            'loss_breakdown',
            'accuracy',  # Standard classification accuracy
            'precision', 
            'recall',
            'f1',
            'map50'     
        ]
        
        self.phase_2_focus_metrics = [
            'train_loss',
            'val_loss',
            'loss_breakdown',
            'accuracy',       # Standard classification accuracy
            'precision', 
            'recall',
            'f1',
            'map50'          
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
        prefix = "val_" if is_validation else "train_"
        standardized = {}
        
        # Handle loss metrics correctly - don't default to 0.0 if it might be legitimately missing
        if is_validation:
            # For validation metrics, convert 'loss' to 'val_loss'
            if 'loss' in raw_metrics:
                standardized["val_loss"] = raw_metrics['loss']
            elif 'val_loss' in raw_metrics:
                standardized["val_loss"] = raw_metrics['val_loss']
        else:
            # For training metrics, handle train_loss
            if 'train_loss' in raw_metrics:
                standardized["train_loss"] = raw_metrics['train_loss']
            else:
                # Only set default for training metrics if we're processing training data
                standardized["train_loss"] = raw_metrics.get('train_loss', 0.0)
        # Phase-aware handling of primary metrics (accuracy, precision, recall, f1)
        # Don't default to 0.0 - preserve original values or absence
        if phase_num == 1:
            # In Phase 1, primary metrics are sourced directly from Layer 1.
            if 'layer_1_accuracy' in raw_metrics:
                standardized[f"{prefix}accuracy"] = raw_metrics['layer_1_accuracy']
            if 'layer_1_precision' in raw_metrics:
                standardized[f"{prefix}precision"] = raw_metrics['layer_1_precision']
            if 'layer_1_recall' in raw_metrics:
                standardized[f"{prefix}recall"] = raw_metrics['layer_1_recall']
            if 'layer_1_f1' in raw_metrics:
                standardized[f"{prefix}f1"] = raw_metrics['layer_1_f1']
            logger.debug("Phase 1: Standardized primary metrics from Layer 1.")
        else:
            # In Phase 2 (and others), use the top-level hierarchical metrics.
            if 'accuracy' in raw_metrics:
                standardized[f"{prefix}accuracy"] = raw_metrics['accuracy']
            if 'precision' in raw_metrics:
                standardized[f"{prefix}precision"] = raw_metrics['precision']
            if 'recall' in raw_metrics:
                standardized[f"{prefix}recall"] = raw_metrics['recall']
            if 'f1' in raw_metrics:
                standardized[f"{prefix}f1"] = raw_metrics['f1']
            logger.debug(f"Phase {phase_num}: Standardized primary metrics from top-level.")

        # Add mAP-related metrics if they exist
        map_keys = ['map50', 'map50_95', 'map_precision', 'map_recall', 'map_f1']
        for key in map_keys:
            if key in raw_metrics:
                standardized[f"{prefix}{key}"] = raw_metrics[key]
        
        # Add all per-layer metrics for detailed analysis (preserve all values)
        for key, value in raw_metrics.items():
            if key.startswith('layer_'):
                standardized[f"{prefix}{key}"] = value

        # For validation, ensure standard names like val_accuracy are present
        # This ensures that val_* metrics are always consistent regardless of phase
        if is_validation:
            # In Phase 1, val_* should be the same as layer_1_* (only if available)
            if phase_num == 1:
                if 'layer_1_accuracy' in raw_metrics:
                    standardized["val_accuracy"] = raw_metrics['layer_1_accuracy']
                if 'layer_1_precision' in raw_metrics:
                    standardized["val_precision"] = raw_metrics['layer_1_precision']
                if 'layer_1_recall' in raw_metrics:
                    standardized["val_recall"] = raw_metrics['layer_1_recall']
                if 'layer_1_f1' in raw_metrics:
                    standardized["val_f1"] = raw_metrics['layer_1_f1']
            # In Phase 2, val_* should be the same as the top-level metrics (only if available)
            else:
                if 'accuracy' in raw_metrics:
                    standardized["val_accuracy"] = raw_metrics['accuracy']
                if 'precision' in raw_metrics:
                    standardized["val_precision"] = raw_metrics['precision']
                if 'recall' in raw_metrics:
                    standardized["val_recall"] = raw_metrics['recall']
                if 'f1' in raw_metrics:
                    standardized["val_f1"] = raw_metrics['f1']

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