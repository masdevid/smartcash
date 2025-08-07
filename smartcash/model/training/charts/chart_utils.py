from typing import Dict, List, Optional

class ChartUtils:
    """
    Utility class for chart-related operations.
    
    Provides helper methods for extracting and processing chart data.
    """
    
    @staticmethod
    def extract_research_accuracy_trends(epoch_metrics: List[Dict]) -> Dict[str, List[float]]:
        """
        Extract research-focused accuracy trends from epoch metrics.
        
        Args:
            epoch_metrics: List of epoch metrics dictionaries
            
        Returns:
            Dictionary mapping metric names to their values over time
        """
        trends = {}
        
        # Primary metrics to track
        research_metrics = [
            'val_accuracy',   # Primary validation metric
            'train_accuracy', # Training comparison
        ]
        
        for metric_name in research_metrics:
            values = []
            for epoch_data in epoch_metrics:
                if metric_name in epoch_data:
                    values.append(epoch_data[metric_name])
            if values:  # Only include if we have data
                trends[metric_name] = values
        
        # Fallback to legacy metrics if research metrics not available
        if not trends:
            legacy_metrics = ['val_accuracy', 'train_accuracy']
            for metric_name in legacy_metrics:
                values = []
                for epoch_data in epoch_metrics:
                    if metric_name in epoch_data:
                        values.append(epoch_data[metric_name])
                if values:
                    trends[metric_name] = values
        
        return trends

    @staticmethod
    def extract_secondary_metrics_trends(epoch_metrics: List[Dict]) -> Dict[str, List[float]]:
        """
        Extract secondary research metrics trends.
        
        Args:
            epoch_metrics: List of epoch metrics dictionaries
            
        Returns:
            Dictionary mapping metric names to their values over time
        """
        trends = {}
        
        # Secondary metrics
        secondary_metrics = [
            'val_map50',    # Object detection metric
            'val_f1',       # F1 score
            'val_precision', # Precision 
            'val_recall'     # Recall
        ]
        
        for metric_name in secondary_metrics:
            values = []
            for epoch_data in epoch_metrics:
                if metric_name in epoch_data and epoch_data[metric_name] > 0:
                    values.append(epoch_data[metric_name])
            if values:  # Only include if we have meaningful data
                trends[metric_name] = values
        
        return trends

    @staticmethod
    def extract_all_research_trends(epoch_metrics: List[Dict]) -> Dict[str, List[float]]:
        """
        Extract all available research trends for comprehensive visualization.
        
        Args:
            epoch_metrics: List of epoch metrics dictionaries
            
        Returns:
            Dictionary mapping all metric names to their values over time
        """
        all_trends = {}
        
        # Combine primary and secondary trends
        primary_trends = ChartUtils.extract_research_accuracy_trends(epoch_metrics)
        secondary_trends = ChartUtils.extract_secondary_metrics_trends(epoch_metrics)
        
        all_trends.update(primary_trends)
        all_trends.update(secondary_trends)
        
        return all_trends

    @staticmethod
    def get_research_summary_metrics(epoch_metrics: List[Dict]) -> str:
        """
        Extract key research metrics for summary display.
        
        Args:
            epoch_metrics: List of epoch metrics dictionaries
            
        Returns:
            Formatted string of research metrics summary
        """
        if not epoch_metrics:
            return "• No training data available"
        
        latest_metrics = epoch_metrics[-1]
        summary_lines = []
        
        # Simple summary for both phases
        val_acc = latest_metrics.get('val_accuracy')
        if val_acc is not None:
            summary_lines.append(f"• Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        val_f1 = latest_metrics.get('val_f1')
        if val_f1 is not None:
            summary_lines.append(f"• Validation F1: {val_f1:.4f}")
        
        val_map50 = latest_metrics.get('val_map50')
        if val_map50 is not None:
            summary_lines.append(f"• mAP@0.5: {val_map50:.4f} ({val_map50*100:.2f}%)")
        
        return '\n'.join(summary_lines) if summary_lines else "• Research metrics not available"
