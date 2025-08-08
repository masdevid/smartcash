import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict

# Visualization imports with graceful fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

from smartcash.common.logger import get_logger

class ChartGenerator:
    """
    Chart generation module for training visualization.
    
    This class handles all chart generation logic, including:
    - Training curves and loss tracking
    - Confusion matrices
    - Per-layer metrics comparison
    - Phase transition analysis
    - Learning rate schedules
    """
    
    def __init__(self, save_dir: str = "data/visualization", verbose: bool = False):
        """
        Initialize the chart generator.
        
        Args:
            save_dir: Directory to save visualization outputs
            verbose: Enable verbose logging
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Initialize logger
        self.logger = get_logger(f'{self.__class__.__module__}.{self.__class__.__name__}')
        
        if verbose:
            self.logger.info(f"📊 Chart generator initialized")
            self.logger.info(f"   • Save directory: {self.save_dir}")
            
    def generate_training_curves_chart(self, 
                                     train_losses: List[float], 
                                     val_losses: List[float], 
                                     learning_rates: List[float],
                                     phase_transitions: List[Dict],
                                     session_id: str) -> Optional[str]:
        """
        Generate training curves chart showing loss and learning rate.
        
        Args:
            train_losses: List of training loss values
            val_losses: List of validation loss values
            learning_rates: List of learning rate values
            phase_transitions: List of phase transition dictionaries
            session_id: Unique identifier for this training session
            
        Returns:
            Path to generated chart image or None if generation failed
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot losses
            epochs = range(1, len(train_losses) + 1)
            plt.plot(epochs, train_losses, 'b-', label='Training Loss')
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
            
            # Plot learning rate
            plt.twinx()
            plt.plot(epochs, learning_rates, 'g--', label='Learning Rate')
            
            # Add phase transitions
            for transition in phase_transitions:
                epoch = transition['epoch']
                plt.axvline(x=epoch, color='k', linestyle='--', alpha=0.3)
                
            # Add title and labels
            plt.title(f'Training Curves - Session {session_id}')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            
            # Save chart
            chart_path = self.save_dir / f'training_curves_{session_id}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating training curves chart: {str(e)}")
            return None

    def generate_confusion_matrix_chart(self, 
                                     confusion_matrix: np.ndarray, 
                                     class_names: List[str],
                                     accuracy: float,
                                     session_id: str,
                                     layer_name: str) -> Optional[str]:
        """
        Generate confusion matrix chart for a specific layer.
        
        Args:
            confusion_matrix: 2D numpy array of confusion matrix values
            class_names: List of class names
            accuracy: Overall accuracy score
            session_id: Unique identifier for this training session
            layer_name: Name of the layer (layer_1, layer_2, etc.)
            
        Returns:
            Path to generated chart image or None if generation failed
        """
        try:
            plt.figure(figsize=(12, 10))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            
            plt.title(f'Confusion Matrix - {layer_name}\nAccuracy: {accuracy:.4f}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            # Save chart
            chart_path = self.save_dir / f'confusion_matrix_{layer_name}_{session_id}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating confusion matrix chart: {str(e)}")
            return None

    def generate_metrics_trends_chart(self, 
                                    trends: Dict[str, List[float]],
                                    session_id: str,
                                    phase_num: int = None) -> Optional[str]:
        """
        Generate metrics trends chart showing accuracy and other metrics over time.
        
        Args:
            trends: Dictionary mapping metric names to their values over time
            session_id: Unique identifier for this training session
            phase_num: Current phase number (1 or 2)
            
        Returns:
            Path to generated chart image or None if generation failed
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot each metric
            for metric_name, values in trends.items():
                plt.plot(range(1, len(values) + 1), values, label=self._get_metric_display_name(metric_name))
            
            # Add title and labels
            plt.title(f'Metrics Trends - {self._get_dashboard_title_context(phase_num)}')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            
            # Save chart
            chart_path = self.save_dir / f'metrics_trends_{session_id}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating metrics trends chart: {str(e)}")
            return None

    def _get_dashboard_title_context(self, phase_num: int = None) -> str:
        """Get dashboard title context based on phase."""
        if phase_num == 1:
            return "Phase 1: Single-Layer Denomination Detection"
        elif phase_num == 2:
            return "Phase 2: Multi-Layer Hierarchical Detection"
        else:
            return "Multi-Layer Banknote Detection Research"

    def _get_metric_display_name(self, metric_name: str) -> str:
        """Convert internal metric names to user-friendly display names."""
        display_names = {
            'val_accuracy': 'Validation Accuracy',
            'train_accuracy': 'Training Accuracy',
            'val_precision': 'Validation Precision',
            'val_recall': 'Validation Recall',
            'val_f1': 'Validation F1',
            'val_map50': 'mAP@0.5'
        }
        return display_names.get(metric_name, metric_name.replace('_', ' ').title())

    def save_metrics_summary(self, 
                           session_dir: Path, 
                           session_id: str,
                           epoch_metrics: List[Dict],
                           phase_transitions: List[Dict],
                           layer_metrics: Dict[str, Dict[str, List[float]]],
                           confusion_matrices: Dict[str, List[Dict]]) -> None:
        """
        Save comprehensive metrics summary to JSON.
        
        Args:
            session_dir: Directory to save the summary
            session_id: Unique identifier for this training session
            epoch_metrics: List of epoch metrics dictionaries
            phase_transitions: List of phase transition dictionaries
            layer_metrics: Dictionary of layer-specific metrics
            confusion_matrices: Dictionary of confusion matrices per layer
        """
        try:
            final_metrics = epoch_metrics[-1] if epoch_metrics else {}
            research_context = final_metrics.get('research_context', 'unknown')
            phase_num = final_metrics.get('phase_num')
            
            summary = {
                'session_id': session_id,
                'timestamp': time.time(),
                'total_epochs': len(epoch_metrics),
                'phase_transitions': phase_transitions,
                'final_metrics': final_metrics,
                'research_context': research_context,
                'phase_num': phase_num,
                'layer_metrics_summary': {},
                'confusion_matrices_summary': {}
            }
            
            # Layer metrics summary
            for layer, layer_data in layer_metrics.items():
                summary['layer_metrics_summary'][layer] = {
                    'best_accuracy': max(layer_data.get('accuracy', [0])),
                    'best_precision': max(layer_data.get('precision', [0])),
                    'best_recall': max(layer_data.get('recall', [0])),
                    'best_f1_score': max(layer_data.get('f1_score', [0])),
                    'final_accuracy': layer_data.get('accuracy', [0])[-1] if layer_data.get('accuracy') else 0
                }
            
            # Confusion matrices summary
            for layer, cms in confusion_matrices.items():
                if cms:
                    latest_cm = cms[-1]
                    summary['confusion_matrices_summary'][layer] = {
                        'final_accuracy': latest_cm['accuracy'],
                        'matrix_shape': latest_cm['matrix'].shape,
                        'total_samples': int(np.sum(latest_cm['matrix']))
                    }
            
            summary_path = session_dir / 'metrics_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            if self.verbose:
                self.logger.info(f"📊 Metrics summary saved: {summary_path.name}")
                
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error saving metrics summary: {str(e)}")
