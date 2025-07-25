#!/usr/bin/env python3
"""
File: /Users/masdevid/Projects/smartcash/smartcash/model/training/visualization_manager.py

Comprehensive training visualization and metrics tracking for SmartCash multi-layer detection model.

This module provides advanced visualization capabilities including:
- Training curves and loss tracking
- Confusion matrices for each detection layer
- Per-layer metrics comparison
- Phase transition analysis
- Learning rate schedules
- Comprehensive training dashboards

Features:
- Real-time metrics tracking during training
- Automatic chart generation with matplotlib/seaborn
- Support for multi-layer detection architectures
- Phase-aware visualization for multi-phase training
- Export capabilities for training analysis
"""

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

class ComprehensiveMetricsTracker:
    """
    Enhanced metrics tracker with confusion matrix and visualization capabilities.
    
    This class handles comprehensive tracking of training metrics across multiple
    detection layers, supporting the SmartCash multi-layer architecture with
    layer_1 (banknote detection), layer_2 (denomination features), and 
    layer_3 (common features).
    """
    
    def __init__(self, num_classes_per_layer: Dict[str, int], class_names: Dict[str, List[str]] = None, 
                 save_dir: str = "data/visualization", verbose: bool = False):
        """
        Initialize the comprehensive metrics tracker.
        
        Args:
            num_classes_per_layer: Dictionary mapping layer names to number of classes
            class_names: Optional mapping of layer names to class name lists
            save_dir: Directory to save visualization outputs
            verbose: Enable verbose logging
        """
        self.num_classes_per_layer = num_classes_per_layer
        self.class_names = class_names or self._generate_default_class_names()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Initialize logger
        self.logger = get_logger(f'{__name__}.ComprehensiveMetricsTracker')
        
        # Metrics storage
        self.epoch_metrics = []
        self.confusion_matrices = defaultdict(list)  # Per layer
        self.predictions_history = defaultdict(list)
        self.ground_truth_history = defaultdict(list)
        
        # Training curves data
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.phase_transitions = []  # Track phase changes
        
        # Per-layer metrics
        self.layer_metrics = {
            layer: {
                'precision': [],
                'recall': [],
                'f1_score': [],
                'accuracy': []
            } for layer in num_classes_per_layer.keys()
        }
        
        if verbose:
            self.logger.info(f"📊 Comprehensive metrics tracker initialized")
            self.logger.info(f"   • Layers: {list(num_classes_per_layer.keys())}")
            self.logger.info(f"   • Classes per layer: {num_classes_per_layer}")
            self.logger.info(f"   • Save directory: {self.save_dir}")
    
    def _generate_default_class_names(self) -> Dict[str, List[str]]:
        """
        Generate default class names based on SmartCash MODEL_ARC.md specifications.
        
        Returns:
            Dictionary mapping layer names to class name lists
        """
        return {
            'layer_1': ['001', '002', '005', '010', '020', '050', '100'],  # Banknote denominations
            'layer_2': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],  # Denomination features
            'layer_3': ['l3_sign', 'l3_text', 'l3_thread']  # Common features
        }
    
    def update_metrics(self, epoch: int, phase: str, metrics: Dict[str, Any], 
                      predictions: Dict[str, np.ndarray] = None, 
                      ground_truth: Dict[str, np.ndarray] = None):
        """
        Update metrics with new epoch data including confusion matrix calculation.
        
        Args:
            epoch: Current epoch number
            phase: Training phase (e.g., 'phase_1', 'phase_2', 'training', 'validation')
            metrics: Dictionary of metric values
            predictions: Per-layer predictions (optional)
            ground_truth: Per-layer ground truth labels (optional)
        """
        try:
            # Store basic metrics
            epoch_data = {
                'epoch': epoch,
                'phase': phase,
                'timestamp': time.time(),
                **metrics
            }
            self.epoch_metrics.append(epoch_data)
            
            # Update training curves
            if 'train_loss' in metrics:
                self.train_losses.append(metrics['train_loss'])
            if 'val_loss' in metrics:
                self.val_losses.append(metrics['val_loss'])
            if 'learning_rate' in metrics:
                self.learning_rates.append(metrics['learning_rate'])
            
            # Calculate confusion matrices for each layer
            if predictions is not None and ground_truth is not None:
                for layer in self.num_classes_per_layer.keys():
                    if layer in predictions and layer in ground_truth:
                        self._calculate_layer_confusion_matrix(layer, predictions[layer], ground_truth[layer], epoch)
            
            # Track phase transitions
            if len(self.epoch_metrics) > 1 and self.epoch_metrics[-2]['phase'] != phase:
                self.phase_transitions.append({
                    'epoch': epoch,
                    'from_phase': self.epoch_metrics[-2]['phase'],
                    'to_phase': phase
                })
                if self.verbose:
                    self.logger.info(f"📍 Phase transition at epoch {epoch}: {self.epoch_metrics[-2]['phase']} → {phase}")
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error updating metrics: {e}")
    
    def _calculate_layer_confusion_matrix(self, layer: str, predictions: np.ndarray, ground_truth: np.ndarray, epoch: int):
        """
        Calculate confusion matrix for a specific layer.
        
        Args:
            layer: Layer name (e.g., 'layer_1', 'layer_2', 'layer_3')
            predictions: Model predictions for this layer
            ground_truth: Ground truth labels for this layer
            epoch: Current epoch number
        """
        try:
            # Convert predictions and ground truth to class indices
            pred_classes = np.argmax(predictions, axis=-1) if predictions.ndim > 1 else predictions
            true_classes = np.argmax(ground_truth, axis=-1) if ground_truth.ndim > 1 else ground_truth
            
            # Calculate confusion matrix
            cm = confusion_matrix(true_classes.flatten(), pred_classes.flatten(), 
                                labels=range(self.num_classes_per_layer[layer]))
            
            # Store confusion matrix
            cm_data = {
                'epoch': epoch,
                'matrix': cm,
                'accuracy': np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
            }
            self.confusion_matrices[layer].append(cm_data)
            
            # Calculate per-class metrics
            if np.sum(cm) > 0:
                precision = np.diag(cm) / (np.sum(cm, axis=0) + 1e-10)
                recall = np.diag(cm) / (np.sum(cm, axis=1) + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
                
                # Store layer metrics
                self.layer_metrics[layer]['precision'].append(np.mean(precision))
                self.layer_metrics[layer]['recall'].append(np.mean(recall))
                self.layer_metrics[layer]['f1_score'].append(np.mean(f1))
                self.layer_metrics[layer]['accuracy'].append(cm_data['accuracy'])
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error calculating confusion matrix for {layer}: {e}")
    
    def generate_comprehensive_charts(self, session_id: str = None) -> Dict[str, str]:
        """
        Generate comprehensive training charts and confusion matrices.
        
        Args:
            session_id: Unique identifier for this training session
            
        Returns:
            Dictionary mapping chart types to file paths
        """
        if not VISUALIZATION_AVAILABLE:
            if self.verbose:
                self.logger.warning("⚠️ Visualization libraries not available. Skipping chart generation.")
            return {}
        
        session_id = session_id or f"training_{int(time.time())}"
        session_dir = self.save_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        generated_charts = {}
        
        try:
            # 1. Training curves
            training_curves_path = self._generate_training_curves(session_dir)
            if training_curves_path:
                generated_charts['training_curves'] = str(training_curves_path)
            
            # 2. Confusion matrices for each layer
            for layer in self.num_classes_per_layer.keys():
                cm_path = self._generate_confusion_matrix_chart(layer, session_dir)
                if cm_path:
                    generated_charts[f'confusion_matrix_{layer}'] = str(cm_path)
            
            # 3. Per-layer metrics comparison
            layer_metrics_path = self._generate_layer_metrics_chart(session_dir)
            if layer_metrics_path:
                generated_charts['layer_metrics'] = str(layer_metrics_path)
            
            # 4. Phase transition analysis
            phase_analysis_path = self._generate_phase_analysis_chart(session_dir)
            if phase_analysis_path:
                generated_charts['phase_analysis'] = str(phase_analysis_path)
            
            # 5. Learning rate schedule
            lr_schedule_path = self._generate_lr_schedule_chart(session_dir)
            if lr_schedule_path:
                generated_charts['lr_schedule'] = str(lr_schedule_path)
            
            # 6. Comprehensive dashboard
            dashboard_path = self._generate_comprehensive_dashboard(session_dir)
            if dashboard_path:
                generated_charts['dashboard'] = str(dashboard_path)
            
            # Save metrics summary
            self._save_metrics_summary(session_dir, session_id)
            
            if self.verbose:
                self.logger.info(f"📊 Generated {len(generated_charts)} visualization charts")
                for chart_type, path in generated_charts.items():
                    self.logger.info(f"   • {chart_type}: {Path(path).name}")
            
            return generated_charts
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"❌ Error generating charts: {e}")
            return {}
    
    def _generate_training_curves(self, session_dir: Path) -> Optional[str]:
        """Generate training and validation loss curves."""
        try:
            if not self.train_losses and not self.val_losses:
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Progress Curves', fontsize=16, fontweight='bold')
            
            epochs = range(1, len(self.train_losses) + 1)
            
            # Loss curves
            if self.train_losses:
                axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
            if self.val_losses:
                axes[0, 0].plot(epochs[:len(self.val_losses)], self.val_losses, 'r-', label='Validation Loss', linewidth=2)
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Learning rate schedule
            if self.learning_rates:
                axes[0, 1].plot(epochs[:len(self.learning_rates)], self.learning_rates, 'g-', linewidth=2)
                axes[0, 1].set_title('Learning Rate Schedule')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Learning Rate')
                axes[0, 1].set_yscale('log')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Phase transitions
            for transition in self.phase_transitions:
                for ax in axes.flat:
                    ax.axvline(x=transition['epoch'], color='orange', linestyle='--', alpha=0.7)
            
            # Accuracy trends (if available)
            layer_accuracies = self.layer_metrics.get('layer_1', {}).get('accuracy', [])
            if layer_accuracies:
                axes[1, 0].plot(epochs[:len(layer_accuracies)], layer_accuracies, 'm-', linewidth=2)
                axes[1, 0].set_title('Layer 1 Accuracy')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Accuracy')
                axes[1, 0].grid(True, alpha=0.3)
            
            # F1 Score trends
            layer_f1 = self.layer_metrics.get('layer_1', {}).get('f1_score', [])
            if layer_f1:
                axes[1, 1].plot(epochs[:len(layer_f1)], layer_f1, 'c-', linewidth=2)
                axes[1, 1].set_title('Layer 1 F1 Score')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('F1 Score')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            chart_path = session_dir / 'training_curves.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating training curves: {e}")
            return None
    
    def _generate_confusion_matrix_chart(self, layer: str, session_dir: Path) -> Optional[str]:
        """Generate confusion matrix heatmap for a specific layer."""
        try:
            layer_cms = self.confusion_matrices.get(layer, [])
            if not layer_cms:
                return None
            
            # Use the latest confusion matrix
            latest_cm = layer_cms[-1]['matrix']
            class_names = self.class_names.get(layer, [f'Class_{i}' for i in range(len(latest_cm))])
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Confusion Matrix Analysis - {layer.upper()}', fontsize=14, fontweight='bold')
            
            # Raw confusion matrix
            sns.heatmap(latest_cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, ax=axes[0])
            axes[0].set_title('Raw Counts')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('Actual')
            
            # Normalized confusion matrix
            cm_normalized = latest_cm.astype('float') / (latest_cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Reds',
                       xticklabels=class_names, yticklabels=class_names, ax=axes[1])
            axes[1].set_title('Normalized (by True Class)')
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('Actual')
            
            plt.tight_layout()
            
            chart_path = session_dir / f'confusion_matrix_{layer}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating confusion matrix for {layer}: {e}")
            return None
    
    def _generate_layer_metrics_chart(self, session_dir: Path) -> Optional[str]:
        """Generate per-layer metrics comparison chart."""
        try:
            if not self.layer_metrics:
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Per-Layer Metrics Comparison', fontsize=16, fontweight='bold')
            
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
            metric_titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            
            for idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
                ax = axes[idx // 2, idx % 2]
                
                for layer, layer_data in self.layer_metrics.items():
                    metric_values = layer_data.get(metric, [])
                    if metric_values:
                        epochs = range(1, len(metric_values) + 1)
                        ax.plot(epochs, metric_values, label=layer, linewidth=2, marker='o', markersize=4)
                
                ax.set_title(title)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
            
            plt.tight_layout()
            
            chart_path = session_dir / 'layer_metrics_comparison.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating layer metrics chart: {e}")
            return None
    
    def _generate_phase_analysis_chart(self, session_dir: Path) -> Optional[str]:
        """Generate phase transition analysis chart."""
        try:
            if not self.phase_transitions or not self.train_losses:
                return None
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            epochs = range(1, len(self.train_losses) + 1)
            ax.plot(epochs, self.train_losses, 'b-', linewidth=2, label='Training Loss')
            
            if self.val_losses:
                ax.plot(epochs[:len(self.val_losses)], self.val_losses, 'r-', linewidth=2, label='Validation Loss')
            
            # Mark phase transitions
            colors = ['orange', 'green', 'purple']
            for idx, transition in enumerate(self.phase_transitions):
                color = colors[idx % len(colors)]
                ax.axvline(x=transition['epoch'], color=color, linestyle='--', linewidth=2, alpha=0.8)
                ax.text(transition['epoch'], max(self.train_losses) * 0.9, 
                       f"{transition['from_phase']} → {transition['to_phase']}", 
                       rotation=90, verticalalignment='bottom', fontsize=10, color=color)
            
            ax.set_title('Training Progress with Phase Transitions', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            chart_path = session_dir / 'phase_analysis.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating phase analysis chart: {e}")
            return None
    
    def _generate_lr_schedule_chart(self, session_dir: Path) -> Optional[str]:
        """Generate learning rate schedule chart."""
        try:
            if not self.learning_rates:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            epochs = range(1, len(self.learning_rates) + 1)
            ax.plot(epochs, self.learning_rates, 'g-', linewidth=2, marker='o', markersize=3)
            
            # Mark phase transitions
            for transition in self.phase_transitions:
                ax.axvline(x=transition['epoch'], color='orange', linestyle='--', alpha=0.7)
            
            ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            chart_path = session_dir / 'learning_rate_schedule.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating LR schedule chart: {e}")
            return None
    
    def _generate_comprehensive_dashboard(self, session_dir: Path) -> Optional[str]:
        """Generate a comprehensive training dashboard."""
        try:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # Title
            fig.suptitle('SmartCash Training Dashboard - Multi-Layer Detection', fontsize=18, fontweight='bold')
            
            # Training curves (top left)
            ax1 = fig.add_subplot(gs[0, :2])
            if self.train_losses:
                epochs = range(1, len(self.train_losses) + 1)
                ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
            if self.val_losses:
                ax1.plot(epochs[:len(self.val_losses)], self.val_losses, 'r-', label='Validation Loss', linewidth=2)
            ax1.set_title('Loss Curves')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Layer metrics (top right)
            ax2 = fig.add_subplot(gs[0, 2:])
            for layer, layer_data in self.layer_metrics.items():
                accuracy_values = layer_data.get('accuracy', [])
                if accuracy_values:
                    epochs = range(1, len(accuracy_values) + 1)
                    ax2.plot(epochs, accuracy_values, label=f'{layer} Accuracy', linewidth=2)
            ax2.set_title('Per-Layer Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Confusion matrices (middle row)
            cm_positions = [gs[1, 0], gs[1, 1], gs[1, 2]]
            for idx, (layer, position) in enumerate(zip(self.num_classes_per_layer.keys(), cm_positions)):
                if layer in self.confusion_matrices and self.confusion_matrices[layer]:
                    ax = fig.add_subplot(position)
                    latest_cm = self.confusion_matrices[layer][-1]['matrix']
                    sns.heatmap(latest_cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                    ax.set_title(f'CM - {layer.upper()}')
            
            # Phase transitions and summary stats (bottom)
            ax3 = fig.add_subplot(gs[2, :])
            if self.train_losses:
                epochs = range(1, len(self.train_losses) + 1)
                ax3.plot(epochs, self.train_losses, 'b-', alpha=0.7, linewidth=1)
                
                # Mark phase transitions
                for transition in self.phase_transitions:
                    ax3.axvline(x=transition['epoch'], color='red', linestyle='--', linewidth=2)
                    ax3.text(transition['epoch'], max(self.train_losses) * 0.8, 
                           f"Phase {transition['from_phase']} → {transition['to_phase']}", 
                           rotation=90, verticalalignment='bottom', fontsize=8)
            
            ax3.set_title('Training Timeline with Phase Transitions')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Training Loss')
            ax3.grid(True, alpha=0.3)
            
            # Add summary text
            summary_text = self._generate_summary_text()
            fig.text(0.02, 0.02, summary_text, fontsize=10, verticalalignment='bottom', 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            
            chart_path = session_dir / 'comprehensive_dashboard.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating comprehensive dashboard: {e}")
            return None
    
    def _generate_summary_text(self) -> str:
        """Generate summary text for the dashboard."""
        try:
            total_epochs = len(self.epoch_metrics)
            final_train_loss = self.train_losses[-1] if self.train_losses else 'N/A'
            final_val_loss = self.val_losses[-1] if self.val_losses else 'N/A'
            
            # Calculate best accuracies per layer
            best_accuracies = {}
            for layer, layer_data in self.layer_metrics.items():
                accuracies = layer_data.get('accuracy', [])
                best_accuracies[layer] = max(accuracies) if accuracies else 0
            
            summary = f"""Training Summary:
• Total Epochs: {total_epochs}
• Final Train Loss: {final_train_loss:.4f if isinstance(final_train_loss, (int, float)) else final_train_loss}
• Final Val Loss: {final_val_loss:.4f if isinstance(final_val_loss, (int, float)) else final_val_loss}
• Phase Transitions: {len(self.phase_transitions)}
• Best Accuracies: {', '.join([f'{k}: {v:.3f}' for k, v in best_accuracies.items()])}"""
            
            return summary
            
        except Exception:
            return "Training Summary: Data processing error"
    
    def _save_metrics_summary(self, session_dir: Path, session_id: str):
        """Save comprehensive metrics summary to JSON."""
        try:
            summary = {
                'session_id': session_id,
                'timestamp': time.time(),
                'total_epochs': len(self.epoch_metrics),
                'phase_transitions': self.phase_transitions,
                'final_metrics': self.epoch_metrics[-1] if self.epoch_metrics else {},
                'layer_metrics_summary': {},
                'confusion_matrices_summary': {}
            }
            
            # Layer metrics summary
            for layer, layer_data in self.layer_metrics.items():
                summary['layer_metrics_summary'][layer] = {
                    'best_accuracy': max(layer_data.get('accuracy', [0])),
                    'best_precision': max(layer_data.get('precision', [0])),
                    'best_recall': max(layer_data.get('recall', [0])),
                    'best_f1_score': max(layer_data.get('f1_score', [0])),
                    'final_accuracy': layer_data.get('accuracy', [0])[-1] if layer_data.get('accuracy') else 0
                }
            
            # Confusion matrices summary
            for layer, cms in self.confusion_matrices.items():
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
                self.logger.warning(f"⚠️ Error saving metrics summary: {e}")


def create_visualization_manager(num_classes_per_layer: Dict[str, int], 
                               class_names: Dict[str, List[str]] = None,
                               save_dir: str = "data/visualization",
                               verbose: bool = False) -> ComprehensiveMetricsTracker:
    """
    Factory function to create a visualization manager for training.
    
    Args:
        num_classes_per_layer: Dictionary mapping layer names to number of classes
        class_names: Optional mapping of layer names to class name lists
        save_dir: Directory to save visualization outputs
        verbose: Enable verbose logging
        
    Returns:
        Configured ComprehensiveMetricsTracker instance
    """
    return ComprehensiveMetricsTracker(
        num_classes_per_layer=num_classes_per_layer,
        class_names=class_names,
        save_dir=save_dir,
        verbose=verbose
    )