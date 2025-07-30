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
                      ground_truth: Dict[str, np.ndarray] = None, phase_num: int = None):
        """
        Update metrics with new epoch data including confusion matrix calculation.
        
        Args:
            epoch: Current epoch number
            phase: Training phase (e.g., 'phase_1', 'phase_2', 'training', 'validation')
            metrics: Dictionary of metric values (using research-focused naming)
            predictions: Per-layer predictions (optional)
            ground_truth: Per-layer ground truth labels (optional)
            phase_num: Training phase number (1 or 2) for research context
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
            
            # Update training curves with research-focused metrics
            if 'train_loss' in metrics:
                self.train_losses.append(metrics['train_loss'])
            if 'val_loss' in metrics:
                self.val_losses.append(metrics['val_loss'])
            if 'learning_rate' in metrics:
                self.learning_rates.append(metrics['learning_rate'])
            
            # Store phase information for research context
            epoch_data['phase_num'] = phase_num
            epoch_data['research_context'] = self._get_research_context(phase_num)
            
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
    
    def generate_comprehensive_charts(self, session_id: str = None, phase_num: int = None) -> Dict[str, str]:
        """
        Generate comprehensive research-focused training charts and confusion matrices.
        
        Args:
            session_id: Unique identifier for this training session
            phase_num: Training phase number for research context
            
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
            
            # 4. Research-focused phase analysis with hypothesis testing context
            phase_analysis_path = self._generate_research_phase_analysis_chart(session_dir, phase_num)
            if phase_analysis_path:
                generated_charts['research_phase_analysis'] = str(phase_analysis_path)
            
            # 5. Learning rate schedule
            lr_schedule_path = self._generate_lr_schedule_chart(session_dir)
            if lr_schedule_path:
                generated_charts['lr_schedule'] = str(lr_schedule_path)
            
            # 6. Research-focused comprehensive dashboard
            dashboard_path = self._generate_research_dashboard(session_dir, phase_num)
            if dashboard_path:
                generated_charts['research_dashboard'] = str(dashboard_path)
                
            # 7. Phase 2 specific: mAP@0.5 trends chart
            if phase_num == 2:
                map_trends_path = self._generate_map_trends_chart(session_dir)
                if map_trends_path:
                    generated_charts['detection_map_trends'] = str(map_trends_path)
            
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
            
            # Research-focused accuracy trends - adapt based on available metrics
            research_accuracy_data = self._extract_research_accuracy_trends()
            if research_accuracy_data:
                for metric_name, values in research_accuracy_data.items():
                    if values:
                        axes[1, 0].plot(epochs[:len(values)], values, linewidth=2, 
                                      label=self._get_metric_display_name(metric_name))
                axes[1, 0].set_title('Research Accuracy Trends')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Accuracy')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Phase-appropriate secondary metrics
            secondary_data = self._extract_secondary_metrics_trends()
            if secondary_data:
                for metric_name, values in secondary_data.items():
                    if values:
                        axes[1, 1].plot(epochs[:len(values)], values, linewidth=2,
                                      label=self._get_metric_display_name(metric_name))
                axes[1, 1].set_title('Secondary Research Metrics')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Value')
                axes[1, 1].legend()
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
    
    def _generate_research_dashboard(self, session_dir: Path, phase_num: int = None) -> Optional[str]:
        """Generate a research-focused comprehensive training dashboard."""
        try:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # Research-focused title
            phase_context = self._get_dashboard_title_context(phase_num)
            fig.suptitle(f'SmartCash Research Dashboard - {phase_context}', fontsize=18, fontweight='bold')
            
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
            
            # Research-focused metrics (top right)
            ax2 = fig.add_subplot(gs[0, 2:])
            research_trends = self._extract_all_research_trends()
            if research_trends:
                for metric_name, values in research_trends.items():
                    if values and len(values) > 0:
                        epochs = range(1, len(values) + 1)
                        ax2.plot(epochs, values, label=self._get_metric_display_name(metric_name), linewidth=2)
                ax2.set_title('Research-Focused Metrics Trends')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.grid(True, alpha=0.3)
            else:
                # Fallback to layer metrics if research trends not available
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
            
            # Add research-focused summary text
            summary_text = self._generate_research_summary_text()
            fig.text(0.02, 0.02, summary_text, fontsize=10, verticalalignment='bottom', 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            
            chart_path = session_dir / f'research_dashboard_phase_{phase_num or "unknown"}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating research dashboard: {e}")
            return None
    
    def _generate_research_phase_analysis_chart(self, session_dir: Path, phase_num: int = None) -> Optional[str]:
        """Generate research-focused phase transition analysis chart."""
        try:
            if not self.phase_transitions or not self.train_losses:
                return None
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            epochs = range(1, len(self.train_losses) + 1)
            
            # Top plot: Loss curves with research context
            ax1.plot(epochs, self.train_losses, 'b-', linewidth=2, label='Training Loss')
            if self.val_losses:
                ax1.plot(epochs[:len(self.val_losses)], self.val_losses, 'r-', linewidth=2, label='Validation Loss')
            
            # Mark phase transitions with research context
            colors = ['orange', 'green', 'purple']
            for idx, transition in enumerate(self.phase_transitions):
                color = colors[idx % len(colors)]
                ax1.axvline(x=transition['epoch'], color=color, linestyle='--', linewidth=2, alpha=0.8)
                
                # Add research context to transition labels
                context = self._get_transition_research_context(transition)
                ax1.text(transition['epoch'], max(self.train_losses) * 0.9, 
                       context, rotation=90, verticalalignment='bottom', fontsize=10, color=color)
            
            ax1.set_title('Research Training Progress with Phase Transitions', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: Research metrics progression
            research_trends = self._extract_research_accuracy_trends()
            for metric_name, values in research_trends.items():
                if values:
                    metric_epochs = range(1, len(values) + 1)
                    display_name = self._get_metric_display_name(metric_name)
                    ax2.plot(metric_epochs, values, linewidth=2, marker='o', markersize=3, label=display_name)
            
            # Mark phase transitions on metrics plot too
            for transition in self.phase_transitions:
                ax2.axvline(x=transition['epoch'], color='orange', linestyle='--', alpha=0.7)
            
            ax2.set_title('Research Metrics Progression', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy / Performance')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            chart_path = session_dir / f'research_phase_analysis_phase_{phase_num or "unknown"}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating research phase analysis chart: {e}")
            return None
    
    def _generate_map_trends_chart(self, session_dir: Path) -> Optional[str]:
        """Generate mAP@0.5 trends chart for Phase 2 additional detection information."""
        try:
            # Extract mAP trends
            map_values = []
            for epoch_data in self.epoch_metrics:
                map_val = epoch_data.get('val_detection_map50')
                if map_val is not None and map_val > 0:
                    map_values.append(map_val)
            
            if not map_values:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            epochs = range(1, len(map_values) + 1)
            ax.plot(epochs, map_values, 'g-', linewidth=2, marker='o', markersize=4, label='Detection mAP@0.5')
            
            # Add trend line
            if len(map_values) > 1:
                z = np.polyfit(epochs, map_values, 1)
                p = np.poly1d(z)
                ax.plot(epochs, p(epochs), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.4f})')
            
            ax.set_title('Phase 2 Additional Detection Information: mAP@0.5 Trends', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP@0.5')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)  # mAP is between 0 and 1
            
            # Add research context annotation
            ax.text(0.02, 0.98, '🔬 Research Context: Additional detection performance metric\nfor multi-layer hierarchical banknote detection',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            chart_path = session_dir / 'detection_map50_trends.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(chart_path)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"⚠️ Error generating mAP trends chart: {e}")
            return None
    
    def _get_dashboard_title_context(self, phase_num: int = None) -> str:
        """Get dashboard title context based on phase."""
        if phase_num == 1:
            return "Phase 1: Single-Layer Denomination Detection"
        elif phase_num == 2:
            return "Phase 2: Multi-Layer Hierarchical Detection"
        else:
            return "Multi-Layer Banknote Detection Research"
    
    def _get_transition_research_context(self, transition: Dict) -> str:
        """Get research context for phase transitions."""
        from_phase = transition.get('from_phase', '')
        to_phase = transition.get('to_phase', '')
        
        if 'phase_1' in to_phase or '1' in to_phase:
            return 'Start\nDenomination\nDetection'
        elif 'phase_2' in to_phase or '2' in to_phase:
            return 'Begin\nHierarchical\nApproach'
        else:
            return f'{from_phase}\n→\n{to_phase}'
    
    def _generate_research_summary_text(self) -> str:
        """Generate research-focused summary text for the dashboard."""
        try:
            total_epochs = len(self.epoch_metrics)
            final_train_loss = self.train_losses[-1] if self.train_losses else 'N/A'
            final_val_loss = self.val_losses[-1] if self.val_losses else 'N/A'
            
            # Extract research-focused metrics from latest epoch
            research_summary = self._extract_research_summary_metrics()
            
            summary = f"""🔬 SmartCash Research Training Summary:
• Total Epochs: {total_epochs}
• Final Train Loss: {final_train_loss:.4f if isinstance(final_train_loss, (int, float)) else final_train_loss}
• Final Val Loss: {final_val_loss:.4f if isinstance(final_val_loss, (int, float)) else final_val_loss}
• Phase Transitions: {len(self.phase_transitions)}
{research_summary}"""
            
            return summary
            
        except Exception:
            return "🔬 Research Training Summary: Data processing error"
    
    def _get_research_context(self, phase_num: int) -> str:
        """Get research context description for the given phase."""
        if phase_num == 1:
            return "single_layer_denomination_detection"
        elif phase_num == 2:
            return "multi_layer_hierarchical_detection"
        else:
            return "unknown_phase"
    
    def _extract_research_accuracy_trends(self) -> Dict[str, List[float]]:
        """Extract research-focused accuracy trends from epoch metrics."""
        trends = {}
        
        # Primary research metrics to track
        research_metrics = [
            'val_denomination_accuracy',  # Phase 1 & 2 primary
            'val_hierarchical_accuracy',  # Phase 2 primary
            'train_denomination_accuracy',  # Training comparison
        ]
        
        for metric_name in research_metrics:
            values = []
            for epoch_data in self.epoch_metrics:
                if metric_name in epoch_data:
                    values.append(epoch_data[metric_name])
            if values:  # Only include if we have data
                trends[metric_name] = values
        
        # Fallback to legacy metrics if research metrics not available
        if not trends:
            legacy_metrics = ['val_accuracy', 'train_accuracy']
            for metric_name in legacy_metrics:
                values = []
                for epoch_data in self.epoch_metrics:
                    if metric_name in epoch_data:
                        values.append(epoch_data[metric_name])
                if values:
                    trends[metric_name] = values
        
        return trends
    
    def _extract_secondary_metrics_trends(self) -> Dict[str, List[float]]:
        """Extract secondary research metrics trends."""
        trends = {}
        
        # Phase-appropriate secondary metrics
        secondary_metrics = [
            'val_multi_layer_benefit',    # Phase 2: improvement benefit
            'val_detection_map50',        # Phase 2: additional detection info
            'val_denomination_f1',        # Both phases: F1 score
            'val_hierarchical_accuracy'   # Phase 2: combined accuracy
        ]
        
        for metric_name in secondary_metrics:
            values = []
            for epoch_data in self.epoch_metrics:
                if metric_name in epoch_data and epoch_data[metric_name] > 0:
                    values.append(epoch_data[metric_name])
            if values:  # Only include if we have meaningful data
                trends[metric_name] = values
        
        return trends
    
    def _extract_all_research_trends(self) -> Dict[str, List[float]]:
        """Extract all available research trends for comprehensive visualization."""
        all_trends = {}
        
        # Combine primary and secondary trends
        primary_trends = self._extract_research_accuracy_trends()
        secondary_trends = self._extract_secondary_metrics_trends()
        
        all_trends.update(primary_trends)
        all_trends.update(secondary_trends)
        
        return all_trends
    
    def _get_metric_display_name(self, metric_name: str) -> str:
        """Convert internal metric names to user-friendly display names."""
        display_names = {
            'val_denomination_accuracy': 'Denomination Accuracy (Val)',
            'train_denomination_accuracy': 'Denomination Accuracy (Train)',
            'val_hierarchical_accuracy': 'Hierarchical Accuracy (Val)',
            'val_multi_layer_benefit': 'Multi-Layer Benefit',
            'val_detection_map50': 'Detection mAP@0.5',
            'val_denomination_f1': 'Denomination F1 Score',
            'val_denomination_precision': 'Denomination Precision',
            'val_denomination_recall': 'Denomination Recall',
            'val_layer_1_contribution': 'Layer 1 Contribution',
            'val_layer_2_contribution': 'Layer 2 Contribution', 
            'val_layer_3_contribution': 'Layer 3 Contribution',
            # Legacy fallbacks
            'val_accuracy': 'Validation Accuracy',
            'train_accuracy': 'Training Accuracy',
            'val_precision': 'Validation Precision',
            'val_recall': 'Validation Recall',
            'val_f1': 'Validation F1'
        }
        
        return display_names.get(metric_name, metric_name.replace('_', ' ').title())
    
    def _extract_research_summary_metrics(self) -> str:
        """Extract key research metrics for summary display."""
        if not self.epoch_metrics:
            return "• No training data available"
        
        latest_metrics = self.epoch_metrics[-1]
        summary_lines = []
        
        # Phase-specific summaries
        phase_num = latest_metrics.get('phase_num')
        research_context = latest_metrics.get('research_context', '')
        
        if phase_num == 1 or 'single_layer' in research_context:
            # Phase 1: Focus on denomination detection
            denom_acc = latest_metrics.get('val_denomination_accuracy')
            if denom_acc is not None:
                summary_lines.append(f"• Denomination Accuracy: {denom_acc:.4f} ({denom_acc*100:.2f}%)")
            
            denom_f1 = latest_metrics.get('val_denomination_f1')
            if denom_f1 is not None:
                summary_lines.append(f"• Denomination F1: {denom_f1:.4f}")
                
        elif phase_num == 2 or 'multi_layer' in research_context:
            # Phase 2: Focus on hierarchical benefit
            hier_acc = latest_metrics.get('val_hierarchical_accuracy')
            if hier_acc is not None:
                summary_lines.append(f"• Hierarchical Accuracy: {hier_acc:.4f} ({hier_acc*100:.2f}%)")
            
            benefit = latest_metrics.get('val_multi_layer_benefit')
            if benefit is not None:
                summary_lines.append(f"• Multi-Layer Benefit: +{benefit:.4f} (+{benefit*100:.2f}%)")
            
            detection_map = latest_metrics.get('val_detection_map50')
            if detection_map is not None:
                summary_lines.append(f"• Detection mAP@0.5: {detection_map:.4f} ({detection_map*100:.2f}%)")
        
        # Fallback to best available metrics
        if not summary_lines:
            val_acc = latest_metrics.get('val_accuracy')
            if val_acc is not None:
                summary_lines.append(f"• Best Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        return '\n'.join(summary_lines) if summary_lines else "• Research metrics not available"
    
    def _save_metrics_summary(self, session_dir: Path, session_id: str):
        """Save comprehensive research-focused metrics summary to JSON."""
        try:
            # Extract research context from final metrics
            final_metrics = self.epoch_metrics[-1] if self.epoch_metrics else {}
            research_context = final_metrics.get('research_context', 'unknown')
            phase_num = final_metrics.get('phase_num')
            
            summary = {
                'session_id': session_id,
                'timestamp': time.time(),
                'total_epochs': len(self.epoch_metrics),
                'phase_transitions': self.phase_transitions,
                'final_metrics': final_metrics,
                'research_context': research_context,
                'phase_num': phase_num,
                'research_summary': self._generate_research_metrics_summary(),
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
                self.logger.warning(f"⚠️ Error saving research metrics summary: {e}")
    
    def _generate_research_metrics_summary(self) -> Dict[str, Any]:
        """Generate comprehensive research metrics summary."""
        if not self.epoch_metrics:
            return {}
        
        research_trends = self._extract_all_research_trends()
        
        summary = {
            'denomination_detection': {},
            'hierarchical_performance': {},
            'training_progression': {}
        }
        
        # Denomination detection analysis
        if 'val_denomination_accuracy' in research_trends:
            values = research_trends['val_denomination_accuracy']
            summary['denomination_detection'] = {
                'best_accuracy': max(values),
                'final_accuracy': values[-1],
                'improvement': values[-1] - values[0] if len(values) > 1 else 0.0,
                'epochs_to_best': values.index(max(values)) + 1
            }
        
        # Hierarchical performance analysis  
        if 'val_hierarchical_accuracy' in research_trends:
            values = research_trends['val_hierarchical_accuracy']
            summary['hierarchical_performance'] = {
                'best_hierarchical_accuracy': max(values),
                'final_hierarchical_accuracy': values[-1],
                'hierarchical_improvement': values[-1] - values[0] if len(values) > 1 else 0.0
            }
        
        if 'val_multi_layer_benefit' in research_trends:
            values = research_trends['val_multi_layer_benefit']
            summary['hierarchical_performance']['max_benefit'] = max(values)
            summary['hierarchical_performance']['final_benefit'] = values[-1]
        
        # Training progression
        summary['training_progression'] = {
            'total_epochs': len(self.epoch_metrics),
            'phase_transitions': len(self.phase_transitions),
            'loss_reduction': self.train_losses[-1] - self.train_losses[0] if len(self.train_losses) > 1 else 0.0
        }
        
        return summary
    
    def update_with_research_metrics(self, epoch: int, phase_num: int, metrics: Dict[str, Any], 
                                   predictions: Dict[str, np.ndarray] = None, 
                                   ground_truth: Dict[str, np.ndarray] = None):
        """
        Convenience method to update metrics with research context.
        
        Args:
            epoch: Current epoch number
            phase_num: Training phase number (1 or 2)
            metrics: Dictionary of research-focused metric values
            predictions: Per-layer predictions (optional)
            ground_truth: Per-layer ground truth labels (optional)
        """
        phase_name = f"phase_{phase_num}"
        self.update_metrics(epoch, phase_name, metrics, predictions, ground_truth, phase_num)


def create_visualization_manager(num_classes_per_layer: Dict[str, int], 
                               class_names: Dict[str, List[str]] = None,
                               save_dir: str = "data/visualization",
                               verbose: bool = False) -> ComprehensiveMetricsTracker:
    """
    Factory function to create a research-focused visualization manager for training.
    
    This creates a visualization manager optimized for SmartCash research with:
    - Research-focused metric tracking (denomination accuracy, hierarchical performance)
    - Phase-appropriate visualizations (Phase 1: single-layer, Phase 2: multi-layer + mAP)
    - Backward compatibility for existing UI components
    
    Args:
        num_classes_per_layer: Dictionary mapping layer names to number of classes
        class_names: Optional mapping of layer names to class name lists
        save_dir: Directory to save visualization outputs
        verbose: Enable verbose logging
        
    Returns:
        Configured ComprehensiveMetricsTracker instance with research focus
    """
    tracker = ComprehensiveMetricsTracker(
        num_classes_per_layer=num_classes_per_layer,
        class_names=class_names,
        save_dir=save_dir,
        verbose=verbose
    )
    
    if verbose:
        from smartcash.common.logger import get_logger
        logger = get_logger(__name__)
        logger.info("🔬 Research-focused visualization manager created")
        logger.info("   • Phase 1 focus: Single-layer denomination detection")
        logger.info("   • Phase 2 focus: Multi-layer hierarchical detection + mAP@0.5")
        logger.info("   • Backward compatibility: Legacy UI metrics supported")
    
    return tracker