"""
File: smartcash/model/analysis/visualization/confusion_matrix_viz.py
Deskripsi: Specialized confusion matrix visualization dengan enhanced styling dan analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import confusion_matrix
from smartcash.common.logger import get_logger

class ConfusionMatrixVisualizer:
    """Specialized visualizer untuk confusion matrix dengan enhanced analysis"""
    
    def __init__(self, config: Dict[str, Any] = None, output_dir: str = 'data/analysis/confusion_matrices', logger=None):
        self.config = config or {}
        self.logger = logger or get_logger('confusion_matrix_viz')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        viz_config = self.config.get('visualization', {})
        self.figure_size = viz_config.get('confusion_matrix', {}).get('figure_size', [10, 8])
        self.dpi = viz_config.get('charts', {}).get('dpi', 150)
        self.cmap = viz_config.get('confusion_matrix', {}).get('cmap', 'Blues')
        self.normalize = viz_config.get('confusion_matrix', {}).get('normalize', 'true')
        
        # Currency class names
        self.class_names = ['Rp1K', 'Rp2K', 'Rp5K', 'Rp10K', 'Rp20K', 'Rp50K', 'Rp100K']
    
    def create_confusion_matrix_visualization(self, confusion_data: Dict[str, Any], 
                                            title: str = "Confusion Matrix", 
                                            save_path: Optional[str] = None) -> Optional[str]:
        """Create comprehensive confusion matrix visualization"""
        try:
            if 'matrix' not in confusion_data:
                self.logger.warning("⚠️ No confusion matrix data found")
                return None
            
            # Extract data
            matrix = np.array(confusion_data['matrix'])
            class_names = confusion_data.get('class_names', self.class_names[:matrix.shape[0]])
            
            # Create figure dengan multiple panels
            fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
            fig.suptitle(f'{title} - Comprehensive Analysis', fontsize=16, fontweight='bold', y=0.98)
            
            # Panel 1: Raw confusion matrix
            self._plot_raw_confusion_matrix(axes[0, 0], matrix, class_names)
            
            # Panel 2: Normalized confusion matrix
            self._plot_normalized_confusion_matrix(axes[0, 1], matrix, class_names)
            
            # Panel 3: Per-class performance
            self._plot_class_performance_metrics(axes[1, 0], matrix, class_names)
            
            # Panel 4: Error analysis
            self._plot_error_analysis(axes[1, 1], matrix, class_names)
            
            plt.tight_layout()
            
            # Save visualization
            if save_path is None:
                save_path = self.output_dir / f"{title.lower().replace(' ', '_')}_comprehensive.png"
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error creating confusion matrix visualization: {str(e)}")
            return None
    
    def _plot_raw_confusion_matrix(self, ax, matrix: np.ndarray, class_names: List[str]) -> None:
        """Plot raw confusion matrix dengan counts"""
        sns.heatmap(matrix, annot=True, fmt='d', cmap=self.cmap, ax=ax,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        
        ax.set_title('Raw Confusion Matrix (Counts)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        
        # Add accuracy on diagonal
        total_correct = np.trace(matrix)
        total_samples = np.sum(matrix)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        ax.text(0.02, 0.98, f'Overall Accuracy: {accuracy:.3f}', 
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def _plot_normalized_confusion_matrix(self, ax, matrix: np.ndarray, class_names: List[str]) -> None:
        """Plot normalized confusion matrix dengan percentages"""
        # Normalize by true class (rows)
        normalized_matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        normalized_matrix = np.nan_to_num(normalized_matrix)
        
        sns.heatmap(normalized_matrix, annot=True, fmt='.3f', cmap=self.cmap, ax=ax,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Recall (True Positive Rate)'},
                   vmin=0, vmax=1)
        
        ax.set_title('Normalized Confusion Matrix (Recall)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        
        # Highlight diagonal
        for i in range(len(class_names)):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))
    
    def _plot_class_performance_metrics(self, ax, matrix: np.ndarray, class_names: List[str]) -> None:
        """Plot per-class performance metrics"""
        # Calculate per-class metrics
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for i in range(len(class_names)):
            # True positives, false positives, false negatives
            tp = matrix[i, i]
            fp = np.sum(matrix[:, i]) - tp
            fn = np.sum(matrix[i, :]) - tp
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        # Create grouped bar chart
        x = np.arange(len(class_names))
        width = 0.25
        
        bars1 = ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x, recall_scores, width, label='Recall', alpha=0.8, color='lightgreen')
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='lightcoral')
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Classes', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Per-Class Performance Metrics', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_error_analysis(self, ax, matrix: np.ndarray, class_names: List[str]) -> None:
        """Plot error analysis dengan misclassification patterns"""
        # Create error matrix (off-diagonal elements)
        error_matrix = matrix.copy()
        np.fill_diagonal(error_matrix, 0)
        
        # Find top misclassifications
        error_pairs = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and error_matrix[i, j] > 0:
                    error_pairs.append((i, j, error_matrix[i, j]))
        
        # Sort by error count
        error_pairs.sort(key=lambda x: x[2], reverse=True)
        top_errors = error_pairs[:min(10, len(error_pairs))]
        
        if top_errors:
            # Create horizontal bar chart
            error_labels = [f'{class_names[true_idx]} → {class_names[pred_idx]}' 
                          for true_idx, pred_idx, _ in top_errors]
            error_counts = [count for _, _, count in top_errors]
            
            y_pos = np.arange(len(error_labels))
            bars = ax.barh(y_pos, error_counts, alpha=0.7, color='salmon')
            
            # Add value labels
            for i, (bar, count) in enumerate(zip(bars, error_counts)):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{count}', ha='left', va='center', fontsize=9, fontweight='bold')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(error_labels, fontsize=9)
            ax.set_xlabel('Misclassification Count', fontweight='bold')
            ax.set_title('Top Misclassification Patterns', fontweight='bold', fontsize=12)
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No misclassifications found\n(Perfect classification)', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Error Analysis', fontweight='bold', fontsize=12)
    
    def create_simple_confusion_matrix(self, confusion_data: Dict[str, Any], 
                                     title: str = "Confusion Matrix", 
                                     save_path: Optional[str] = None) -> Optional[str]:
        """Create simple confusion matrix visualization"""
        try:
            if 'matrix' not in confusion_data:
                return None
            
            matrix = np.array(confusion_data['matrix'])
            class_names = confusion_data.get('class_names', self.class_names[:matrix.shape[0]])
            
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Determine normalization
            if self.normalize == 'true':
                display_matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
                display_matrix = np.nan_to_num(display_matrix)
                fmt = '.3f'
                cbar_label = 'Recall (Normalized by True Class)'
            else:
                display_matrix = matrix
                fmt = 'd'
                cbar_label = 'Count'
            
            # Create heatmap
            sns.heatmap(display_matrix, annot=True, fmt=fmt, cmap=self.cmap, ax=ax,
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': cbar_label}, square=True)
            
            # Customize
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
            ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
            
            # Add accuracy annotation
            if self.normalize != 'true':
                total_correct = np.trace(matrix)
                total_samples = np.sum(matrix)
                accuracy = total_correct / total_samples if total_samples > 0 else 0
                ax.text(0.02, 0.98, f'Accuracy: {accuracy:.3f}', 
                       transform=ax.transAxes, fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            
            # Save
            if save_path is None:
                save_path = self.output_dir / f"{title.lower().replace(' ', '_')}_simple.png"
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error creating simple confusion matrix: {str(e)}")
            return None
    
    def create_class_specific_analysis(self, confusion_data: Dict[str, Any], 
                                     target_class: str, 
                                     title: str = None, 
                                     save_path: Optional[str] = None) -> Optional[str]:
        """Create class-specific detailed analysis"""
        try:
            if 'matrix' not in confusion_data:
                return None
            
            matrix = np.array(confusion_data['matrix'])
            class_names = confusion_data.get('class_names', self.class_names[:matrix.shape[0]])
            
            if target_class not in class_names:
                self.logger.warning(f"⚠️ Class {target_class} not found in class names")
                return None
            
            class_idx = class_names.index(target_class)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
            if title is None:
                title = f"Class Analysis: {target_class}"
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
            
            # Panel 1: Class confusion focus
            self._plot_class_confusion_focus(axes[0, 0], matrix, class_names, class_idx, target_class)
            
            # Panel 2: Class performance radar
            self._plot_class_performance_radar(axes[0, 1], matrix, class_names, class_idx, target_class)
            
            # Panel 3: Prediction distribution
            self._plot_class_prediction_distribution(axes[1, 0], matrix, class_names, class_idx, target_class)
            
            # Panel 4: Error sources
            self._plot_class_error_sources(axes[1, 1], matrix, class_names, class_idx, target_class)
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = self.output_dir / f"class_analysis_{target_class.lower()}.png"
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error creating class-specific analysis: {str(e)}")
            return None
    
    def _plot_class_confusion_focus(self, ax, matrix: np.ndarray, class_names: List[str], 
                                  class_idx: int, target_class: str) -> None:
        """Plot confusion matrix dengan focus pada specific class"""
        # Highlight target class row and column
        highlighted_matrix = matrix.astype(float)
        
        # Create mask untuk highlighting
        mask = np.zeros_like(highlighted_matrix, dtype=bool)
        mask[class_idx, :] = True  # True class row
        mask[:, class_idx] = True  # Predicted class column
        
        sns.heatmap(highlighted_matrix, annot=True, fmt='d', cmap=self.cmap, ax=ax,
                   xticklabels=class_names, yticklabels=class_names,
                   mask=~mask, alpha=0.3)
        
        # Overlay highlighted cells
        sns.heatmap(highlighted_matrix, annot=True, fmt='d', cmap='Reds', ax=ax,
                   xticklabels=class_names, yticklabels=class_names,
                   mask=mask, alpha=0.8, cbar=False)
        
        ax.set_title(f'Confusion Focus: {target_class}', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
    
    def _plot_class_performance_radar(self, ax, matrix: np.ndarray, class_names: List[str], 
                                    class_idx: int, target_class: str) -> None:
        """Plot performance radar untuk specific class"""
        # Calculate metrics
        tp = matrix[class_idx, class_idx]
        fp = np.sum(matrix[:, class_idx]) - tp
        fn = np.sum(matrix[class_idx, :]) - tp
        tn = np.sum(matrix) - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Radar chart
        metrics = ['Precision', 'Recall', 'Specificity', 'F1-Score']
        values = [precision, recall, specificity, f1]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='red', markersize=8)
        ax.fill(angles, values, alpha=0.25, color='red')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(f'{target_class} Performance Metrics', fontweight='bold')
        ax.grid(True)
        
        # Add metric values as text
        for angle, value, metric in zip(angles[:-1], values[:-1], metrics):
            ax.text(angle, value + 0.1, f'{value:.3f}', ha='center', va='center',
                   fontsize=8, fontweight='bold')
    
    def _plot_class_prediction_distribution(self, ax, matrix: np.ndarray, class_names: List[str], 
                                          class_idx: int, target_class: str) -> None:
        """Plot prediction distribution untuk true class"""
        # Get row for true class
        true_class_predictions = matrix[class_idx, :]
        
        colors = ['red' if i == class_idx else 'lightblue' for i in range(len(class_names))]
        bars = ax.bar(class_names, true_class_predictions, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, count in zip(bars, true_class_predictions):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{count}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'Predictions for True {target_class}', fontweight='bold')
        ax.set_xlabel('Predicted Class', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Add total annotation
        total = np.sum(true_class_predictions)
        correct = true_class_predictions[class_idx]
        accuracy = correct / total if total > 0 else 0
        ax.text(0.02, 0.98, f'Class Recall: {accuracy:.3f}', 
               transform=ax.transAxes, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def _plot_class_error_sources(self, ax, matrix: np.ndarray, class_names: List[str], 
                                class_idx: int, target_class: str) -> None:
        """Plot error sources (what gets misclassified as target class)"""
        # Get column for predicted class (false positives)
        predicted_as_target = matrix[:, class_idx]
        
        # Remove true positives
        error_sources = predicted_as_target.copy()
        error_sources[class_idx] = 0
        
        if np.sum(error_sources) > 0:
            # Create pie chart of error sources
            non_zero_indices = error_sources > 0
            error_labels = [class_names[i] for i in range(len(class_names)) if error_sources[i] > 0]
            error_values = error_sources[non_zero_indices]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(error_labels)))
            wedges, texts, autotexts = ax.pie(error_values, labels=error_labels, autopct='%1.1f%%',
                                            colors=colors, startangle=90)
            
            ax.set_title(f'False Positives: What gets\nmisclassified as {target_class}', fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'No False Positives\nfor {target_class}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    
    def create_matrix_comparison(self, matrices_data: Dict[str, Dict[str, Any]], 
                               title: str = "Confusion Matrix Comparison", 
                               save_path: Optional[str] = None) -> Optional[str]:
        """Create side-by-side comparison of multiple confusion matrices"""
        try:
            if not matrices_data:
                return None
            
            n_matrices = len(matrices_data)
            cols = min(3, n_matrices)
            rows = (n_matrices + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), dpi=self.dpi)
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
            
            if n_matrices == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if n_matrices > 1 else [axes]
            else:
                axes = axes.flatten()
            
            for i, (name, matrix_data) in enumerate(matrices_data.items()):
                if i >= len(axes):
                    break
                
                matrix = np.array(matrix_data['matrix'])
                class_names = matrix_data.get('class_names', self.class_names[:matrix.shape[0]])
                
                # Normalize matrix
                normalized_matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
                normalized_matrix = np.nan_to_num(normalized_matrix)
                
                sns.heatmap(normalized_matrix, annot=True, fmt='.2f', cmap=self.cmap, ax=axes[i],
                           xticklabels=class_names, yticklabels=class_names,
                           cbar_kws={'shrink': 0.8}, square=True)
                
                axes[i].set_title(name, fontweight='bold')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('True')
                
                # Add accuracy
                accuracy = np.trace(matrix) / np.sum(matrix) if np.sum(matrix) > 0 else 0
                axes[i].text(0.02, 0.98, f'Acc: {accuracy:.3f}', 
                           transform=axes[i].transAxes, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for i in range(len(matrices_data), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = self.output_dir / f"{title.lower().replace(' ', '_')}_comparison.png"
            
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error creating matrix comparison: {str(e)}")
            return None
    
    def cleanup_visualizations(self, keep_latest: int = 15) -> None:
        """Cleanup old visualization files"""
        try:
            viz_files = list(self.output_dir.glob('*.png'))
            viz_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for old_file in viz_files[keep_latest:]:
                old_file.unlink()
            
            self.logger.info(f"🧹 Confusion matrix cleanup: kept {min(len(viz_files), keep_latest)} latest files")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error during visualization cleanup: {str(e)}")