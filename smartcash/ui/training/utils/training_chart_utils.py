"""
File: smartcash/ui/training/utils/training_chart_utils.py
Deskripsi: Chart utilities untuk visualisasi training metrics dengan matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
from IPython.display import display, clear_output


# Global metrics storage untuk chart persistence
_training_metrics = {'epochs': [], 'train_loss': [], 'val_loss': [], 'map': [], 'f1': []}


def update_training_chart(ui_components: Dict[str, Any], metrics: Dict[str, float], epoch: int, total_epochs: int):
    """Update training chart dengan metrics baru"""
    chart_output = ui_components.get('chart_output')
    if not chart_output:
        return
    
    # Update global metrics
    _training_metrics['epochs'].append(epoch)
    _training_metrics['train_loss'].append(metrics.get('train_loss', 0))
    _training_metrics['val_loss'].append(metrics.get('val_loss', 0))
    _training_metrics['map'].append(metrics.get('map', 0))
    _training_metrics['f1'].append(metrics.get('f1', 0))
    
    # Generate chart
    _generate_training_chart(chart_output, total_epochs)


def initialize_empty_training_chart(ui_components: Dict[str, Any]):
    """Initialize empty training chart"""
    chart_output = ui_components.get('chart_output')
    if not chart_output:
        return
    
    # Reset global metrics
    for key in _training_metrics:
        _training_metrics[key] = []
    
    # Generate empty chart
    _generate_empty_chart(chart_output)


def _generate_training_chart(chart_output, total_epochs: int):
    """Generate training chart dengan matplotlib"""
    with chart_output:
        clear_output(wait=True)
        
        if not _training_metrics['epochs']:
            _generate_empty_chart(chart_output)
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('🧠 EfficientNet-B4 Training Progress', fontsize=14, fontweight='bold')
        
        epochs = _training_metrics['epochs']
        
        # Loss chart
        ax1.plot(epochs, _training_metrics['train_loss'], 'b-', label='Train Loss', marker='o', markersize=4)
        ax1.plot(epochs, _training_metrics['val_loss'], 'r-', label='Val Loss', marker='s', markersize=4)
        ax1.set_title('📉 Loss Progress')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # mAP chart
        ax2.plot(epochs, _training_metrics['map'], 'g-', label='mAP', marker='^', markersize=4)
        ax2.set_title('📊 mAP Progress')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # F1 Score chart
        ax3.plot(epochs, _training_metrics['f1'], 'm-', label='F1 Score', marker='d', markersize=4)
        ax3.set_title('🎯 F1 Score Progress')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Combined metrics
        ax4.plot(epochs, _training_metrics['map'], 'g-', label='mAP', marker='^', markersize=3)
        ax4.plot(epochs, _training_metrics['f1'], 'm-', label='F1', marker='d', markersize=3)
        ax4.set_title('📈 Combined Metrics')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # Set x-axis limits
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim(0, max(total_epochs, max(epochs) + 5) if epochs else total_epochs)
        
        plt.tight_layout()
        plt.show()


def _generate_empty_chart(chart_output):
    """Generate empty chart placeholder"""
    with chart_output:
        clear_output(wait=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Empty chart dengan styling
        ax.text(0.5, 0.5, '📊 Training Chart\n\nChart akan muncul saat training dimulai...', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metrics')
        ax.set_title('🧠 EfficientNet-B4 Training Metrics')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def save_training_chart(ui_components: Dict[str, Any], filename: str = "training_chart.png"):
    """Save current training chart"""
    if not _training_metrics['epochs']:
        return None
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('EfficientNet-B4 Training Results', fontsize=14, fontweight='bold')
        
        epochs = _training_metrics['epochs']
        
        # Loss chart
        ax1.plot(epochs, _training_metrics['train_loss'], 'b-', label='Train Loss', marker='o')
        ax1.plot(epochs, _training_metrics['val_loss'], 'r-', label='Val Loss', marker='s')
        ax1.set_title('Loss Progress')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # mAP chart
        ax2.plot(epochs, _training_metrics['map'], 'g-', label='mAP', marker='^')
        ax2.set_title('mAP Progress')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # F1 Score chart
        ax3.plot(epochs, _training_metrics['f1'], 'm-', label='F1 Score', marker='d')
        ax3.set_title('F1 Score Progress')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Combined metrics
        ax4.plot(epochs, _training_metrics['map'], 'g-', label='mAP', marker='^')
        ax4.plot(epochs, _training_metrics['f1'], 'm-', label='F1', marker='d')
        ax4.set_title('Combined Metrics')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
        
    except Exception as e:
        print(f"⚠️ Error saving chart: {str(e)}")
        return None


def get_training_summary() -> Dict[str, Any]:
    """Get training metrics summary"""
    if not _training_metrics['epochs']:
        return {'status': 'no_data'}
    
    epochs = _training_metrics['epochs']
    final_epoch = max(epochs)
    final_idx = epochs.index(final_epoch)
    
    return {
        'status': 'completed',
        'total_epochs': len(epochs),
        'final_epoch': final_epoch,
        'final_metrics': {
            'train_loss': _training_metrics['train_loss'][final_idx],
            'val_loss': _training_metrics['val_loss'][final_idx],
            'map': _training_metrics['map'][final_idx],
            'f1': _training_metrics['f1'][final_idx]
        },
        'best_metrics': {
            'best_map': max(_training_metrics['map']),
            'best_f1': max(_training_metrics['f1']),
            'lowest_loss': min(_training_metrics['train_loss'])
        }
    }


# One-liner utilities
clear_metrics = lambda: [_training_metrics.update({k: []}) for k in _training_metrics.keys()]
has_metrics = lambda: len(_training_metrics['epochs']) > 0
get_current_epoch = lambda: max(_training_metrics['epochs']) if _training_metrics['epochs'] else 0
get_latest_loss = lambda: _training_metrics['train_loss'][-1] if _training_metrics['train_loss'] else 0
get_latest_map = lambda: _training_metrics['map'][-1] if _training_metrics['map'] else 0