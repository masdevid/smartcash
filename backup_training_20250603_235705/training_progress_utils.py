"""
File: smartcash/ui/training/utils/training_progress_utils.py
Deskripsi: Utilities untuk update progress training dan visualisasi
"""

import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from IPython.display import display, HTML, clear_output

def update_training_progress(ui_components: Dict[str, Any], progress_type: str, current: int, total: int, message: str) -> None:
    """Update progress training dengan tipe yang spesifik"""
    try:
        # Update progress tracker jika ada
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'update'):
            progress_tracker.update(progress_type, int((current / max(total, 1)) * 100), message)
            
    except Exception:
        # Silent fail untuk progress updates
        pass

def update_model_loading_progress(ui_components: Dict[str, Any], current: int, total: int, message: str) -> None:
    """Update progress untuk model loading"""
    update_training_progress(ui_components, 'current', current, total, message)

def update_chart_display(chart_output, chart_data: Dict[str, list]) -> None:
    """Update chart display dengan data training"""
    if not chart_output or not chart_data.get('epochs'):
        return
        
    try:
        with chart_output:
            clear_output(wait=True)
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            epochs = chart_data['epochs']
            
            # Loss plot
            if chart_data.get('train_loss'):
                ax1.plot(epochs, chart_data['train_loss'], 'b-', label='Train Loss', linewidth=2)
            if chart_data.get('val_loss'):
                ax1.plot(epochs, chart_data['val_loss'], 'r-', label='Val Loss', linewidth=2)
                
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training & Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Learning rate plot
            if chart_data.get('learning_rate'):
                ax2.plot(epochs, chart_data['learning_rate'], 'g-', label='Learning Rate', linewidth=2)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Learning Rate')
                ax2.set_title('Learning Rate Schedule')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No Learning Rate Data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Learning Rate Schedule')
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        with chart_output:
            clear_output(wait=True)
            display(HTML(f"<div style='color: red;'>❌ Error updating chart: {str(e)}</div>"))

def update_metrics_display(metrics_output, metrics: Dict[str, float]) -> None:
    """Update metrics display dengan format yang rapi"""
    if not metrics_output or not metrics:
        return
        
    try:
        with metrics_output:
            clear_output(wait=True)
            
            # Format metrics dalam bentuk card
            metrics_html = """
            <div style="display: flex; flex-wrap: wrap; gap: 10px; padding: 5px;">
            """
            
            for key, value in metrics.items():
                # Tentukan warna berdasarkan tipe metric
                if 'loss' in key.lower():
                    color = '#dc3545'  # Red untuk loss
                    icon = '📉'
                elif any(term in key.lower() for term in ['accuracy', 'precision', 'recall', 'map', 'f1']):
                    color = '#28a745'  # Green untuk accuracy metrics
                    icon = '📈'
                elif 'learning_rate' in key.lower() or 'lr' in key.lower():
                    color = '#17a2b8'  # Blue untuk learning rate
                    icon = '📊'
                else:
                    color = '#6c757d'  # Gray untuk lainnya
                    icon = '📋'
                
                # Format nilai
                if isinstance(value, float):
                    if value < 0.01:
                        formatted_value = f"{value:.6f}"
                    elif value < 1:
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                metrics_html += f"""
                <div style="flex: 1; min-width: 120px; padding: 8px; 
                           background: rgba({','.join(map(str, _hex_to_rgb(color)))}, 0.1); 
                           border: 1px solid {color}; border-radius: 4px;">
                    <div style="font-size: 12px; color: #555; margin-bottom: 2px;">
                        {icon} {key.replace('_', ' ').title()}
                    </div>
                    <div style="font-size: 16px; font-weight: 600; color: {color};">
                        {formatted_value}
                    </div>
                </div>
                """
            
            metrics_html += "</div>"
            display(HTML(metrics_html))
            
    except Exception as e:
        with metrics_output:
            clear_output(wait=True)
            display(HTML(f"<div style='color: red;'>❌ Error updating metrics: {str(e)}</div>"))

def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color ke RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def show_training_progress_container(ui_components: Dict[str, Any], show: bool = True) -> None:
    """Show atau hide progress container"""
    progress_container = ui_components.get('progress_container')
    if progress_container and hasattr(progress_container, 'layout'):
        progress_container.layout.display = 'flex' if show else 'none'

def reset_training_displays(ui_components: Dict[str, Any]) -> None:
    """Reset semua training displays"""
    # Clear chart
    chart_output = ui_components.get('chart_output')
    chart_output and chart_output.clear_output(wait=True)
    
    # Clear metrics
    metrics_output = ui_components.get('metrics_output')
    metrics_output and metrics_output.clear_output(wait=True)
    
    # Clear log
    log_output = ui_components.get('log_output')
    log_output and log_output.clear_output(wait=True)
    
    # Hide progress
    show_training_progress_container(ui_components, False)