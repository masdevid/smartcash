"""
File: smartcash/ui/training/utils/training_display_utils.py
Deskripsi: Utilities untuk training display dan UI updates
"""

import threading
from typing import Dict, Any
from IPython.display import display, HTML
from smartcash.ui.training.handlers.training_button_handlers import get_state, set_state
from smartcash.ui.training.utils.training_status_utils import update_training_status


def update_training_info(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Update training info dengan model information"""
    training_config = config.get('training', {})
    model_type = training_config.get('model_type', 'efficient_optimized')
    
    # Get model description
    model_descriptions = {
        'efficient_basic': 'EfficientNet-B4 Basic',
        'efficient_optimized': 'EfficientNet-B4 + FeatureAdapter',
        'efficient_advanced': 'EfficientNet-B4 + FeatureAdapter + ResidualAdapter + CIoU'
    }
    
    model_desc = model_descriptions.get(model_type, 'EfficientNet-B4 Custom')
    
    info_html = f"""
    <div style="padding: 12px; background-color: #f8f9fa; border-radius: 6px;">
        <h5>🧠 Model Configuration</h5>
        <ul style="margin: 10px 0;">
            <li><b>Model Type:</b> {model_desc}</li>
            <li><b>Backbone:</b> {training_config.get('backbone', 'efficientnet_b4')}</li>
            <li><b>Detection Layers:</b> {', '.join(training_config.get('detection_layers', ['banknote']))}</li>
            <li><b>Epochs:</b> {training_config.get('epochs', 100)}</li>
            <li><b>Batch Size:</b> {training_config.get('batch_size', 16)}</li>
            <li><b>Learning Rate:</b> {training_config.get('learning_rate', 0.001)}</li>
            <li><b>Image Size:</b> {training_config.get('image_size', 640)}</li>
            <li><b>Optimizer:</b> {training_config.get('optimizer', 'Adam')}</li>
        </ul>
        <div style="background: #e3f2fd; padding: 8px; border-radius: 4px; margin-top: 10px;">
            <b>🎯 Optimizations:</b>
            <span style="color: #1976d2;">
                FeatureAdapter: {config.get('model_optimization', {}).get('use_attention', True)} |
                ResidualAdapter: {config.get('model_optimization', {}).get('use_residual', True)} |
                CIoU Loss: {config.get('model_optimization', {}).get('use_ciou', True)}
            </span>
        </div>
    </div>
    """
    
    training_config_display = ui_components.get('training_config_display')
    if training_config_display:
        with training_config_display:
            training_config_display.clear_output(wait=True)
            display(HTML(info_html))


def prepare_model_background(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Prepare model secara background untuk prevent UI blocking"""
    def build_model():
        try:
            model_manager = ui_components.get('model_manager')
            if model_manager and not get_state().get('model_ready', False):
                update_training_status(ui_components, "🔧 Membangun model EfficientNet-B4...", 'info')
                model_manager.build_model()
                set_state(model_ready=True)
                update_training_status(ui_components, "✅ Model siap untuk training", 'success')
        except Exception as e:
            update_training_status(ui_components, f"❌ Error building model: {str(e)}", 'error')
    
    # Build model dalam background thread
    threading.Thread(target=build_model, daemon=True).start()


def display_validation_results(display_widget, results: Dict[str, Any]):
    """Display hasil model readiness validation"""
    model_status = "✅" if results['model_status']['model_ready'] else "❌"
    models_status = "✅" if results['pretrained_models']['available_models'] else "⚠️"
    config_status = "✅" if results['training_config']['valid'] else "❌"
    gpu_status = "✅" if results['gpu_status']['cuda_available'] else "❌"
    
    html_content = f"""
    <div style="padding: 12px;">
        <h4>🔍 Model Readiness Validation</h4>
        
        <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 6px;">
            <h5>{model_status} Model Status</h5>
            <ul>
                <li><b>Model Manager:</b> {'Available' if results['model_status']['model_manager_available'] else 'Missing'}</li>
                <li><b>Training Service:</b> {'Available' if results['model_status']['training_service_available'] else 'Missing'}</li>
                <li><b>Model Built:</b> {'Yes' if results['model_status']['model_built'] else 'No'}</li>
                <li><b>Model Type:</b> {results['model_status']['model_type'] or 'N/A'}</li>
                <li><b>Ready for Training:</b> {'Yes' if results['model_status']['model_ready'] else 'No'}</li>
            </ul>
        </div>
        
        <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 6px;">
            <h5>{models_status} Pre-trained Models</h5>
            <ul>
                <li><b>Drive Mounted:</b> {'Yes' if results['pretrained_models']['drive_mounted'] else 'No'}</li>
                <li><b>Available Models:</b> {len(results['pretrained_models']['available_models'])}/3</li>
                <li><b>Missing Models:</b> {', '.join(results['pretrained_models']['missing_models']) or 'None'}</li>
                <li><b>Total Size:</b> {format_file_size(results['pretrained_models']['total_size'])}</li>
            </ul>
        </div>
        
        <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 6px;">
            <h5>{config_status} Training Configuration</h5>
            <ul>
                <li><b>Model Type:</b> {results['training_config']['model_type']}</li>
                <li><b>Backbone:</b> {results['training_config']['backbone']}</li>
                <li><b>Epochs:</b> {results['training_config']['epochs']}</li>
                <li><b>Batch Size:</b> {results['training_config']['batch_size']}</li>
                <li><b>Learning Rate:</b> {results['training_config']['learning_rate']}</li>
                <li><b>Optimizations:</b> Attention({results['training_config']['optimizations']['use_attention']}), 
                    Residual({results['training_config']['optimizations']['use_residual']}), 
                    CIoU({results['training_config']['optimizations']['use_ciou']})</li>
            </ul>
        </div>
        
        <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 6px;">
            <h5>{gpu_status} GPU Status</h5>
            <ul>
                <li><b>CUDA Available:</b> {'Yes' if results['gpu_status']['cuda_available'] else 'No'}</li>
                <li><b>Device Name:</b> {results['gpu_status'].get('device_name', 'N/A')}</li>
                <li><b>Memory Allocated:</b> {results['gpu_status']['memory_info'].get('allocated', 'N/A')}</li>
                <li><b>Memory Reserved:</b> {results['gpu_status']['memory_info'].get('reserved', 'N/A')}</li>
            </ul>
        </div>
        
        <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 6px;">
            <h5>🎯 Detection Layers</h5>
            <ul>
                <li><b>Configured:</b> {', '.join(results['detection_layers']['configured_layers'])}</li>
                <li><b>Valid Layers:</b> {', '.join(results['detection_layers']['valid_layers'])}</li>
                <li><b>Total Classes:</b> {results['detection_layers']['total_classes']}</li>
            </ul>
        </div>
    </div>
    """
    
    with display_widget:
        display_widget.clear_output(wait=True)
        display(HTML(html_content))


def display_cleanup_results(display_widget, results: Dict[str, Any]):
    """Display hasil GPU cleanup"""
    memory_before = results['gpu_memory_before']
    memory_after = results['gpu_memory_after']
    
    html_content = f"""
    <div style="padding: 12px;">
        <h4>🧹 Hasil GPU Cleanup</h4>
        
        <div style="margin: 10px 0; padding: 10px; background: #f8fff8; border-radius: 6px;">
            <h5>📊 Memory Status</h5>
            <ul>
                <li><b>Before Cleanup:</b> {memory_before.get('allocated', 'N/A')} allocated, {memory_before.get('reserved', 'N/A')} reserved</li>
                <li><b>After Cleanup:</b> {memory_after.get('allocated', 'N/A')} allocated, {memory_after.get('reserved', 'N/A')} reserved</li>
                <li><b>Memory Freed:</b> {calculate_memory_freed(results)}</li>
            </ul>
        </div>
        
        <div style="margin: 10px 0; padding: 10px; background: #f8fff8; border-radius: 6px;">
            <h5>🗑️ Resources Cleaned</h5>
            <ul>
                <li><b>Models Cleaned:</b> {', '.join(results['model_cleanup']['models_cleaned']) or 'None'}</li>
                <li><b>Cache Methods:</b> {', '.join(results['cache_cleanup']['methods_used'])}</li>
                <li><b>UI Outputs:</b> {', '.join(results['ui_cleanup']['outputs_cleared'])}</li>
            </ul>
        </div>
        
        <div style="margin: 10px 0; padding: 10px; background: #f0f8ff; border-radius: 6px;">
            <h5>⚡ Performance Impact</h5>
            <ul>
                <li><b>Cache Cleared:</b> {'Yes' if results['cache_cleanup']['cache_cleared'] else 'No'}</li>
                <li><b>Models Freed:</b> {len(results['model_cleanup']['models_cleaned'])}</li>
                <li><b>Errors:</b> {len(results['model_cleanup']['errors']) + len(results['ui_cleanup']['errors'])}</li>
            </ul>
        </div>
    </div>
    """
    
    with display_widget:
        display_widget.clear_output(wait=True)
        display(HTML(html_content))


def format_file_size(size_bytes: int) -> str:
    """Format file size dalam human readable format"""
    if size_bytes == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def calculate_memory_freed(results: Dict[str, Any]) -> str:
    """Calculate memory freed dari cleanup"""
    from smartcash.ui.training.utils.cleanup_utils import calculate_memory_freed as calc_freed
    return calc_freed(results)