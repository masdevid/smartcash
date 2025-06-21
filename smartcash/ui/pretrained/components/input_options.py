"""
File: smartcash/ui/pretrained/components/input_options.py
Deskripsi: Input options untuk pretrained module mengikuti pattern preprocessing
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_pretrained_input_options(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create input options untuk pretrained module dengan pattern yang konsisten"""
    config = config or {}
    
    try:
        # Extract config values
        pretrained_config = config.get('pretrained_models', {})
        models_config = pretrained_config.get('models', {})
        yolov5_config = models_config.get('yolov5s', {})
        efficientnet_config = models_config.get('efficientnet_b4', {})
        
        # === MODEL URL INPUTS ===
        
        yolov5_url_input = widgets.Text(
            value=yolov5_config.get('url', 'https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt'),
            placeholder='YOLOv5s model URL dari Ultralytics',
            description='YOLOv5s URL:',
            layout=widgets.Layout(width='100%', margin='5px 0'),
            style={'description_width': '120px'}
        )
        
        efficientnet_url_input = widgets.Text(
            value=efficientnet_config.get('url', 'https://huggingface.co/timm/efficientnet_b4.ra2_in1k/resolve/main/pytorch_model.bin'),
            placeholder='EfficientNet-B4 model URL dari TIMM',
            description='EfficientNet URL:',
            layout=widgets.Layout(width='100%', margin='5px 0'),
            style={'description_width': '120px'}
        )
        
        # === DIRECTORY INPUTS ===
        
        models_dir_input = widgets.Text(
            value=pretrained_config.get('models_dir', '/content/models'),
            placeholder='Directory untuk menyimpan models',
            description='Models Dir:',
            layout=widgets.Layout(width='100%', margin='5px 0'),
            style={'description_width': '120px'}
        )
        
        drive_models_dir_input = widgets.Text(
            value=pretrained_config.get('drive_models_dir', '/content/drive/MyDrive/SmartCash/models'),
            placeholder='Directory di Google Drive untuk backup',
            description='Drive Dir:',
            layout=widgets.Layout(width='100%', margin='5px 0'),
            style={'description_width': '120px'}
        )
        
        # === MODEL INFO DISPLAY ===
        
        model_info_html = widgets.HTML(
            value=f"""
            <div style='padding:10px; background-color:#f8f9fa; border-radius:5px; margin:10px 0;'>
                <p><b>📦 Models yang akan disetup:</b></p>
                <ul style='margin:5px 0; padding-left:20px;'>
                    <li><b>YOLOv5s:</b> Object detection backbone (~14MB)</li>
                    <li><b>EfficientNet-B4:</b> Feature extraction backbone (~70MB)</li>
                </ul>
                <p><small>💡 Models akan disimpan lokal dan disync ke Drive</small></p>
            </div>
            """
        )
        
        # === FORM SECTIONS ===
        
        # URL Configuration section
        url_section = widgets.VBox([
            widgets.HTML("<h4>🔗 Model URLs</h4>"),
            yolov5_url_input,
            efficientnet_url_input
        ], layout=widgets.Layout(width='100%', margin='10px 0'))
        
        # Directory Configuration section  
        dir_section = widgets.VBox([
            widgets.HTML("<h4>📁 Storage Directories</h4>"),
            models_dir_input,
            drive_models_dir_input
        ], layout=widgets.Layout(width='100%', margin='10px 0'))
        
        # === MAIN CONTAINER ===
        
        main_container = widgets.VBox([
            model_info_html,
            url_section,
            dir_section
        ], layout=widgets.Layout(width='100%', padding='10px'))
        
        # === RETURN COMPONENTS ===
        
        return {
            # Main container
            'main_container': main_container,
            
            # Individual inputs
            'yolov5_url_input': yolov5_url_input,
            'efficientnet_url_input': efficientnet_url_input,
            'models_dir_input': models_dir_input,
            'drive_models_dir_input': drive_models_dir_input,
            
            # Sections
            'url_section': url_section,
            'dir_section': dir_section,
            'model_info_html': model_info_html,
            
            # Metadata
            'input_type': 'pretrained_models',
            'components_count': 4
        }
        
    except Exception as e:
        # Fallback minimal UI
        error_html = widgets.HTML(f"""
            <div style='padding:15px; border:2px solid #dc3545; border-radius:5px; background:#f8d7da; color:#721c24;'>
                <h4>⚠️ Input Options Error</h4>
                <p>{str(e)}</p>
                <small>💡 Check component dependencies</small>
            </div>
        """)
        
        return {
            'main_container': error_html,
            'yolov5_url_input': widgets.Text(description="Error", disabled=True),
            'efficientnet_url_input': widgets.Text(description="Error", disabled=True),
            'models_dir_input': widgets.Text(description="Error", disabled=True),
            'drive_models_dir_input': widgets.Text(description="Error", disabled=True),
            'error': str(e)
        }