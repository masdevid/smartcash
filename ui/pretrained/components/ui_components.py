"""
File: smartcash/ui/pretrained/components/ui_components.py
Deskripsi: UI components untuk pretrained module dengan correct imports
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_pretrained_main_ui(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create main UI untuk pretrained module dengan existing component imports"""
    try:
        # Correct imports dari existing components
        from smartcash.ui.components.progress_tracker.factory import create_dual_progress_tracker
        from smartcash.ui.components.status_panel import create_status_panel
        from smartcash.ui.components.action_buttons import create_action_buttons
        
        # Extract config values
        pretrained_config = config.get('pretrained_models', {})
        models_config = pretrained_config.get('models', {})
        yolov5_config = models_config.get('yolov5s', {})
        efficientnet_config = models_config.get('efficientnet_b4', {})
        
        # Header
        header = widgets.HTML(
            value="<h3>🔽 <b>Setup Pretrained Models</b></h3>",
            layout=widgets.Layout(margin='0 0 15px 0')
        )
        
        # Model URL inputs
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
        
        # Input form container
        input_form = widgets.VBox([
            widgets.HTML("<p><b>📝 Model URLs Configuration:</b></p>"),
            yolov5_url_input,
            efficientnet_url_input,
            widgets.HTML("<p><small>💡 Models akan disimpan di: <code>/data/pretrained</code></small></p>")
        ], layout=widgets.Layout(margin='10px 0'))
        
        # Progress tracker
        progress_tracker = create_dual_progress_tracker(
            operation="Pretrained Models Setup",
            auto_hide=True
        )
        
        # Action buttons (using existing create_action_buttons)
        action_buttons = create_action_buttons(
            primary_label="🔽 Check, Download & Sync Models",
            primary_icon="download",
            secondary_buttons=[],  # No secondary button
            cleanup_enabled=False,  # No cleanup needed
            button_width='250px'
        )
        download_sync_button = action_buttons['download_button']
        
        # Status panel
        status_panel = create_status_panel()
        
        # Log output
        log_output = widgets.Output(
            layout=widgets.Layout(
                height='200px',
                border='1px solid #ddd',
                padding='10px',
                margin='10px 0'
            )
        )
        
        # Simple confirmation area (no complex dialog needed)
        confirmation_area = widgets.Output(
            layout=widgets.Layout(
                width='100%',
                min_height='0px',
                margin='10px 0',
                padding='5px',
                border='1px solid #e0e0e0',
                border_radius='4px',
                visibility='hidden'
            )
        )
        
        # Main layout
        main_layout = widgets.VBox([
            header,
            input_form,
            action_buttons['container'],
            progress_tracker.container,
            status_panel,
            log_output,
            confirmation_area
        ], layout=widgets.Layout(
            padding='20px',
            border='1px solid #e1e4e8',
            border_radius='6px',
            background_color='#f6f8fa'
        ))
        
        return {
            'ui': main_layout,
            'yolov5_url_input': yolov5_url_input,
            'efficientnet_url_input': efficientnet_url_input,
            'input_form': input_form,
            'download_sync_button': download_sync_button,
            'progress_tracker': progress_tracker,
            'progress_container': progress_tracker.container,
            'status_panel': status_panel,
            'log_output': log_output,
            'confirmation_area': confirmation_area,
            'header': header,
            'main_layout': main_layout
        }
        
    except Exception as e:
        # Fallback minimal UI
        error_ui = widgets.VBox([
            widgets.HTML(f"<h3>❌ Error creating UI: {str(e)}</h3>"),
            widgets.HTML("<p>Please check imports and try again.</p>")
        ])
        
        return {
            'ui': error_ui,
            'error': True,
            'error_message': str(e)
        }