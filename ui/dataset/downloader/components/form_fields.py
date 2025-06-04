"""
File: smartcash/ui/dataset/downloader/components/form_fields.py
Deskripsi: Form fields components dengan responsive layout dan auto-detection
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.layout_utils import create_responsive_container

def create_form_fields(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create form fields dengan responsive two-panel layout."""
    
    # Left panel: Dataset Information
    left_components = _create_dataset_info_panel(config)
    
    # Right panel: Download Options
    right_components = _create_options_panel(config)
    
    return {
        'left_panel': left_components['container'],
        'right_panel': right_components['container'],
        **left_components,
        **right_components
    }

def _create_dataset_info_panel(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create dataset information panel."""
    
    # Section header
    header = widgets.HTML(
        "<h4 style='margin:5px 0 10px 0; color:#495057; border-bottom:2px solid #007bff; padding-bottom:5px;'>📊 Dataset Information</h4>"
    )
    
    # Workspace field
    workspace_field = widgets.Text(
        value=config.get('workspace', 'smartcash-wo2us'),
        description='Workspace:',
        placeholder='Roboflow workspace name',
        layout=widgets.Layout(width='100%', margin='3px 0'),
        style={'description_width': '80px'}
    )
    
    # Project field
    project_field = widgets.Text(
        value=config.get('project', 'rupiah-emisi-2022'),
        description='Project:',
        placeholder='Roboflow project name',
        layout=widgets.Layout(width='100%', margin='3px 0'),
        style={'description_width': '80px'}
    )
    
    # Version field
    version_field = widgets.Text(
        value=config.get('version', '3'),
        description='Version:',
        placeholder='Dataset version',
        layout=widgets.Layout(width='100%', margin='3px 0'),
        style={'description_width': '80px'}
    )
    
    # API key field dengan auto-detection
    api_key_field = widgets.Password(
        value=config.get('api_key', ''),
        description='API Key:',
        placeholder='Roboflow API key (auto-detected)',
        layout=widgets.Layout(width='100%', margin='3px 0'),
        style={'description_width': '80px'}
    )
    
    # Format dropdown
    format_dropdown = widgets.Dropdown(
        options=['yolov5pytorch', 'yolov8', 'coco', 'createml'],
        value=config.get('output_format', 'yolov5pytorch'),
        description='Format:',
        layout=widgets.Layout(width='100%', margin='3px 0'),
        style={'description_width': '80px'}
    )
    
    # Info panel
    info_panel = widgets.HTML(
        value="""
        <div style='background:#e3f2fd; border-left:4px solid #2196f3; padding:8px; margin:10px 0; border-radius:4px;'>
            <small style='color:#0d47a1;'>
                💡 <strong>Info:</strong> API key akan auto-detected dari environment variables atau Colab secrets.
            </small>
        </div>
        """
    )
    
    # Container dengan responsive layout
    container = create_responsive_container([
        header,
        workspace_field,
        project_field,
        version_field,
        api_key_field,
        format_dropdown,
        info_panel
    ], padding='15px', max_width='100%')
    
    # Add styling
    container.layout.border = '1px solid #ddd'
    container.layout.border_radius = '8px'
    container.layout.background_color = '#f8f9fa'
    
    return {
        'container': container,
        'workspace_field': workspace_field,
        'project_field': project_field,
        'version_field': version_field,
        'api_key_field': api_key_field,
        'format_dropdown': format_dropdown,
        'info_panel': info_panel
    }

def _create_options_panel(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create download options panel."""
    
    # Section header
    header = widgets.HTML(
        "<h4 style='margin:5px 0 10px 0; color:#495057; border-bottom:2px solid #28a745; padding-bottom:5px;'>⚙️ Download Options</h4>"
    )
    
    # Main options
    validate_checkbox = widgets.Checkbox(
        value=config.get('validate_download', True),
        description='Validate download integrity',
        layout=widgets.Layout(width='100%', margin='5px 0'),
        style={'description_width': 'initial'}
    )
    
    organize_checkbox = widgets.Checkbox(
        value=config.get('organize_dataset', True),
        description='Organize to train/valid/test structure',
        layout=widgets.Layout(width='100%', margin='5px 0'),
        style={'description_width': 'initial'}
    )
    
    backup_checkbox = widgets.Checkbox(
        value=config.get('backup_existing', False),
        description='Backup existing dataset before download',
        layout=widgets.Layout(width='100%', margin='5px 0'),
        style={'description_width': 'initial'}
    )
    
    # Progress options
    progress_checkbox = widgets.Checkbox(
        value=config.get('progress_enabled', True),
        description='Show progress tracking',
        layout=widgets.Layout(width='100%', margin='5px 0'),
        style={'description_width': 'initial'}
    )
    
    detailed_progress_checkbox = widgets.Checkbox(
        value=config.get('show_detailed_progress', False),
        description='Show detailed step-by-step progress',
        layout=widgets.Layout(width='100%', margin='5px 0'),
        style={'description_width': 'initial'}
    )
    
    # Advanced options dalam accordion
    advanced_accordion = _create_advanced_options_accordion(config)
    
    # Environment status
    env_status = _create_environment_status()
    
    # Container dengan responsive layout
    container = create_responsive_container([
        header,
        validate_checkbox,
        organize_checkbox,
        backup_checkbox,
        progress_checkbox,
        detailed_progress_checkbox,
        advanced_accordion,
        env_status
    ], padding='15px', max_width='100%')
    
    # Add styling
    container.layout.border = '1px solid #ddd'
    container.layout.border_radius = '8px'
    container.layout.background_color = '#f8f9fa'
    
    return {
        'container': container,
        'validate_checkbox': validate_checkbox,
        'organize_checkbox': organize_checkbox,
        'backup_checkbox': backup_checkbox,
        'progress_checkbox': progress_checkbox,
        'detailed_progress_checkbox': detailed_progress_checkbox,
        'advanced_accordion': advanced_accordion,
        'env_status': env_status
    }

def _create_advanced_options_accordion(config: Dict[str, Any]) -> widgets.Accordion:
    """Create advanced options dalam accordion untuk save space."""
    
    # Retry attempts
    retry_field = widgets.IntSlider(
        value=config.get('retry_attempts', 3),
        min=1,
        max=10,
        step=1,
        description='Retry:',
        layout=widgets.Layout(width='100%', margin='3px 0'),
        style={'description_width': '80px'}
    )
    
    # Timeout
    timeout_field = widgets.IntSlider(
        value=config.get('timeout_seconds', 30),
        min=10,
        max=300,
        step=10,
        description='Timeout (s):',
        layout=widgets.Layout(width='100%', margin='3px 0'),
        style={'description_width': '80px'}
    )
    
    # Chunk size
    chunk_size_field = widgets.IntSlider(
        value=config.get('chunk_size_kb', 8),
        min=1,
        max=64,
        step=1,
        description='Chunk (KB):',
        layout=widgets.Layout(width='100%', margin='3px 0'),
        style={'description_width': '80px'}
    )
    
    # Advanced container
    advanced_container = create_responsive_container([
        retry_field,
        timeout_field, 
        chunk_size_field
    ], padding='5px')
    
    # Create accordion
    accordion = widgets.Accordion([advanced_container])
    accordion.set_title(0, "🔧 Advanced Options")
    accordion.selected_index = None  # Collapsed by default
    accordion.layout.margin = '5px 0'
    
    return accordion

def _create_environment_status() -> widgets.HTML:
    """Create environment status indicator."""
    try:
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        
        if env_manager.is_colab and env_manager.is_drive_mounted:
            status_html = """
            <div style='background:#e8f5e8; border:1px solid #4caf50; border-radius:4px; padding:8px; margin:5px 0;'>
                <span style='color:#2e7d32;'>✅ Google Drive connected - dataset akan tersimpan permanen</span>
            </div>
            """
        elif env_manager.is_colab:
            status_html = """
            <div style='background:#fff3cd; border:1px solid #ffc107; border-radius:4px; padding:8px; margin:5px 0;'>
                <span style='color:#856404;'>⚠️ Local storage - dataset akan hilang saat restart</span>
            </div>
            """
        else:
            status_html = """
            <div style='background:#e3f2fd; border:1px solid #2196f3; border-radius:4px; padding:8px; margin:5px 0;'>
                <span style='color:#1565c0;'>💻 Local environment detected</span>
            </div>
            """
    except Exception:
        status_html = """
        <div style='background:#f5f5f5; border:1px solid #ccc; border-radius:4px; padding:8px; margin:5px 0;'>
            <span style='color:#666;'>🌐 Environment status unknown</span>
        </div>
        """
    
    return widgets.HTML(value=status_html)