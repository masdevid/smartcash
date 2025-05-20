"""
File: smartcash/ui/components/progress_tracking.py
Deskripsi: Komponen progress tracking yang dapat digunakan kembali
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Tuple
from IPython.display import display, clear_output
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

def create_progress_tracking(
    module_name: str = 'process',
    show_step_progress: bool = True,
    show_overall_progress: bool = True,
    width: str = '100%'
) -> Dict[str, widgets.Widget]:
    """
    Membuat komponen progress tracking yang dapat digunakan kembali.
    
    Args:
        module_name: Nama modul untuk label progress
        show_step_progress: Tampilkan progress untuk step saat ini
        show_overall_progress: Tampilkan progress keseluruhan
        width: Lebar komponen
        
    Returns:
        Dictionary berisi komponen progress tracking
    """
    # Container untuk status output
    status = widgets.Output(
        layout=widgets.Layout(
            width=width,
            min_height='50px',
            max_height='150px',
            overflow='auto',
            margin='5px 0'
        )
    )
    
    # Progress bar untuk keseluruhan proses
    progress_bar = widgets.IntProgress(
        min=0,
        max=100,
        bar_style='info',
        orientation='horizontal',
        layout=widgets.Layout(
            width=width,
            visibility='hidden',
            margin='5px 0'
        )
    )
    
    # Set nilai awal progress bar
    try:
        progress_bar.value = 0
        progress_bar.description = "Progress: 0%"
    except Exception:
        # Jika gagal set nilai awal, abaikan
        pass
    
    # Label untuk overall progress
    overall_label = widgets.HTML(
        value="",
        layout=widgets.Layout(
            width=width,
            margin='5px 0',
            visibility='hidden'
        )
    )
    
    # Komponen untuk step saat ini
    current_progress = None
    step_label = None
    
    if show_step_progress:
        # Progress untuk step saat ini
        current_progress = widgets.IntProgress(
            min=0,
            max=1,
            bar_style='info',
            orientation='horizontal',
            layout=widgets.Layout(
                width=width,
                visibility='hidden',
                margin='5px 0'
            )
        )
        
        # Set nilai awal step progress
        try:
            current_progress.value = 0
            current_progress.description = "Step 0/0"
        except Exception:
            # Jika gagal set nilai awal, abaikan
            pass
        
        # Label untuk step saat ini
        step_label = widgets.HTML(
            value="",
            layout=widgets.Layout(
                width=width,
                margin='5px 0',
                visibility='hidden'
            )
        )
    
    # Container untuk semua komponen progress
    components = [progress_bar]
    
    if show_overall_progress:
        components.append(overall_label)
    
    if show_step_progress:
        components.extend([current_progress, step_label])
    
    progress_container = widgets.VBox(
        components,
        layout=widgets.Layout(
            width=width,
            margin='10px 0'
        )
    )
    
    # Kembalikan dictionary berisi semua komponen
    result = {
        'status': status,
        'progress_bar': progress_bar,
        'progress_container': progress_container
    }
    
    if show_overall_progress:
        result['overall_label'] = overall_label
    
    if show_step_progress:
        result['current_progress'] = current_progress
        result['step_label'] = step_label
    
    return result

def update_progress(
    ui_components: Dict[str, Any],
    progress: int,
    total: int = 100,
    message: str = None,
    step: int = None,
    total_steps: int = None,
    step_message: str = None,
    status_type: str = 'info'
) -> None:
    """
    Update komponen progress tracking.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        progress: Nilai progress saat ini
        total: Total progress
        message: Pesan untuk overall progress
        step: Step saat ini
        total_steps: Total jumlah step
        step_message: Pesan untuk step saat ini
        status_type: Tipe status (success, error, info, warning)
    """
    # Ensure progress is an integer
    try:
        progress = int(float(progress))
    except (ValueError, TypeError):
        progress = 0
        
    # Ensure total is an integer
    try:
        total = int(float(total))
    except (ValueError, TypeError):
        total = 100
    
    # Update progress bar
    if 'progress_bar' in ui_components:
        progress_bar = ui_components['progress_bar']
        progress_bar.layout.visibility = 'visible'
        progress_bar.max = total
        progress_bar.value = min(progress, total)
        progress_bar.description = f"Progress: {int(progress/total*100)}%"
    
    # Update overall label
    if message and 'overall_label' in ui_components:
        overall_label = ui_components['overall_label']
        overall_label.layout.visibility = 'visible'
        overall_label.value = message
    
    # Update step progress
    if step is not None and 'current_progress' in ui_components:
        current_progress = ui_components['current_progress']
        current_progress.layout.visibility = 'visible'
        current_progress.max = total_steps or 1
        current_progress.value = min(step, total_steps or 1)
        current_progress.description = f"Step {step}/{total_steps or 1}"
    
    # Update step label
    if step_message and 'step_label' in ui_components:
        step_label = ui_components['step_label']
        step_label.layout.visibility = 'visible'
        step_label.value = step_message
    
    # Update status
    if message and 'status' in ui_components:
        with ui_components['status']:
            display(create_status_indicator(status_type, f"{ICONS.get('processing', '🔄')} {message}"))

def reset_progress(ui_components: Dict[str, Any]) -> None:
    """
    Reset komponen progress tracking.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    # Reset progress bar
    if 'progress_bar' in ui_components:
        progress_bar = ui_components['progress_bar']
        progress_bar.value = 0
        progress_bar.description = "Progress: 0%"
        progress_bar.layout.visibility = 'hidden'
    
    # Reset overall label
    if 'overall_label' in ui_components:
        ui_components['overall_label'].value = ""
        ui_components['overall_label'].layout.visibility = 'hidden'
    
    # Reset step progress
    if 'current_progress' in ui_components:
        current_progress = ui_components['current_progress']
        current_progress.value = 0
        current_progress.max = 1
        current_progress.description = "Step 0/0"
        current_progress.layout.visibility = 'hidden'
    
    # Reset step label
    if 'step_label' in ui_components:
        ui_components['step_label'].value = ""
        ui_components['step_label'].layout.visibility = 'hidden'
