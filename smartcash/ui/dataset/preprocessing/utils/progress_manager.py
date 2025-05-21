"""
File: smartcash/ui/dataset/preprocessing/utils/progress_manager.py
Deskripsi: Utilitas untuk mengelola progress bar dan tracking dalam UI preprocessing
"""

from typing import Dict, Any, Optional, Tuple, Callable
import ipywidgets as widgets
from smartcash.ui.utils.constants import COLORS
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message

def setup_multi_progress(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup progress tracking dengan dukungan multi-level.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Jika komponen progress sudah disetup, tidak perlu setup ulang
    if all(k in ui_components for k in ['progress_bar', 'overall_label', 'step_label']):
        return ui_components
    
    # Pastikan progress_container telah dibuat
    if 'progress_container' not in ui_components:
        ui_components['progress_container'] = widgets.VBox([])
    
    # Setup progress bar jika belum ada
    if 'progress_bar' not in ui_components:
        ui_components['progress_bar'] = widgets.FloatProgress(
            value=0,
            min=0,
            max=100,
            description='',
            bar_style='info',
            orientation='horizontal',
            layout=widgets.Layout(width='100%', visibility='hidden')
        )
    
    # Setup label untuk overall progress
    if 'overall_label' not in ui_components:
        ui_components['overall_label'] = widgets.HTML(
            value="",
            layout=widgets.Layout(margin='5px 0', visibility='hidden')
        )
    
    # Setup label untuk step progress
    if 'step_label' not in ui_components:
        ui_components['step_label'] = widgets.HTML(
            value="",
            layout=widgets.Layout(margin='0 0 5px 0', visibility='hidden')
        )
    
    # Setup current progress bar untuk step progress
    if 'current_progress' not in ui_components:
        ui_components['current_progress'] = widgets.FloatProgress(
            value=0,
            min=0,
            max=100,
            description='',
            bar_style='info',
            orientation='horizontal',
            layout=widgets.Layout(width='100%', visibility='hidden')
        )
    
    # Update progress container dengan komponen yang telah dibuat
    ui_components['progress_container'].children = [
        ui_components['overall_label'],
        ui_components['step_label'],
        ui_components['progress_bar'],
        ui_components['current_progress']
    ]
    
    return ui_components

def setup_progress_indicator(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup indikator progress yang lebih simpel (tidak multi-level).
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Jika sudah ada progress bar, gunakan yang sudah ada
    if 'progress_bar' in ui_components:
        return ui_components
    
    # Setup progress bar baru
    ui_components['progress_bar'] = widgets.FloatProgress(
        value=0,
        min=0,
        max=100,
        description='',
        bar_style='info',
        orientation='horizontal',
        layout=widgets.Layout(width='100%', visibility='hidden')
    )
    
    # Tambahkan ke container jika belum ada
    if 'progress_container' not in ui_components:
        ui_components['progress_container'] = widgets.VBox([ui_components['progress_bar']])
    else:
        current_children = list(ui_components['progress_container'].children)
        if ui_components['progress_bar'] not in current_children:
            ui_components['progress_container'].children = current_children + [ui_components['progress_bar']]
    
    return ui_components

def update_progress(ui_components: Dict[str, Any], 
                    value: float, 
                    max_value: float = 100,
                    overall_message: Optional[str] = None,
                    step_message: Optional[str] = None,
                    **kwargs) -> None:
    """
    Update nilai progress bar dan pesan progress.
    
    Args:
        ui_components: Dictionary komponen UI
        value: Nilai progress saat ini
        max_value: Nilai progress maksimum
        overall_message: Pesan untuk overall progress
        step_message: Pesan untuk step progress
        **kwargs: Parameter tambahan untuk progress tracking
    """
    # Pastikan progress bar tersedia
    if 'progress_bar' not in ui_components:
        return
    
    # Normalisasi nilai untuk progress bar
    normalized_value = min(100, (value / max_value) * 100) if max_value > 0 else 0
    
    # Update progress bar
    progress_bar = ui_components['progress_bar']
    progress_bar.value = normalized_value
    
    # Tampilkan progress bar jika tersembunyi
    if hasattr(progress_bar, 'layout') and progress_bar.layout.visibility == 'hidden':
        progress_bar.layout.visibility = 'visible'
    
    # Update warna progress bar berdasarkan nilai
    if normalized_value < 30:
        progress_bar.bar_style = 'info'
    elif normalized_value < 70:
        progress_bar.bar_style = 'warning'
    else:
        progress_bar.bar_style = 'success'
    
    # Update overall message
    if overall_message and 'overall_label' in ui_components:
        overall_label = ui_components['overall_label']
        overall_label.value = f"<div>{overall_message}</div>"
        
        # Tampilkan overall label jika tersembunyi
        if hasattr(overall_label, 'layout') and overall_label.layout.visibility == 'hidden':
            overall_label.layout.visibility = 'visible'
    
    # Update step message
    if step_message and 'step_label' in ui_components:
        step_label = ui_components['step_label']
        step_label.value = f"<div style='color:{COLORS['secondary']};font-size:0.9em;'>{step_message}</div>"
        
        # Tampilkan step label jika tersembunyi
        if hasattr(step_label, 'layout') and step_label.layout.visibility == 'hidden':
            step_label.layout.visibility = 'visible'
    
    # Check for current_step progress
    current_progress = kwargs.get('current_progress')
    current_total = kwargs.get('current_total')
    
    if current_progress is not None and current_total is not None and current_total > 0 and 'current_progress' in ui_components:
        # Normalisasi nilai untuk step progress
        step_value = min(100, (current_progress / current_total) * 100)
        
        # Update step progress bar
        step_progress_bar = ui_components['current_progress']
        step_progress_bar.value = step_value
        
        # Tampilkan step progress bar
        if hasattr(step_progress_bar, 'layout') and step_progress_bar.layout.visibility == 'hidden':
            step_progress_bar.layout.visibility = 'visible'
        
        # Update step description
        step = kwargs.get('step', 0)
        total_steps = kwargs.get('total_steps', 3)
        split = kwargs.get('split', '')
        
        if split:
            step_progress_bar.description = f"Split {split}"
        else:
            step_progress_bar.description = f"Step {step+1}/{total_steps}"
    
    # Log progress jika > 10% perubahan dari terakhir kali
    if 'last_logged_progress' not in ui_components:
        ui_components['last_logged_progress'] = 0
    
    if abs(normalized_value - ui_components['last_logged_progress']) >= 10 or normalized_value >= 100:
        if overall_message:
            log_message(ui_components, f"Progress: {normalized_value:.1f}% - {overall_message}", "debug", "📊")
        else:
            log_message(ui_components, f"Progress: {normalized_value:.1f}%", "debug", "📊")
            
        ui_components['last_logged_progress'] = normalized_value

def reset_progress_bar(ui_components: Dict[str, Any]) -> None:
    """
    Reset progress bar dan semua label progress.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Reset progress bar
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = 0
        if hasattr(ui_components['progress_bar'], 'layout'):
            ui_components['progress_bar'].layout.visibility = 'hidden'
    
    # Reset overall label
    if 'overall_label' in ui_components:
        ui_components['overall_label'].value = ""
        if hasattr(ui_components['overall_label'], 'layout'):
            ui_components['overall_label'].layout.visibility = 'hidden'
    
    # Reset step label
    if 'step_label' in ui_components:
        ui_components['step_label'].value = ""
        if hasattr(ui_components['step_label'], 'layout'):
            ui_components['step_label'].layout.visibility = 'hidden'
    
    # Reset current progress
    if 'current_progress' in ui_components:
        ui_components['current_progress'].value = 0
        if hasattr(ui_components['current_progress'], 'layout'):
            ui_components['current_progress'].layout.visibility = 'hidden'
    
    # Reset value tracking
    ui_components['last_logged_progress'] = 0

def start_progress(ui_components: Dict[str, Any], message: str = "Memulai preprocessing...") -> None:
    """
    Memulai progress tracking.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan awal progress
    """
    # Skip jika progress tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Reset progress dulu
    reset_progress_bar(ui_components)
    
    # Tampilkan progress container
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.visibility = 'visible'
        ui_components['progress_container'].layout.display = 'block'
    
    # Update progress awal
    update_progress(ui_components, 0, 100, message)

def complete_progress(ui_components: Dict[str, Any], message: str = "Preprocessing selesai") -> None:
    """
    Menyelesaikan progress tracking.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan akhir progress
    """
    # Skip jika progress tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Update progress ke 100%
    update_progress(ui_components, 100, 100, message)

def create_progress_callback(ui_components: Dict[str, Any]) -> Callable:
    """
    Membuat callback function untuk progress tracking yang kompatibel dengan DatasetPreprocessor.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Fungsi callback yang bisa digunakan oleh DatasetPreprocessor
    """
    def progress_callback(**kwargs):
        # Extract parameters
        progress = kwargs.get('progress', 0)
        total = kwargs.get('total', 100)
        if 'total_files_all' in kwargs:
            total = kwargs.get('total_files_all', 100)
            
        message = kwargs.get('message', '')
        status = kwargs.get('status', 'info')
        
        # Extract parameters untuk step progress
        step = kwargs.get('step', 0)
        total_steps = kwargs.get('total_steps', 3)
        split = kwargs.get('split', '')
        split_step = kwargs.get('split_step', '')
        current_progress = kwargs.get('current_progress', 0)
        current_total = kwargs.get('current_total', 0)
        
        # Format step message
        step_message = split_step
        if not step_message and split:
            step_message = f"Split: {split}"
        
        # Update progress dengan parameter yang diekstrak
        update_progress(
            ui_components, 
            progress, 
            total, 
            overall_message=message,
            step_message=step_message,
            step=step,
            total_steps=total_steps,
            split=split,
            current_progress=current_progress,
            current_total=current_total
        )
        
        # Log message jika diperlukan
        if message and message != ui_components.get('last_progress_message', ''):
            icon_map = {
                'info': "ℹ️",
                'success': "✅",
                'warning': "⚠️",
                'error': "❌"
            }
            icon = icon_map.get(status, "ℹ️")
            
            # Log message dengan level yang sesuai
            log_message(ui_components, message, status, icon)
            
            # Update last message
            ui_components['last_progress_message'] = message
        
        # Return True untuk menunjukkan progress berhasil diupdate
        return True
    
    return progress_callback 