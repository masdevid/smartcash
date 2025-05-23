"""
File: smartcash/ui/dataset/preprocessing/handlers/progress_handler.py
Deskripsi: Handler untuk progress tracking preprocessing dengan 2-level progress display
"""

from typing import Dict, Any


def setup_progress_handler(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup handler untuk progress tracking preprocessing."""
    logger = ui_components.get('logger')
    
    # Initialize progress state
    ui_components['progress_state'] = {
        'current_step': 0,
        'total_steps': 3,
        'current_progress': 0,
        'total_progress': 100,
        'current_split': None,
        'split_progress': 0,
        'split_total': 0
    }
    
    def reset_progress():
        """Reset semua progress indicators."""
        state = ui_components['progress_state']
        state.update({
            'current_step': 0,
            'total_steps': 3,
            'current_progress': 0,
            'total_progress': 100,
            'current_split': None,
            'split_progress': 0,
            'split_total': 0
        })
        
        # Reset UI elements
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
        
        if 'overall_label' in ui_components:
            ui_components['overall_label'].value = "Siap memulai preprocessing"
        
        if 'step_label' in ui_components:
            ui_components['step_label'].value = "Langkah: Menunggu"
        
        if 'current_progress' in ui_components:
            ui_components['current_progress'].value = 0
    
    def update_overall_progress(progress: int, total: int, message: str = ""):
        """Update overall progress bar."""
        if 'progress_bar' in ui_components:
            percentage = min((progress / max(total, 1)) * 100, 100)
            ui_components['progress_bar'].value = percentage
        
        if 'overall_label' in ui_components and message:
            ui_components['overall_label'].value = f"Progress: {progress}/{total} - {message}"
        
        # Update state
        ui_components['progress_state']['current_progress'] = progress
        ui_components['progress_state']['total_progress'] = total
    
    def update_step_progress(step: int, total_steps: int, step_name: str = ""):
        """Update step progress indicator."""
        if 'step_label' in ui_components:
            step_text = f"Langkah {step}/{total_steps}"
            if step_name:
                step_text += f": {step_name}"
            ui_components['step_label'].value = step_text
        
        # Update state
        ui_components['progress_state']['current_step'] = step
        ui_components['progress_state']['total_steps'] = total_steps
    
    def update_split_progress(split: str, progress: int, total: int):
        """Update progress untuk split tertentu."""
        if 'current_progress' in ui_components:
            if total > 0:
                percentage = min((progress / total) * 100, 100)
                ui_components['current_progress'].value = percentage
        
        # Update state
        ui_components['progress_state'].update({
            'current_split': split,
            'split_progress': progress,
            'split_total': total
        })
    
    def show_progress_container():
        """Show progress tracking container."""
        if 'progress_container' in ui_components:
            ui_components['progress_container'].layout.visibility = 'visible'
            ui_components['progress_container'].layout.display = 'block'
    
    def hide_progress_container():
        """Hide progress tracking container."""
        if 'progress_container' in ui_components:
            ui_components['progress_container'].layout.visibility = 'hidden'
            ui_components['progress_container'].layout.display = 'none'
    
    def set_progress_message(message: str, level: str = "info"):
        """Set pesan progress dengan level tertentu."""
        if 'overall_label' in ui_components:
            state = ui_components['progress_state']
            progress_text = f"Progress: {state['current_progress']}/{state['total_progress']}"
            ui_components['overall_label'].value = f"{progress_text} - {message}"
        
        # Log juga ke logger jika ada
        if logger:
            if level == "success":
                logger.success(message)
            elif level == "error":
                logger.error(message)
            elif level == "warning":
                logger.warning(message)
            else:
                logger.info(message)
    
    # Attach helper functions ke ui_components untuk akses mudah
    ui_components['progress_helpers'] = {
        'reset': reset_progress,
        'update_overall': update_overall_progress,
        'update_step': update_step_progress,
        'update_split': update_split_progress,
        'show_container': show_progress_container,
        'hide_container': hide_progress_container,
        'set_message': set_progress_message
    }
    
    # Initialize state
    reset_progress()
    hide_progress_container()
    
    # Mark setup as complete
    ui_components['progress_setup'] = True
    
    if logger:
        logger.debug("✅ Progress handler preprocessing setup selesai")
    
    return ui_components