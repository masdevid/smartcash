"""
File: smartcash/ui/pretrained_model/handlers/progress_handler.py
Deskripsi: Handler khusus untuk progress tracking dengan SRP approach
"""

from typing import Dict, Any

def setup_progress_handler(ui_components: Dict[str, Any]):
    """Setup progress callback untuk UI updates"""
    
    def create_progress_callback():
        def progress_callback(**kwargs):
            progress = kwargs.get('progress', 0)
            message = kwargs.get('message', 'Processing...')
            ui_components.get('update_progress', lambda *a: None)('overall', progress, message)
        return progress_callback
    
    ui_components['progress_callback'] = create_progress_callback()