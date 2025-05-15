"""
File: smartcash/ui/dataset/split/split_initializer.py
Deskripsi: Initializer untuk modul konfigurasi split dataset
"""

from typing import Dict, Any
from smartcash.ui.utils.base_initializer import initialize_module_ui
from smartcash.ui.dataset.split.components.split_component import create_split_ui
from smartcash.ui.dataset.split.handlers.button_handlers import setup_button_handlers

def setup_split_handlers(ui_components: Dict[str, Any], env: Any, config: Any) -> Dict[str, Any]:
    """Setup handler spesifik untuk modul split dataset"""
    # Setup handlers (non-singleton)
    setup_button_handlers(ui_components, config, env)
    
    # Register cleanup handler untuk slider
    from IPython import get_ipython
    if get_ipython():
        def cleanup():
            try:
                # Unobserve all sliders
                for s in ui_components.get('split_sliders', []):
                    if hasattr(s, 'unobserve_all'):
                        s.unobserve_all()
                # Log cleanup
                if 'logger' in ui_components: 
                    ui_components['logger'].info("🧹 UI split config event handlers cleaned up")
                return True
            except Exception as e:
                if 'logger' in ui_components:
                    ui_components['logger'].error(f"❌ Error during cleanup: {str(e)}")
                return False
        
        ui_components['cleanup'] = cleanup
        get_ipython().events.register('pre_run_cell', cleanup)
    
    return ui_components

def initialize_split_ui() -> Dict[str, Any]:
    """Inisialisasi UI modul konfigurasi split dataset."""
    
    # Konfigurasi multi-progress tracking
    multi_progress_config = {
        "module_name": "split",
        "step_key": "split_step",
        "progress_bar_key": "progress_bar",
        "current_progress_key": "current_progress",
        "overall_label_key": "overall_label",
        "step_label_key": "step_label"
    }
    
    # Tombol yang perlu diattach dengan ui_components
    button_keys = ['save_button', 'reset_button']
    
    # Gunakan base initializer
    return initialize_module_ui(
        module_name='split_config',
        create_ui_func=create_split_ui,
        setup_specific_handlers_func=setup_split_handlers,
        button_keys=button_keys,
        multi_progress_config=multi_progress_config
    )
