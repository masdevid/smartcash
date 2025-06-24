"""
File: smartcash/ui/strategy/__init__.py
Deskripsi: Strategy module exports dan factory functions
"""

from IPython.display import display
from ipywidgets import VBox
from .strategy_init import create_strategy_config_cell, StrategyInitializer
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def init_strategy_ui(**kwargs):
    """
    Factory function yang langsung menampilkan UI strategy 📊
    
    Returns:
        ipywidgets.Widget: Widget UI yang sudah di-render
    """
    try:
        logger.info("🚀 Memulai inisialisasi UI Strategy...")
        
        # Dapatkan komponen UI
        ui_components = create_strategy_config_cell(**kwargs)
        
        if not ui_components:
            raise ValueError("Gagal membuat komponen UI")
        
        # Cek tipe return value
        if isinstance(ui_components, dict):
            # Jika ada 'main_layout', gunakan itu
            if 'main_layout' in ui_components:
                main_widget = ui_components['main_layout']
            # Jika ada 'ui_components', coba gabungkan
            elif 'ui_components' in ui_components:
                main_widget = VBox(ui_components['ui_components'])
            # Jika tidak ada, coba gabungkan semua widget yang ada
            else:
                widgets = [
                    comp for comp in ui_components.values() 
                    if hasattr(comp, 'layout') or hasattr(comp, 'value')
                ]
                main_widget = VBox(widgets) if widgets else None
        else:
            # Jika bukan dictionary, asumsikan ini adalah widget langsung
            main_widget = ui_components
        
        if main_widget is None:
            raise ValueError("Tidak dapat menentukan widget utama untuk ditampilkan")
        
        # Tampilkan widget
        display(main_widget)
        logger.info("✅ UI Strategy berhasil ditampilkan")
        
        return main_widget
        
    except Exception as e:
        error_msg = f"❌ Gagal menampilkan UI Strategy: {str(e)}"
        logger.error(error_msg)
        from ipywidgets import HTML
        display(HTML(f'<div style="color:red">{error_msg}</div>'))
        raise


# Export semua yang diperlukan
__all__ = [
    'init_strategy_ui',
    'create_strategy_config_cell', 
    'StrategyInitializer'
]