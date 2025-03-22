"""
File: smartcash/ui/dataset/preprocessing_visualization_integration.py
Deskripsi: Integrasi visualisasi distribusi kelas dengan modul preprocessing yang efisien
"""

from typing import Dict, Any, Optional
from IPython.display import display
import matplotlib.pyplot as plt

def setup_preprocessing_visualization(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup integrasi visualisasi distribusi kelas untuk modul preprocessing dengan pendekatan DRY.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    from smartcash.ui.utils.constants import ICONS
    
    try:
        # Import visualisasi integrator dengan single import
        from smartcash.ui.dataset.visualization_integrator import setup_visualization_handlers
        
        # Setup handlers visualisasi dengan context preprocessing
        ui_components = setup_visualization_handlers(
            ui_components, env, config, context="preprocessing")
        
        # Tambahkan handler kustom jika diperlukan untuk preprocessing
        if 'distribution_button' in ui_components and 'visualization_buttons' in ui_components:
            # Set label khusus preprocessing
            ui_components['distribution_button'].description = 'Distribusi Dataset'
            ui_components['distribution_button'].tooltip = 'Visualisasi distribusi kelas setelah preprocessing'
            
            # Tambahkan handler spesifik untuk melihat detail preprocessing jika button tersedia
            if 'summary_button' in ui_components:
                # Tambahkan integrasi antara summary dan visualisasi
                original_handler = ui_components['summary_button']._click_handlers.callbacks[0] if ui_components['summary_button']._click_handlers.callbacks else None
                
                def enhanced_summary_handler(b):
                    # Panggil original handler terlebih dahulu jika ada
                    if original_handler: original_handler(b)
                    
                    # Tambahkan tombol show distribution setelah summary
                    from smartcash.ui.utils.alert_utils import create_info_alert
                    with ui_components.get('summary_container', ui_components.get('status')):
                        display(create_info_alert(
                            "Untuk melihat distribusi kelas, klik tombol 'Distribusi Dataset'", 
                            "info", ICONS.get('chart', '📊')))
                        
                    # Tampilkan tombol distribusi
                    ui_components['visualization_buttons'].layout.display = 'flex'
                
                # Replace handler jika ada summary button
                ui_components['summary_button'].on_click(enhanced_summary_handler)
            
            if logger: logger.info(f"{ICONS.get('success', '✅')} Integrasi visualisasi preprocessing berhasil")
        
        return ui_components
    
    except ImportError as e:
        if logger: logger.warning(f"{ICONS.get('warning', '⚠️')} Modul visualization_integrator tidak tersedia: {str(e)}")
        return ui_components
    except Exception as e:
        if logger: logger.warning(f"{ICONS.get('warning', '⚠️')} Error integrasi visualisasi preprocessing: {str(e)}")
        return ui_components