"""
File: smartcash/ui/dataset/visualization/visualization_initializer.py
Deskripsi: Inisialisasi UI untuk visualisasi dataset dengan pendekatan minimalis
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display
import threading

from smartcash.ui.dataset.visualization.handlers.dashboard_handler import create_dashboard_handler, update_dashboard_cards
from smartcash.ui.utils.loading_indicator import LoadingIndicator
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def initialize_visualization_ui(loading_indicator: Optional[LoadingIndicator] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI untuk visualisasi dataset dengan pendekatan minimalis.
    
    Args:
        loading_indicator: Indikator loading opsional untuk menampilkan progress
        
    Returns:
        Dictionary berisi komponen UI
    """
    try:
        # Update loading indicator jika ada
        if loading_indicator:
            loading_indicator.update(25, "Mempersiapkan komponen visualisasi...")
            
        # Buat container utama dengan layout minimalis
        main_container = widgets.VBox([], layout=widgets.Layout(width='100%', padding='10px'))
        
        # Buat tab untuk visualisasi dengan layout minimalis
        tab = widgets.Tab(layout=widgets.Layout(width='100%', margin='10px 0'))
        tab.children = []
        
        # Buat handler untuk dashboard
        dashboard_handler = create_dashboard_handler()
        
        # Tambahkan tab dashboard
        tab.children = [dashboard_handler['container']]
        tab.set_title(0, 'Dashboard')
        
        # Tambahkan tab ke container utama
        main_container.children = [tab]
        
        # Buat dictionary untuk komponen UI
        ui_components = {
            'main_container': main_container,
            'tab': tab,
            'dashboard_handler': dashboard_handler
        }
        
        # Update dashboard cards secara asinkron untuk menghindari UI freeze
        if loading_indicator:
            loading_indicator.update(50, "Memperbarui dashboard...")
            
            # Gunakan threading untuk memperbarui dashboard secara asinkron
            def update_dashboard_async():
                try:
                    update_dashboard_cards(ui_components)
                    if loading_indicator:
                        loading_indicator.complete("Visualisasi dataset berhasil dimuat")
                except Exception as e:
                    logger.error(f"Error saat memperbarui dashboard: {str(e)}")
                    if loading_indicator:
                        loading_indicator.error(f"Error saat memperbarui dashboard: {str(e)}")
            
            threading.Thread(target=update_dashboard_async, daemon=True).start()
        else:
            # Jika tidak ada loading indicator, update langsung
            update_dashboard_cards(ui_components)
        
        return ui_components
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat inisialisasi visualisasi dataset: {str(e)}")
        
        # Update loading indicator jika ada
        if loading_indicator:
            loading_indicator.error(f"Error: {str(e)}")
        
        # Buat container minimal untuk menampilkan error
        error_container = widgets.VBox([
            widgets.HTML(f"<div style='color: #d9534f; padding: 10px; border-left: 5px solid #d9534f;'>"
                        f"<h3>{ICONS.get('error', '❌')} Error saat inisialisasi visualisasi dataset</h3>"
                        f"<p>{str(e)}</p></div>")
        ])
        
        return {'error_container': error_container, 'main_container': error_container}
