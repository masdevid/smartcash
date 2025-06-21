from smartcash.common.logger import get_logger
from smartcash.ui.dataset.visualization.components.dataset_comparator import DatasetComparator
import ipywidgets as widgets

logger = get_logger(__name__)


def create_comparison_tab() -> widgets.Widget:
    """Handler untuk membuat tab perbandingan dataset"""
    try:
        # Buat instance komparator
        comparator = DatasetComparator()
        
        # Dapatkan komponen UI
        ui_components = comparator.get_ui_components()
        
        # Buat header tab
        header = widgets.HTML(
            """<div style="padding: 10px; background-color: #f0f0f0; border-bottom: 1px solid #ddd;">
                <h2 style="margin: 0;">Perbandingan Dataset</h2>
            </div>"""
        )
        
        # Gabungkan semua komponen dalam VBox
        tab_content = widgets.VBox([header, ui_components['main_container']])
        
        return tab_content
    except Exception as e:
        logger.error(f"Gagal membuat tab perbandingan: {str(e)}")
        return widgets.HTML(f"<div style='color: red; padding: 20px;'>Error: {str(e)}</div>")
