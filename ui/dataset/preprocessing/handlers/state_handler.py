"""
File: smartcash/ui/dataset/preprocessing/handlers/state_handler.py
Deskripsi: Handler state untuk preprocessing dataset
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
from smartcash.ui.dataset.preprocessing.handlers.status_handler import update_status_panel
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS

def detect_preprocessing_state(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deteksi status data preprocessing dan update tampilan UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    try:
        # Dapatkan paths dari ui_components
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Gunakan Google Drive jika tersedia
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            if env_manager.is_drive_mounted:
                drive_path = str(env_manager.drive_path)
                preprocessed_dir = os.path.join(drive_path, 'SmartCash', preprocessed_dir)
                if logger: logger.debug(f"🔍 Mencari data preprocessing di drive: {preprocessed_dir}")
        except (ImportError, AttributeError):
            pass
        
        # Konversi ke path absolut
        abs_preprocessed_dir = os.path.abspath(preprocessed_dir)
        
        # Cek apakah direktori preprocessing ada
        preproc_path = Path(preprocessed_dir)
        is_preprocessed = False
        
        # Cek apakah ada data preprocessed dalam split
        for split in DEFAULT_SPLITS:
            split_dir = preproc_path / split
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                # Cek apakah ada file di dalamnya
                image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')) + list(images_dir.glob('*.npy'))
                label_files = list(labels_dir.glob('*.txt'))
                
                if image_files and label_files:
                    is_preprocessed = True
                    if logger: logger.info(f"✅ Data preprocessing ditemukan di split {split}: {len(image_files)} gambar, {len(label_files)} label")
                    break
        
        # Update UI berdasarkan hasil deteksi
        ui_components['is_preprocessed'] = is_preprocessed
        
        if is_preprocessed:
            # Update status panel
            message = f"Dataset preprocessed sudah tersedia di: {abs_preprocessed_dir}"
            update_status_panel(ui_components, "success", f"{ICONS['success']} {message}")
            
            # Tampilkan tombol yang relevan
            ui_components['cleanup_button'].layout.display = 'block'
            ui_components['visualization_buttons'].layout.display = 'flex'
            
            # Tampilkan tombol visualisasi individual
            for btn in ['visualize_button', 'compare_button', 'distribution_button']:
                if btn in ui_components:
                    ui_components[btn].disabled = False
            
            # Persiapkan container visualisasi dan summary
            for container in ['visualization_container', 'summary_container']:
                if container in ui_components:
                    ui_components[container].layout.display = 'block'
            
            # Tampilkan summary
            generate_preprocessing_summary(ui_components, preprocessed_dir)
        else:
            # Update status panel
            message = "Belum ada data preprocessing. Silakan jalankan preprocessing."
            update_status_panel(ui_components, "info", f"{ICONS['info']} {message}")
            
            # Sembunyikan tombol visualisasi dan cleanup
            ui_components['cleanup_button'].layout.display = 'none'
            ui_components['visualization_buttons'].layout.display = 'none'
            
            # Sembunyikan container
            for container in ['visualization_container', 'summary_container']:
                if container in ui_components:
                    ui_components[container].layout.display = 'none'
                    
            if logger: logger.info(f"ℹ️ Tidak ditemukan data preprocessing di {abs_preprocessed_dir}")
    
    except Exception as e:
        # Log error
        if logger: logger.warning(f"⚠️ Error saat mendeteksi status preprocessing: {str(e)}")
    
    return ui_components

def generate_preprocessing_summary(ui_components: Dict[str, Any], preprocessed_dir: Optional[str] = None) -> None:
    """
    Generate dan tampilkan ringkasan dataset preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        preprocessed_dir: Direktori dataset preprocessed
    """
    logger = ui_components.get('logger')
    
    try:
        # Gunakan preprocessed_dir dari parameter atau ui_components
        preprocessed_dir = preprocessed_dir or ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Get preprocessing stats
        from smartcash.ui.dataset.shared.get_preprocessing_stats import get_preprocessing_stats
        stats = get_preprocessing_stats(ui_components, preprocessed_dir)
        
        # Dapatkan UI options
        img_size = 640  # Default
        if 'preprocess_options' in ui_components and hasattr(ui_components['preprocess_options'], 'children'):
            img_size = ui_components['preprocess_options'].children[0].value
        
        # Tampilkan summary jika ditemukan data
        if stats['valid'] and 'summary_container' in ui_components:
            with ui_components['summary_container']:
                clear_output(wait=True)
                
                # Header
                display(widgets.HTML(f"<h3 style='color:{COLORS['dark']}'>{ICONS['stats']} Ringkasan Dataset</h3>"))
                
                # Metrics grid
                metrics_container = widgets.HBox(layout=widgets.Layout(
                    display='flex', flex_flow='row wrap', align_items='flex-start',
                    justify_content='space-around', width='100%', margin='10px 0'
                ))
                
                # Create metric displays
                from smartcash.ui.utils.metric_utils import create_metric_display
                metrics_container.children = [
                    create_metric_display("Total Gambar", stats['total']['images']),
                    create_metric_display("Total Label", stats['total']['labels']),
                    create_metric_display("Resolusi", f"{img_size}x{img_size}")
                ]
                
                display(metrics_container)
                
                # Show split stats in a table
                th_style = f"padding:8px; text-align:center; border:1px solid #ddd; color:#343a40;";
                split_stats_html = f"""
                <table style="width:100%; border-collapse:collapse; margin:10px 0; border:1px solid #ddd;">
                    <thead>
                        <tr style="background-color:#f8f9fa;">
                            <th style="{th_style}">Split</th>
                            <th style="{th_style}">Gambar</th>
                            <th style="{th_style}">Label</th>
                            <th style="{th_style}">Status</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for split, split_info in stats['splits'].items():
                    # Warna status
                    status_color = COLORS['success'] if split_info.get('complete', False) else COLORS['warning']
                    status_icon = ICONS['success'] if split_info.get('complete', False) else ICONS['warning']
                    status_text = "Lengkap" if split_info.get('complete', False) else "Tidak Lengkap"
                    td_style = f"padding:8px; text-align:center; border:1px solid #ddd;"
                    
                    split_stats_html += f"""
                    <tr>
                        <td style="{td_style}color:{COLORS['dark']}; border:1px solid #ddd;">{split.capitalize()}</td>
                        <td style="{td_style}color:{COLORS['dark']};">{split_info.get('images', 0)}</td>
                        <td style="{td_style}color:{COLORS['dark']};">{split_info.get('labels', 0)}</td>
                        <td style="{td_style}color:{status_color};">
                            {status_icon} {status_text}
                        </td>
                    </tr>
                    """
                
                split_stats_html += """
                    </tbody>
                </table>
                """
                
                display(HTML(split_stats_html))
                
                # Output directory info
                from smartcash.ui.utils.metric_utils import styled_html
                display(styled_html(
                    f"<p style='color:{COLORS['dark']};'><strong>Path Output:</strong> {preprocessed_dir}</p>", 
                    bg_color=COLORS['light'], text_color=COLORS['dark']
                ))
                
            # Aktifkan tombol visualisasi jika ada data
            for btn in ['visualize_button', 'compare_button', 'distribution_button']:
                if btn in ui_components:
                    ui_components[btn].disabled = False
    except Exception as e:
        if logger: logger.warning(f"⚠️ Error saat membuat ringkasan preprocessing: {str(e)}")