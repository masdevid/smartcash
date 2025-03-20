"""
File: smartcash/ui/dataset/preprocessing_summary_handler.py
Deskripsi: Handler untuk menampilkan ringkasan dan statistik hasil preprocessing dengan komponen UI standar dan perbaikan warna header
"""

from typing import Dict, Any, Optional
from IPython.display import display, clear_output, HTML
import time
from pathlib import Path
import ipywidgets as widgets

from smartcash.ui.utils.constants import COLORS, ICONS


def setup_summary_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk ringkasan preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Fungsi utama update summary dengan berbagai visualisasi dan styling konsisten
    def update_summary(result):
        """
        Update komponen summary dengan hasil preprocessing.
        
        Args:
            result: Hasil preprocessing
        """
        if not result or 'summary_container' not in ui_components:
            return
            
        # Pastikan container ditampilkan
        ui_components['summary_container'].layout.display = 'block'
        
        with ui_components['summary_container']:
            clear_output(wait=True)
            
            # Gunakan komponen standar untuk styling
            from smartcash.ui.utils.metric_utils import styled_html, create_metric_display
            from smartcash.ui.utils.constants import COLORS, ICONS
            
            # Header dengan warna yang terlihat
            display(styled_html(
                f"<h3 style='margin-top:0; color:{COLORS['dark']}'>{ICONS.get('stats', '📊')} Hasil Preprocessing</h3>", 
                bg_color=COLORS['light'], text_color=COLORS['dark'], 
                border_color=COLORS['primary']
            ))
            
            # Metrics grid
            metrics_container = widgets.HBox(layout=widgets.Layout(
                display='flex', flex_flow='row wrap', align_items='flex-start',
                justify_content='space-around', width='100%', margin='10px 0'
            ))
            
            # Jumlah gambar
            metrics_container.children = [
                create_metric_display("Jumlah Gambar", result.get('total_images', 0)),
                create_metric_display("Resolusi", f"{result.get('image_size', [0, 0])[0]}x{result.get('image_size', [0, 0])[1]}"),
                create_metric_display("Waktu Proses", f"{result.get('processing_time', 0):.2f}s")
            ]
            
            display(metrics_container)
            # Split stats jika tersedia
            if 'split_stats' in result:
                # Buat tabel untuk split stats
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
                
                for split, stats in result['split_stats'].items():
                    # Warna status
                    status_color = COLORS['success'] if stats.get('complete', False) else COLORS['warning']
                    status_icon = ICONS['success'] if stats.get('complete', False) else ICONS['warning']
                    status_text = "Lengkap" if stats.get('complete', False) else "Tidak Lengkap"
                    td_style = f"padding:8px; text-align:center; border:1px solid #ddd;"
                    
                    split_stats_html += f"""
                    <tr>
                        <td style="{td_style} color:{COLORS['dark']}; border:1px solid #ddd; ">{split.capitalize()}</td>
                        <td style="{td_style} color:{COLORS['dark']};">{stats.get('images', 0)}</td>
                        <td style="{td_style} color:{COLORS['dark']};">{stats.get('labels', 0)}</td>
                        <td style="{td_style} color:{status_color}">
                            {status_icon} {status_text}
                        </td>
                    </tr>
                    """
                
                split_stats_html += """
                    </tbody>
                </table>
                """
                
                display(HTML(split_stats_html))
            
            # Info tambahan
            if 'output_dir' in result:
                display(styled_html(
                    f"<p style='color:{COLORS['dark']};'><strong>Path Output:</strong> {result.get('output_dir', '')}</p>", 
                    bg_color=COLORS['light'], text_color=COLORS['dark']
                ))
            
            # Tombol visualisasi jika tersedia
            if 'visualize_button' in ui_components:
                ui_components['visualize_button'].layout.display = 'inline-flex'
                
            if 'compare_button' in ui_components:
                ui_components['compare_button'].layout.display = 'inline-flex'
    
    # Fungsi untuk generate ringkasan
    def generate_preprocessing_summary(preprocessed_dir: Optional[str] = None, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate ringkasan hasil preprocessing dari direktori.
        
        Args:
            preprocessed_dir: Direktori hasil preprocessing
            data_dir: Direktori dataset mentah
            
        Returns:
            Dictionary ringkasan hasil preprocessing
        """
        from smartcash.ui.dataset.preprocessing_visualization_handler import get_preprocessing_stats
        
        # Gunakan default jika tidak disediakan
        preprocessed_dir = preprocessed_dir or ui_components.get('preprocessed_dir', 'data/preprocessed')
        data_dir = data_dir or ui_components.get('data_dir', 'data')
        
        # Dapatkan konfigurasi
        img_size = [640, 640]  # Default
        try:
            # Ambil dari ui_components
            if 'preprocess_options' in ui_components and hasattr(ui_components['preprocess_options'], 'children'):
                img_size_value = ui_components['preprocess_options'].children[0].value
                img_size = [img_size_value, img_size_value]
            # Atau dari config
            elif config and 'preprocessing' in config and 'img_size' in config['preprocessing']:
                img_size = config['preprocessing']['img_size']
        except Exception:
            pass
        
        # Dapatkan statistik
        stats = get_preprocessing_stats(ui_components, preprocessed_dir)
        
        # Buat ringkasan
        summary = {
            'total_images': stats['total']['images'],
            'total_labels': stats['total']['labels'],
            'valid': stats['valid'],
            'image_size': img_size,
            'processing_time': 0,  # Tidak diketahui karena loading dari disk
            'output_dir': preprocessed_dir,
            'split_stats': stats['splits']
        }
        
        return summary
    
    # Handler untuk tombol summary
    def on_summary_click(b):
        """Handler untuk tombol summary."""
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan ringkasan preprocessing..."))
        
        try:
            # Generate ringkasan
            summary = generate_preprocessing_summary()
            
            # Update UI dengan ringkasan
            update_summary(summary)
            
            # Tampilkan status sukses
            with ui_components['status']:
                display(create_status_indicator('success', f"{ICONS['success']} Ringkasan preprocessing berhasil ditampilkan"))
                
        except Exception as e:
            with ui_components['status']:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat membuat ringkasan: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat membuat ringkasan preprocessing: {str(e)}")
    
    # Tambahkan handler ke tombol jika tersedia
    if 'summary_button' in ui_components:
        ui_components['summary_button'].on_click(on_summary_click)
    
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'update_summary': update_summary,
        'generate_preprocessing_summary': generate_preprocessing_summary,
        'on_summary_click': on_summary_click
    })
    
    return ui_components

def create_preprocessing_metrics(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Buat dan tampilkan metrik preprocessing dalam format card.
    
    Args:
        ui_components: Dictionary komponen UI
        result: Hasil preprocessing
    """
    import ipywidgets as widgets
    from smartcash.ui.utils.metric_utils import create_metric_display
    
    # Container untuk metrik
    metrics_container = widgets.HBox(
        layout=widgets.Layout(
            display='flex',
            flex_flow='row wrap',
            align_items='stretch',
            width='100%'
        )
    )
    
    # Buat metric cards
    metrics_container.children = [
        create_metric_display("Jumlah Gambar", result.get('total_images', 0)),
        create_metric_display("Resolusi", f"{result.get('image_size', [0, 0])[0]}x{result.get('image_size', [0, 0])[1]}"),
        create_metric_display("Waktu Proses", f"{result.get('processing_time', 0):.2f}s", "detik"),
        create_metric_display("Status", "Lengkap" if result.get('valid', False) else "Tidak Lengkap", 
                             is_good=result.get('valid', False))
    ]
    
    # Tampilkan dalam container
    if 'summary_container' in ui_components:
        with ui_components['summary_container']:
            display(metrics_container)