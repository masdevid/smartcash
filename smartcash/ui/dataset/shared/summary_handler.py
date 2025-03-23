"""
File: smartcash/ui/dataset/shared/summary_handler.py
Deskripsi: Handler bersama untuk tampilan ringkasan hasil preprocessing/augmentasi
"""

from typing import Dict, Any, Optional
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
from smartcash.ui.utils.constants import COLORS, ICONS

def setup_shared_summary_handler(ui_components: Dict[str, Any], env=None, config=None, 
                               module_type: str = 'preprocessing') -> Dict[str, Any]:
    """
    Setup handler untuk ringkasan hasil processing.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Fungsi utama update summary dengan visualisasi yang konsisten
    def update_summary(result):
        """
        Update komponen summary dengan hasil preprocessing/augmentasi.
        
        Args:
            result: Hasil processing
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
            title = "Hasil Preprocessing" if module_type == 'preprocessing' else "Hasil Augmentasi"
            display(styled_html(
                f"<h3 style='margin-top:0; color:{COLORS['dark']}'>{ICONS.get('stats', '📊')} {title}</h3>", 
                bg_color=COLORS['light'], text_color=COLORS['dark'], 
                border_color=COLORS['primary']
            ))
            
            # Metrics grid
            metrics_container = widgets.HBox(layout=widgets.Layout(
                display='flex', flex_flow='row wrap', align_items='flex-start',
                justify_content='space-around', width='100%', margin='10px 0'
            ))
            
            # Metrik khusus berdasarkan jenis modul
            if module_type == 'preprocessing':
                metrics_container.children = [
                    create_metric_display("Jumlah Gambar", result.get('total_images', 0)),
                    create_metric_display("Resolusi", f"{result.get('image_size', [0, 0])[0]}x{result.get('image_size', [0, 0])[1]}"),
                    create_metric_display("Waktu Proses", f"{result.get('processing_time', 0):.2f}s")
                ]
            else:
                metrics_container.children = [
                    create_metric_display("File Original", result.get('original', 0)),
                    create_metric_display("File Augmentasi", result.get('generated', 0)),
                    create_metric_display("Total File", result.get('total_files', 0)),
                    create_metric_display("Waktu Proses", f"{result.get('duration', 0):.2f}s")
                ]
            
            display(metrics_container)
            
            # Split stats jika tersedia (hanya di preprocessing)
            if 'split_stats' in result and module_type == 'preprocessing':
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
            
            # Jenis augmentasi (hanya di augmentation)
            if 'augmentation_types' in result and module_type == 'augmentation':
                display(styled_html(
                    f"<p style='color:{COLORS['dark']};'><strong>Jenis augmentasi:</strong> {', '.join(result.get('augmentation_types', []))}</p>", 
                    bg_color=COLORS['light'], text_color=COLORS['dark']
                ))
            
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
            if 'distribution_button' in ui_components:
                ui_components['distribution_button'].layout.display = 'inline-flex'
    
    # Fungsi untuk generate ringkasan preprocessing
    def generate_preprocessing_summary(preprocessed_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate ringkasan hasil preprocessing dari direktori.
        
        Args:
            preprocessed_dir: Direktori hasil preprocessing
            
        Returns:
            Dictionary ringkasan hasil preprocessing
        """
        from smartcash.ui.dataset.shared.get_preprocessing_stats import get_preprocessing_stats
        
        # Gunakan default jika tidak disediakan
        preprocessed_dir = preprocessed_dir or ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Dapatkan konfigurasi
        img_size = [640, 640]  # Default
        try:
            # Ambil dari ui_components khusus preprocessing
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
    
    # Fungsi untuk generate ringkasan augmentasi
    def generate_augmentation_summary(augmented_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate ringkasan hasil augmentasi dari direktori.
        
        Args:
            augmented_dir: Direktori hasil augmentasi
            
        Returns:
            Dictionary ringkasan hasil augmentasi
        """
        from smartcash.ui.visualization.class_distribution_analyzer import count_files_by_prefix
        
        # Gunakan default jika tidak disediakan
        augmented_dir = augmented_dir or ui_components.get('augmented_dir', 'data/augmented')
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Cari semua file di kedua lokasi
        aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
        
        # Hitung file di folder augmentasi
        aug_files = 0
        aug_path = Path(augmented_dir)
        if aug_path.exists():
            aug_images_dir = aug_path / 'images'
            if aug_images_dir.exists():
                aug_files = len(list(aug_images_dir.glob(f"{aug_prefix}_*.*")))
        
        # Hitung file di folder preprocessed
        orig_files = 0
        preproc_files = 0
        preproc_path = Path(preprocessed_dir)
        if preproc_path.exists():
            for split in ['train', 'valid', 'test']:
                images_dir = preproc_path / split / 'images'
                if not images_dir.exists():
                    continue
                
                orig_prefix = 'rp'  # Default prefix untuk file original
                orig_files += len(list(images_dir.glob(f"{orig_prefix}_*.*")))
                preproc_files += len(list(images_dir.glob(f"{aug_prefix}_*.*")))
        
        # Dapatkan nilai dari aug_options jika tersedia
        aug_types = []
        if 'aug_options' in ui_components and hasattr(ui_components['aug_options'], 'children') and len(ui_components['aug_options'].children) > 0:
            # Map UI types to config types
            type_map = {'Combined (Recommended)': 'combined', 'Position Variations': 'position', 
                      'Lighting Variations': 'lighting', 'Extreme Rotation': 'extreme_rotation'}
            
            selected_types = ui_components['aug_options'].children[0].value
            aug_types = [type_map.get(t, 'combined') for t in selected_types]
        
        # Buat ringkasan
        summary = {
            'original': orig_files,
            'generated': aug_files + preproc_files,
            'total_files': orig_files + aug_files + preproc_files,
            'duration': 0,  # Tidak diketahui karena loading dari disk
            'augmentation_types': aug_types,
            'output_dir': preprocessed_dir
        }
        
        return summary
    
    # Handler untuk tombol summary
    def on_summary_click(b):
        """Handler untuk tombol summary."""
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator('info', f"{ICONS['processing']} Mempersiapkan ringkasan..."))
        
        try:
            # Generate ringkasan berdasarkan module_type
            summary = generate_preprocessing_summary() if module_type == 'preprocessing' else generate_augmentation_summary()
            
            # Update UI dengan ringkasan
            update_summary(summary)
            
            # Tampilkan status sukses
            with ui_components['status']:
                display(create_status_indicator('success', f"{ICONS['success']} Ringkasan berhasil ditampilkan"))
                
        except Exception as e:
            with ui_components['status']:
                display(create_status_indicator('error', f"{ICONS['error']} Error saat membuat ringkasan: {str(e)}"))
            if logger: logger.error(f"{ICONS['error']} Error saat membuat ringkasan: {str(e)}")
    
    # Tambahkan handler ke tombol jika tersedia
    if 'summary_button' in ui_components:
        ui_components['summary_button'].on_click(on_summary_click)
    
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'update_summary': update_summary,
        'generate_summary': generate_preprocessing_summary if module_type == 'preprocessing' else generate_augmentation_summary,
        'on_summary_click': on_summary_click
    })
    
    return ui_components