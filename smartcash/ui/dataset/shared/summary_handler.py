"""
File: smartcash/ui/dataset/shared/summary_handler.py
Deskripsi: Handler bersama untuk menampilkan ringkasan hasil processing dengan
styling konsisten antara preprocessing dan augmentasi
"""

from typing import Dict, Any, Optional, List, Tuple
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
from pathlib import Path
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS, DEFAULT_IMG_SIZE

def setup_shared_summary_handler(ui_components: Dict[str, Any], env=None, config=None, 
                                module_type: str = 'preprocessing') -> Dict[str, Any]:
    """
    Setup handler untuk ringkasan hasil processing dengan styling yang konsisten.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        
    Returns:
        Dictionary UI components yang diupdate
    """
    logger = ui_components.get('logger')
    
    # Fungsi utama untuk update summary dengan styling yang konsisten
    def update_summary(result: Dict[str, Any]) -> None:
        """
        Update summary container dengan hasil processing.
        
        Args:
            result: Dictionary hasil processing
        """
        if not result or 'summary_container' not in ui_components:
            return
        
        # Pastikan container ditampilkan
        ui_components['summary_container'].layout.display = 'block'
        
        with ui_components['summary_container']:
            clear_output(wait=True)
            
            # Import utilitas standar untuk styling
            from smartcash.ui.utils.metric_utils import styled_html, create_metric_display
            
            # Header dengan styling yang konsisten
            title = "Hasil Preprocessing" if module_type == 'preprocessing' else "Hasil Augmentasi"
            display(styled_html(
                f"<h3 style='margin-top:0; color:{COLORS['dark']}'>{ICONS.get('stats', '📊')} {title}</h3>", 
                bg_color=COLORS['light'], text_color=COLORS['dark'], 
                border_color=COLORS['primary']
            ))
            
            # Metrics grid yang bisa diperluas
            metrics_container = widgets.HBox(layout=widgets.Layout(
                display='flex', flex_flow='row wrap', align_items='flex-start',
                justify_content='space-around', width='100%', margin='10px 0'
            ))
            
            # Metrik berbeda berdasarkan module_type
            if module_type == 'preprocessing':
                metrics_container.children = [
                    create_metric_display("Jumlah Gambar", result.get('total_images', 0)),
                    create_metric_display("Resolusi", f"{result.get('image_size', [0, 0])[0]}x{result.get('image_size', [0, 0])[1]}"),
                    create_metric_display("Waktu Proses", f"{result.get('processing_time', 0):.2f}s"),
                    create_metric_display("Status", "Lengkap" if result.get('valid', False) else "Tidak Lengkap", 
                                        is_good=result.get('valid', False))
                ]
            else:
                # Metrik untuk augmentasi
                metrics_container.children = [
                    create_metric_display("File Original", result.get('original', 0)),
                    create_metric_display("File Augmentasi", result.get('generated', 0)),
                    create_metric_display("Total File", result.get('total_files', 0)),
                    create_metric_display("Waktu Proses", f"{result.get('duration', 0):.2f}s")
                ]
            
            display(metrics_container)
            
            # Tampilkan informasi split (hanya di preprocessing)
            if 'split_stats' in result and module_type == 'preprocessing':
                _display_split_statistics(result['split_stats'])
            
            # Tampilkan info tambahan berdasarkan module_type
            if module_type == 'augmentation' and 'augmentation_types' in result:
                display(styled_html(
                    f"<p style='color:{COLORS['dark']};'><strong>Jenis augmentasi:</strong> {', '.join(result.get('augmentation_types', []))}</p>", 
                    bg_color=COLORS['light'], text_color=COLORS['dark']
                ))
            
            # Tampilkan path output
            if 'output_dir' in result:
                display(styled_html(
                    f"<p style='color:{COLORS['dark']};'><strong>Path Output:</strong> {result.get('output_dir', '')}</p>", 
                    bg_color=COLORS['light'], text_color=COLORS['dark']
                ))
            
            # Update visibility tombol visualisasi jika tersedia
            for btn in ['visualize_button', 'compare_button', 'distribution_button']:
                if btn in ui_components:
                    ui_components[btn].layout.display = 'inline-flex'
    
    # Fungsi helper untuk tampilkan statistik split dataset
    def _display_split_statistics(split_stats: Dict[str, Dict[str, Any]]) -> None:
        """
        Tampilkan statistik split dataset dengan tabel HTML yang konsisten.
        
        Args:
            split_stats: Dictionary statistik per split
        """
        # Styling untuk tabel
        th_style = f"padding:8px; text-align:center; border:1px solid #ddd; color:#343a40; background-color:#f8f9fa;"
        td_style = f"padding:8px; text-align:center; border:1px solid #ddd; color:{COLORS['dark']};"
        
        # Generate HTML tabel
        table_html = f"""
        <table style="width:100%; border-collapse:collapse; margin:10px 0; border:1px solid #ddd;">
            <thead>
                <tr>
                    <th style="{th_style}">Split</th>
                    <th style="{th_style}">Gambar</th>
                    <th style="{th_style}">Label</th>
                    <th style="{th_style}">Status</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Tambahkan baris untuk setiap split
        for split, stats in split_stats.items():
            # Style untuk status
            status_color = COLORS['success'] if stats.get('complete', False) else COLORS['warning']
            status_icon = ICONS['success'] if stats.get('complete', False) else ICONS['warning']
            status_text = "Lengkap" if stats.get('complete', False) else "Tidak Lengkap"
            
            # Tambahkan baris ke tabel
            table_html += f"""
            <tr>
                <td style="{td_style} font-weight:bold;">{split.capitalize()}</td>
                <td style="{td_style}">{stats.get('images', 0)}</td>
                <td style="{td_style}">{stats.get('labels', 0)}</td>
                <td style="{td_style} color:{status_color};">
                    {status_icon} {status_text}
                </td>
            </tr>
            """
        
        # Tutup tabel
        table_html += """
            </tbody>
        </table>
        """
        
        # Tampilkan tabel
        display(HTML(table_html))
    
    # Fungsi untuk generate ringkasan preprocessing dari direktori
    def generate_preprocessing_summary(preprocessed_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate ringkasan hasil preprocessing dari direktori.
        
        Args:
            preprocessed_dir: Path direktori hasil preprocessing
            
        Returns:
            Dictionary ringkasan hasil
        """
        from smartcash.ui.dataset.shared.get_preprocessing_stats import get_preprocessing_stats
        
        # Gunakan default jika tidak disediakan
        preprocessed_dir = preprocessed_dir or ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Dapatkan konfigurasi resolusi gambar
        img_size = DEFAULT_IMG_SIZE
        try:
            # Coba dapatkan dari UI
            if 'preprocess_options' in ui_components and hasattr(ui_components['preprocess_options'], 'children'):
                img_size_value = ui_components['preprocess_options'].children[0].value
                img_size = [img_size_value, img_size_value]
            # Atau dari config
            elif config and 'preprocessing' in config and 'img_size' in config['preprocessing']:
                img_size = config['preprocessing']['img_size']
        except Exception:
            pass
        
        # Hitung statistik dataset
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
    
    # Fungsi untuk generate ringkasan augmentasi dari direktori
    def generate_augmentation_summary(target_dir: Optional[str] = None, temp_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate ringkasan hasil augmentasi dari direktori.
        
        Args:
            target_dir: Path direktori hasil (biasanya preprocessed)
            temp_dir: Path direktori temporary (augmented)
            
        Returns:
            Dictionary ringkasan hasil
        """
        # Import utility untuk analisis distribusi
        from smartcash.ui.charts.class_distribution_analyzer import count_files_by_prefix
        
        # Gunakan default paths jika tidak disediakan
        target_dir = target_dir or ui_components.get('preprocessed_dir', 'data/preprocessed')
        temp_dir = temp_dir or ui_components.get('augmented_dir', 'data/augmented')
        
        # Get augmentation prefix dari UI
        aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
        orig_prefix = 'rp'  # Default preprocessed prefix
        
        # Hitung file di kedua lokasi
        target_path = Path(target_dir)
        temp_path = Path(temp_dir)
        
        # Inisialisasi counters
        orig_files = 0
        aug_files_target = 0
        aug_files_temp = 0
        
        # Hitung file original dan augmented di target_dir
        for split in DEFAULT_SPLITS:
            # Cek direktori images
            images_dir = target_path / split / 'images'
            if not images_dir.exists():
                continue
            
            # Hitung file original
            orig_files += len(list(images_dir.glob(f"{orig_prefix}_*.*")))
            
            # Hitung file augmentasi di target
            aug_files_target += len(list(images_dir.glob(f"{aug_prefix}_*.*")))
        
        # Cek juga di direktori temporary jika ada
        temp_images_dir = temp_path / 'images'
        if temp_images_dir.exists():
            aug_files_temp = len(list(temp_images_dir.glob(f"{aug_prefix}_*.*")))
        
        # Dapatkan jenis augmentasi dari UI
        aug_types = []
        if 'aug_options' in ui_components and hasattr(ui_components['aug_options'], 'children') and len(ui_components['aug_options'].children) > 0:
            # Mapping tipe UI ke nama internal
            type_map = {'Combined (Recommended)': 'combined', 'Position Variations': 'position', 
                      'Lighting Variations': 'lighting', 'Extreme Rotation': 'extreme_rotation'}
            
            selected_types = ui_components['aug_options'].children[0].value
            aug_types = [type_map.get(t, 'combined') for t in selected_types]
        
        # Buat ringkasan
        summary = {
            'original': orig_files,
            'generated': aug_files_target + aug_files_temp,
            'total_files': orig_files + aug_files_target + aug_files_temp,
            'duration': 0,  # Tidak diketahui karena loading dari disk
            'augmentation_types': aug_types,
            'output_dir': target_dir
        }
        
        return summary
    
    # Handler untuk tombol summary jika tersedia
    def on_summary_click(b) -> None:
        """Handler untuk tombol tampilkan summary."""
        # Update UI dengan indikator proses
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
            # Tampilkan error
            with ui_components['status']:
                display(create_status_indicator('error', f"{ICONS['error']} Error membuat ringkasan: {str(e)}"))
            
            if logger: logger.error(f"{ICONS['error']} Error membuat ringkasan: {str(e)}")
    
    # Register handler untuk tombol summary jika tersedia
    if 'summary_button' in ui_components:
        ui_components['summary_button'].on_click(on_summary_click)
    
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'update_summary': update_summary,
        'generate_summary': generate_preprocessing_summary if module_type == 'preprocessing' else generate_augmentation_summary,
        'on_summary_click': on_summary_click
    })
    
    return ui_components