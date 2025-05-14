"""
File: smartcash/ui/dataset/split/handlers/visualization_handler.py
Deskripsi: Handler untuk visualisasi dataset pada konfigurasi split dataset dengan memanfaatkan komponen yang sudah ada
"""

from typing import Dict, Any, Tuple, Optional
import os
from pathlib import Path
from IPython.display import display, HTML, clear_output

# Import komponen yang sudah ada
from smartcash.ui.charts.class_distribution_analyzer import analyze_class_distribution
from smartcash.ui.charts.plot_stacked import plot_class_distribution_stacked
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS

def setup_visualization_handler(ui_components: Dict[str, Any], config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """
    Setup handler untuk visualisasi dataset pada UI konfigurasi split.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Inisialisasi ui_components jika None
    if ui_components is None:
        ui_components = {}
    
    # Pastikan konfigurasi data ada
    if not config:
        config = {}
    if 'data' not in config:
        config['data'] = {}
    
    # Dapatkan logger jika tersedia
    logger = ui_components.get('logger', None)
    
    # Register handler untuk visualize button
    if 'visualize_button' in ui_components and ui_components['visualize_button']:
        ui_components['visualize_button'].on_click(
            lambda b: handle_visualize_button(b, ui_components, config, env, logger)
        )
        if logger: logger.info("🔗 Handler untuk visualize button terdaftar")
    
    # Load dan tampilkan statistik dasar dataset
    load_and_display_dataset_stats(ui_components, config, env, logger)
    
    return ui_components

def handle_visualize_button(b, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> None:
    """
    Handler untuk tombol visualisasi dataset.
    
    Args:
        b: Button widget yang dipicu
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
    """
    # Validasi komponen yang diperlukan
    if 'output_box' not in ui_components:
        if logger: logger.error("❌ Output box tidak tersedia untuk visualisasi")
        return
    
    # Tampilkan visualisasi distribusi kelas
    show_distribution_visualization(ui_components['output_box'], config, env, logger)

def show_distribution_visualization(output_widget, config: Dict[str, Any], env=None, logger=None) -> None:
    """
    Tampilkan visualisasi distribusi kelas dataset.
    
    Args:
        output_widget: Widget output untuk menampilkan visualisasi
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    with output_widget:
        clear_output(wait=True)
        display(HTML(f"""
            <div style="text-align:center; padding:15px;">
                <p style="color:{COLORS['muted']};">{ICONS['processing']} Memuat visualisasi distribusi kelas...</p>
            </div>
        """))
        
        try:
            # Import libraries yang diperlukan
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            
            # Dapatkan distribusi kelas dari dataset
            class_distribution = get_class_distribution(config, env, logger)
            
            if not class_distribution:
                display(HTML(f"""
                    <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                                color:{COLORS['alert_warning_text']}; border-radius:4px;">
                        <p>{ICONS['warning']} Dataset tidak ditemukan atau tidak memiliki label yang valid.</p>
                    </div>
                """))
                return
            
            # Dapatkan path dataset untuk ditampilkan
            dataset_path, _ = get_dataset_paths(config, env)
            
            # Plot distribusi kelas
            _plot_distribution(class_distribution, dataset_path, logger)
            
            if logger: logger.info("✅ Visualisasi distribusi kelas berhasil ditampilkan")
            
        except Exception as e:
            clear_output(wait=True)
            display(HTML(f"""
                <div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                            color:{COLORS['alert_danger_text']}; border-radius:4px;">
                    <p>{ICONS['error']} Error saat membuat visualisasi: {str(e)}</p>
                </div>
            """))
            if logger: logger.error(f"❌ Error saat membuat visualisasi: {str(e)}")

def get_dataset_paths(config: Dict[str, Any], env=None) -> Tuple[str, str]:
    """
    Dapatkan path dataset dan preprocessed dataset.
    
    Args:
        config: Konfigurasi aplikasi
        env: Environment manager
        
    Returns:
        Tuple berisi (dataset_path, preprocessed_path)
    """
    from smartcash.dataset.utils.dataset_constants import DRIVE_DATASET_PATH, DRIVE_PREPROCESSED_PATH
    
    drive_mounted = env and getattr(env, 'is_drive_mounted', False)
    base_path = '/content/drive/MyDrive/SmartCash/data' if drive_mounted else 'data'
    
    # Default paths
    dataset_path = base_path
    preprocessed_path = f'{base_path}/preprocessed'
    
    # Get from config if available
    if config and isinstance(config, dict) and 'data' in config and isinstance(config['data'], dict):
        dataset_path = config['data'].get('dataset_path', base_path)
        preprocessed_path = config['data'].get('preprocessed_path', f'{base_path}/preprocessed')
        
    return (dataset_path, preprocessed_path)

def get_class_distribution(config: Dict[str, Any], env=None, logger=None) -> Dict[str, Dict[str, int]]:
    """
    Dapatkan distribusi kelas dari dataset labels menggunakan fungsi yang sudah ada.
    
    Args:
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
        
    Returns:
        Dictionary berisi distribusi kelas per split
    """
    try:
        dataset_path, _ = get_dataset_paths(config, env)
        
        # Cek jika dataset ada
        if not os.path.exists(dataset_path):
            if logger: logger.warning(f"⚠️ Dataset path tidak ditemukan: {dataset_path}")
            return _dummy_distribution()
        
        # Dictionary untuk menyimpan distribusi kelas per split
        class_distribution = {}
        
        # Iterasi melalui setiap split menggunakan fungsi yang sudah ada
        for split in DEFAULT_SPLITS:
            # Gunakan fungsi analyze_class_distribution yang sudah ada
            split_classes = analyze_class_distribution(dataset_path, split)
            
            # Skip jika tidak ada data
            if not split_classes:
                continue
                
            # Map class IDs ke nama kelas yang lebih deskriptif
            class_distribution[split] = _map_class_names(split_classes, logger)
        
        # Jika tidak ada data yang valid, gunakan dummy data
        if not any(class_distribution.values()):
            if logger: logger.warning("⚠️ Tidak ada data kelas yang valid, menggunakan data dummy")
            return _dummy_distribution()
            
        return class_distribution
    except Exception as e:
        if logger: logger.error(f"❌ Error mendapatkan distribusi kelas: {str(e)}")
        return _dummy_distribution()

def _map_class_names(split_classes: Dict[int, int], logger=None) -> Dict[str, int]:
    """
    Map ID kelas ke notasi kode yang diformat dengan baik.
    
    Args:
        split_classes: Dictionary berisi {class_id: count}
        logger: Logger untuk logging
        
    Returns:
        Dictionary berisi {class_name: count}
    """
    try:
        # Coba load dari file kelas jika ada
        class_names = {}
        
        # Coba baca dari file data.yaml atau classes.txt
        for file_path in ['data/data.yaml', 'data/classes.txt']:
            if os.path.exists(file_path):
                if file_path.endswith('.yaml'):
                    import yaml
                    with open(file_path, 'r') as f:
                        data = yaml.safe_load(f)
                        if data and 'names' in data and isinstance(data['names'], list):
                            for i, name in enumerate(data['names']):
                                class_names[i] = f"{name} (ID: {i})"
                else:
                    with open(file_path, 'r') as f:
                        lines = f.read().splitlines()
                        for i, name in enumerate(lines):
                            class_names[i] = f"{name} (ID: {i})"
                break
        
        # Jika tidak ada file kelas, gunakan ID
        if not class_names:
            for class_id in split_classes.keys():
                class_names[class_id] = f"Class {class_id}"
        
        # Map counts dengan nama kelas
        return {class_names.get(class_id, f"Class {class_id}"): count 
                for class_id, count in split_classes.items()}
    except Exception as e:
        if logger: logger.warning(f"⚠️ Error mapping class names: {str(e)}")
        return {f"Class {class_id}": count for class_id, count in split_classes.items()}

def _dummy_distribution() -> Dict[str, Dict[str, int]]:
    """
    Kembalikan distribusi kelas dummy dengan nama kelas yang diformat dengan benar.
    
    Returns:
        Dictionary berisi distribusi kelas dummy
    """
    return {
        'train': {
            '001:0': 120, '002:1': 110, '003:2': 130, '004:3': 140, 
            '005:4': 125, '006:5': 115, '007:6': 135,
            'l2_001:7': 80, 'l2_002:8': 85, 'l2_003:9': 75,
            'l3_001:16': 60, 'l3_002:17': 55
        },
        'valid': {
            '001:0': 30, '002:1': 25, '003:2': 35, '004:3': 40, 
            '005:4': 35, '006:5': 28, '007:6': 32,
            'l2_001:7': 20, 'l2_002:8': 22, 'l2_003:9': 18,
            'l3_001:16': 15, 'l3_002:17': 12
        },
        'test': {
            '001:0': 30, '002:1': 25, '003:2': 35, '004:3': 35, 
            '005:4': 30, '006:5': 27, '007:6': 33,
            'l2_001:7': 18, 'l2_002:8': 20, 'l2_003:9': 19,
            'l3_001:16': 14, 'l3_002:17': 13
        }
    }

def _plot_distribution(class_distribution: Dict[str, Dict[str, int]], dataset_path: str, logger=None) -> None:
    """
    Plot distribusi kelas menggunakan komponen visualisasi yang sudah ada.
    
    Args:
        class_distribution: Dictionary berisi distribusi kelas per split
        dataset_path: Path ke dataset untuk ditampilkan
        logger: Logger untuk logging
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from smartcash.ui.utils.alert_utils import create_info_alert
        
        # Dapatkan semua kelas unik dari semua splits
        all_classes = set()
        for split_data in class_distribution.values():
            all_classes.update(split_data.keys())
        all_classes = sorted(list(all_classes))
        
        # Membuat dictionary yang akan dikonversi ke DataFrame
        data = {
            'Split': ['train', 'valid', 'test'],
        }
        
        # Tambahkan data untuk setiap kelas
        for cls in all_classes:
            data[cls] = [
                class_distribution.get('train', {}).get(cls, 0),
                class_distribution.get('valid', {}).get(cls, 0),
                class_distribution.get('test', {}).get(cls, 0)
            ]
        
        # Buat DataFrame dengan Split sebagai index
        df = pd.DataFrame(data).set_index('Split')
        
        # Visualisasi heatmap untuk memudahkan melihat distribusi kelas per split
        plt.figure(figsize=(14, 6))
        sns.heatmap(df, annot=True, fmt='d', cmap='Blues', linewidths=.5, cbar_kws={'label': 'Samples'})
        plt.title('Distribusi Kelas per Split Dataset')
        plt.ylabel('Split')
        plt.xlabel('Kelas')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Gunakan plot_class_distribution_stacked untuk train vs valid
        if 'train' in class_distribution and 'valid' in class_distribution:
            fig = plot_class_distribution_stacked(
                class_distribution['train'],
                class_distribution['valid'],
                title='Perbandingan Distribusi Train vs Valid',
                display_numbers=True
            )
            plt.show()
        
        # Tampilkan informasi dataset dan tabel dalam satu blok
        display(create_info_alert(
            f"<strong>Informasi Dataset:</strong> Visualisasi menunjukkan distribusi kelas pada setiap split.<br>"
            f"Path dataset: <code>{dataset_path}</code>", 'info'))
        
        # Tampilkan tabel dengan highlight heatmap
        display(HTML(f'<h3>{ICONS["chart"]} Tabel Distribusi Kelas</h3>'))
        display(df.style.background_gradient(cmap=sns.light_palette("blue", as_cmap=True)))
        
    except Exception as e:
        if logger: logger.error(f"❌ Error membuat visualisasi: {str(e)}")
        from smartcash.ui.utils.alert_utils import create_alert_html
        try:
            display(HTML(create_alert_html(f"Error membuat visualisasi: {str(e)}", "error")))
        except (ImportError, AttributeError):
            display(HTML(f'<div style="padding:10px; background-color:{COLORS["alert_danger_bg"]}; '
                        f'color:{COLORS["alert_danger_text"]}; border-radius:4px;">'
                        f'<p>{ICONS["error"]} Error membuat visualisasi: {str(e)}</p></div>'))

def load_and_display_dataset_stats(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> None:
    """
    Load dan tampilkan statistik dasar dataset.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
    """
    try:
        # Validasi input
        if ui_components is None:
            return
        
        # Tampilkan status loading
        if 'stats_container' in ui_components:
            ui_components['stats_container'].value = f"""
                <div style="text-align:center; padding:15px;">
                    <p>{ICONS['processing']} Memuat statistik dataset...</p>
                </div>
            """
        
        # Import dengan timeout untuk menghindari hanging
        import threading
        import time
        
        # Variabel untuk menyimpan hasil dan status
        result = {'stats': None, 'done': False, 'error': None}
        
        # Fungsi untuk mendapatkan statistik dengan timeout
        def get_stats_with_timeout():
            try:
                result['stats'] = get_dataset_stats(config, env, logger)
                result['done'] = True
            except Exception as e:
                result['error'] = str(e)
                result['done'] = True
        
        # Jalankan di thread terpisah
        stats_thread = threading.Thread(target=get_stats_with_timeout)
        stats_thread.daemon = True
        stats_thread.start()
        
        # Tunggu maksimal 10 detik
        timeout = 10
        start_time = time.time()
        while not result['done'] and time.time() - start_time < timeout:
            time.sleep(0.5)
        
        # Cek hasil
        if not result['done']:
            # Timeout terjadi
            if logger: logger.warning(f"⚠️ Timeout saat memuat statistik dataset setelah {timeout} detik")
            if 'stats_container' in ui_components:
                ui_components['stats_container'].value = f"""
                    <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                         color:{COLORS['alert_warning_text']}; border-radius:4px;">
                        <p>{ICONS['warning']} Timeout saat memuat statistik dataset. Silakan coba lagi nanti.</p>
                    </div>
                """
            return
        
        # Cek error
        if result['error']:
            raise Exception(result['error'])
        
        # Update UI dengan statistik
        if 'stats_container' in ui_components:
            # Selalu tampilkan kartu statistik, bahkan jika kosong
            if result['stats']:
                update_stats_cards(ui_components['stats_container'], result['stats'], COLORS, ICONS)
                if logger: logger.info("✅ Statistik dataset berhasil dimuat")
            else:
                # Buat statistik kosong dengan nilai 0 untuk ditampilkan
                empty_stats = {
                    'raw': {
                        'exists': True,
                        'stats': {split: {'images': 0, 'labels': 0, 'valid': True} for split in DEFAULT_SPLITS}
                    },
                    'preprocessed': {
                        'exists': True,
                        'stats': {split: {'images': 0, 'labels': 0, 'valid': True} for split in DEFAULT_SPLITS}
                    }
                }
                update_stats_cards(ui_components['stats_container'], empty_stats, COLORS, ICONS)
                if logger: logger.info("ℹ️ Menampilkan statistik dataset kosong dengan nilai 0")
        
    except Exception as e:
        # Handle error
        if logger: logger.error(f"❌ Error menampilkan statistik: {str(e)}")
        
        # Tampilkan pesan error
        if 'stats_container' in ui_components:
            from smartcash.ui.utils.alert_utils import create_alert_html
            try:
                ui_components['stats_container'].value = create_alert_html(f"Error menampilkan statistik: {str(e)}", "error")
            except (ImportError, AttributeError):
                ui_components['stats_container'].value = f'<div style="padding:10px; background-color:{COLORS["alert_danger_bg"]}; color:{COLORS["alert_danger_text"]}; border-radius:4px;"><p>{ICONS["error"]} Error menampilkan statistik: {str(e)}</p></div>'

def get_dataset_stats(config: Dict[str, Any], env=None, logger=None) -> Dict[str, Any]:
    """
    Dapatkan statistik dataset untuk raw dan preprocessed data.
    
    Args:
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
        
    Returns:
        Dictionary berisi statistik dataset
    """
    try:
        from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS
        
        # Default empty stats
        stats = {'raw': {'exists': False, 'stats': {}}, 'preprocessed': {'exists': False, 'stats': {}}}
        
        # Get paths safely
        if config is None: config = {}
        raw_path, preprocessed_path = get_dataset_paths(config, env)
        
        # Update stats with existence info
        stats['raw']['exists'] = os.path.exists(raw_path)
        stats['preprocessed']['exists'] = os.path.exists(preprocessed_path)
        
        # Count files if directories exist
        if stats['raw']['exists']:
            stats['raw']['stats'] = count_files(raw_path)
            if logger: logger.info(f"📊 Statistik raw dataset dihitung: {raw_path}")
            
        if stats['preprocessed']['exists']:
            stats['preprocessed']['stats'] = count_files(preprocessed_path)
            if logger: logger.info(f"📊 Statistik preprocessed dataset dihitung: {preprocessed_path}")
            
        return stats
    except Exception as e:
        if logger: logger.error(f"❌ Error mendapatkan statistik dataset: {str(e)}")
        return {'raw': {'exists': False, 'stats': {}}, 'preprocessed': {'exists': False, 'stats': {}}}

def count_files(dataset_dir: str) -> Dict[str, Dict[str, int]]:
    """
    Hitung file dalam struktur dataset YOLO.
    
    Args:
        dataset_dir: Path ke direktori dataset
        
    Returns:
        Dictionary berisi statistik file per split
    """
    from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS
    
    stats = {}
    
    # Check if directory exists
    if not os.path.exists(dataset_dir): return stats
        
    for split in DEFAULT_SPLITS:
        split_dir = Path(dataset_dir) / split
        
        # Initialize counters
        stats[split] = {'images': 0, 'labels': 0, 'valid': False}
        
        # Count files if directories exist
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if images_dir.exists(): stats[split]['images'] = len(list(images_dir.glob('*.*')))
        if labels_dir.exists(): stats[split]['labels'] = len(list(labels_dir.glob('*.txt')))
        stats[split]['valid'] = stats[split]['images'] > 0 and stats[split]['labels'] > 0
        
    return stats

def update_stats_cards(html_component, stats: Dict[str, Any], COLORS: Dict[str, str], ICONS: Dict[str, str]) -> None:
    """
    Update komponen HTML dengan kartu statistik dataset menggunakan komponen yang sudah ada.
    
    Args:
        html_component: Komponen HTML untuk diupdate
        stats: Statistik dataset
        COLORS: Dictionary warna UI
        ICONS: Dictionary ikon UI
    """
    from smartcash.ui.utils.card_utils import create_card_html
    
    # Header untuk informasi dataset
    cards_html = f'<div style="text-align:center; padding:15px;">'
    cards_html += f'<h3 style="color:{COLORS["dark"]}; margin-bottom:10px;">{ICONS["dataset"]} Informasi Dataset</h3>'
    
    # Tampilkan kartu untuk dataset raw jika ada
    if stats['raw']['exists'] and stats['raw']['stats']:
        try:
            # Gunakan create_card_html jika tersedia
            cards_html += create_card_html(title="Dataset Raw", icon=ICONS['folder'], 
                                          value=_generate_stats_table(stats['raw']['stats']),
                                          color=COLORS['card'])
        except (ImportError, AttributeError):
            # Fallback ke implementasi lokal
            cards_html += _generate_card("Dataset Raw", ICONS['folder'], COLORS['card'], stats['raw']['stats'])
    
    # Tampilkan kartu untuk dataset preprocessed jika ada
    if stats['preprocessed']['exists'] and stats['preprocessed']['stats']:
        try:
            cards_html += create_card_html(title="Dataset Preprocessed", icon=ICONS['processing'], 
                                          value=_generate_stats_table(stats['preprocessed']['stats']),
                                          color=COLORS['card'])
        except (ImportError, AttributeError):
            cards_html += _generate_card("Dataset Preprocessed", ICONS['processing'], COLORS['card'], 
                                       stats['preprocessed']['stats'])
    
    # Tampilkan pesan jika tidak ada dataset atau semua dataset kosong
    all_empty = True
    
    # Periksa apakah semua dataset kosong (0 gambar)
    if stats['raw']['exists'] and stats['raw']['stats']:
        for split_stats in stats['raw']['stats'].values():
            if split_stats.get('images', 0) > 0:
                all_empty = False
                break
                
    if stats['preprocessed']['exists'] and stats['preprocessed']['stats']:
        for split_stats in stats['preprocessed']['stats'].values():
            if split_stats.get('images', 0) > 0:
                all_empty = False
                break
    
    # Tampilkan pesan informatif jika tidak ada dataset atau semua dataset kosong
    if not (stats['raw']['exists'] or stats['preprocessed']['exists']) or all_empty:
        from smartcash.ui.utils.alert_utils import create_alert_html
        try:
            if all_empty:
                cards_html += create_alert_html(
                    message="Dataset kosong (0 gambar). Silakan lakukan preprocessing terlebih dahulu atau tambahkan gambar ke dataset.",
                    alert_type="info")
            else:
                cards_html += create_alert_html(
                    message="Dataset tidak ditemukan. Klik tombol <strong>Visualisasi Distribusi Kelas</strong> untuk melihat contoh visualisasi.",
                    alert_type="warning")
        except (ImportError, AttributeError):
            if all_empty:
                cards_html += f'<div style="padding:10px; background-color:{COLORS["alert_info_bg"]}; '\
                             f'color:{COLORS["alert_info_text"]}; border-radius:4px;">'\
                             f'<p>{ICONS["info"]} Dataset kosong (0 gambar). Silakan lakukan preprocessing terlebih dahulu atau tambahkan gambar ke dataset.</p></div>'
            else:
                cards_html += f'<div style="padding:10px; background-color:{COLORS["alert_warning_bg"]}; '\
                             f'color:{COLORS["alert_warning_text"]}; border-radius:4px;">'\
                             f'<p>{ICONS["warning"]} Dataset tidak ditemukan. Klik tombol <strong>Visualisasi '\
                             f'Distribusi Kelas</strong> untuk melihat contoh visualisasi.</p></div>'
    
    cards_html += '</div>'
    
    # Update HTML component
    html_component.value = cards_html

def _generate_stats_table(data: Dict[str, Any]) -> str:
    """Generate table HTML for dataset stats dengan format yang lebih ringkas."""
    # Style untuk semua sel tabel
    th_style = "text-align:center; padding:5px; border-bottom:1px solid #ddd;"
    td_style = "text-align:center; padding:5px;"
    td_left_style = "text-align:left; padding:5px;"
    
    # Header tabel
    table_html = f'<table style="width:100%; border-collapse:collapse;"><tr>'
    table_html += f'<th style="{td_left_style}">Split</th>'
    for header in ['Images', 'Labels', 'Status']:
        table_html += f'<th style="{th_style}">{header}</th>'
    table_html += '</tr>'
    
    # Baris data
    for split, info in data.items():
        status_icon = "✅" if info.get('valid', False) else "❌"
        table_html += f'<tr><td style="{td_left_style}">{split}</td>'
        table_html += f'<td style="{td_style}">{info.get("images", 0)}</td>'
        table_html += f'<td style="{td_style}">{info.get("labels", 0)}</td>'
        table_html += f'<td style="{td_style}">{status_icon}</td></tr>'
    
    table_html += '</table>'
    return table_html

def _generate_card(title: str, icon: str, color: str, data: Dict[str, Any]) -> str:
    """Generate card HTML for dataset stats dengan format yang lebih ringkas."""
    return f'<div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin:5px; background-color:{color};">'\
           f'<h4 style="margin-top:0; margin-bottom:10px; color:#333;">{icon} {title}</h4>'\
           f'{_generate_stats_table(data)}'\
           f'</div>'
