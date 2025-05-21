"""
File: smartcash/ui/dataset/download/handlers/check_handler.py
Deskripsi: Handler untuk pengecekan status dan validasi dataset
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from IPython.display import display
from smartcash.ui.dataset.download.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.download.utils.ui_state_manager import disable_buttons, update_status_panel
from smartcash.ui.dataset.download.utils.progress_manager import show_progress, update_progress

def handle_check_button_click(ui_components: Dict[str, Any], b: Any = None) -> None:
    """
    Handler untuk tombol cek status dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        b: Button widget (opsional)
    """
    # Setup logger jika belum
    ui_components = setup_ui_logger(ui_components)

    # Reset log output saat tombol diklik
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        ui_components['log_output'].clear_output(wait=True)
    
    try:
        # Nonaktifkan tombol selama proses
        disable_buttons(ui_components, True)
        
        # Dapatkan output directory dari UI
        output_dir = ui_components.get('output_dir', {}).value or 'data'
        
        # Tampilkan progress
        show_progress(ui_components, "Memeriksa dataset...")
        
        # Jalankan check langsung (tanpa threading untuk kompatibilitas Colab)
        stats, message = _check_dataset_status(ui_components, output_dir)
        
        # Tampilkan hasil dengan visualisasi
        _display_check_results(ui_components, stats, message)
    
    except Exception as e:
        # Tampilkan error
        error_msg = f"Error saat memeriksa dataset: {str(e)}"
        log_message(ui_components, error_msg, "error", "❌")
    
    finally:
        # Aktifkan kembali tombol
        disable_buttons(ui_components, False)

def _check_dataset_status(ui_components: Dict[str, Any], output_dir: str) -> Tuple[Dict[str, Any], str]:
    """
    Cek status dataset dalam direktori.
    
    Args:
        ui_components: Dictionary komponen UI
        output_dir: Direktori output dataset
        
    Returns:
        Tuple (stats, message)
    """
    # Stats untuk report
    stats = {
        'total_images': 0,
        'total_labels': 0,
        'train': {'images': 0, 'labels': 0},
        'valid': {'images': 0, 'labels': 0},
        'test': {'images': 0, 'labels': 0},
        'classes': {},
        'missing_labels': 0,
        'empty_labels': 0
    }
    
    # Update progress
    update_progress(ui_components, 10, "Memeriksa struktur direktori...")
    
    # Cek apakah direktori ada
    base_dir = Path(output_dir)
    if not base_dir.exists():
        return stats, f"Direktori dataset tidak ditemukan: {output_dir}"
    
    # Struktur direktori yang diharapkan (untuk YOLO)
    expected_dirs = ['train/images', 'train/labels', 'valid/images', 'valid/labels', 'test/images', 'test/labels']
    for idx, subdir in enumerate(expected_dirs):
        full_path = base_dir / subdir
        
        # Update progress untuk setiap subdirektori
        progress = 10 + ((idx + 1) * 10)
        update_progress(ui_components, progress, f"Memeriksa {subdir}...")
        
        if not full_path.exists():
            log_message(ui_components, f"Direktori {subdir} tidak ditemukan", "warning", "⚠️")
            continue
        
        # Hitung file dalam direktori
        split, file_type = subdir.split('/')
        files = list(full_path.glob('*.*'))
        count = len(files)
        
        # Update stats
        if file_type == 'images':
            stats[split]['images'] = count
            stats['total_images'] += count
        elif file_type == 'labels':
            stats[split]['labels'] = count
            stats['total_labels'] += count
            
            # Analisis label jika ada
            _analyze_labels(files, stats)
    
    # Format message berdasarkan hasil
    if stats['total_images'] == 0:
        message = "Dataset masih kosong, silahkan mulai download dataset terlebih dahulu"
    else:
        message = f"Dataset ditemukan: {stats['total_images']} gambar, {stats['total_labels']} label"
    
    return stats, message

def _analyze_labels(label_files: list, stats: Dict[str, Any]) -> None:
    """
    Analisis file label untuk mendapatkan statistik kelas.
    
    Args:
        label_files: List path file label
        stats: Dictionary statistik untuk diupdate
    """
    for file_path in label_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                
                # Tandai label kosong
                if not content:
                    stats['empty_labels'] += 1
                    continue
                
                # Analisis kelas dalam label
                for line in content.splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                        
                    # Format YOLO: class_id x y width height
                    class_id = parts[0]
                    stats['classes'][class_id] = stats['classes'].get(class_id, 0) + 1
        except Exception:
            # Abaikan file yang tidak dapat dibaca
            pass
    
    # Hitung missing labels (gambar tanpa label)
    stats['missing_labels'] = stats['total_images'] - stats['total_labels']

def _display_check_results(ui_components: Dict[str, Any], stats: Dict[str, Any], message: str) -> None:
    """
    Tampilkan hasil pengecekan dataset dalam UI.
    
    Args:
        ui_components: Dictionary komponen UI
        stats: Statistik dataset
        message: Pesan status
    """
    # Update status panel
    level = "warning" if stats['total_images'] == 0 else "success"
    icon = "⚠️" if stats['total_images'] == 0 else "✅"
    update_status_panel(ui_components, message, level)
    
    # Update progress
    update_progress(ui_components, 100, "Pengecekan selesai")
    
    # Tampilkan statistik dalam summary container
    summary_container = ui_components.get('summary_container')
    if summary_container:
        summary_container.clear_output()
        summary_container.layout.display = 'block'
        
        # Tampilkan statistik dalam container
        with summary_container:
            display_dataset_stats(stats)
    
    # Tampilkan statistik dalam output status jika summary container tidak tersedia
    status_output = ui_components.get('log_output') or ui_components.get('status')
    if status_output and not ui_components.get('summary_container'):
        with status_output:
            try:
                # Tampilkan statistik dengan tabel dan chart
                display_dataset_stats(stats)
            except Exception as e:
                # Fallback ke tampilan teks
                from IPython.display import HTML
                display(HTML(f"<p style='color:red'>Error saat visualisasi: {str(e)}</p>"))
                display(HTML(f"<pre>{stats}</pre>"))
                log_message(ui_components, f"Error saat visualisasi statistik: {str(e)}", "error", "❌")

def display_dataset_stats(stats: Dict[str, Any]) -> None:
    """
    Tampilkan statistik dataset dengan visualisasi.
    
    Args:
        stats: Statistik dataset
    """
    import pandas as pd
    from IPython.display import display, HTML
    
    # Buat tampilan header
    display(HTML("<h3>📊 Statistik Dataset</h3>"))
    
    # Statistik split
    splits = ['train', 'valid', 'test']
    split_data = {
        'Split': splits,
        'Images': [stats[split]['images'] for split in splits],
        'Labels': [stats[split]['labels'] for split in splits],
        'Rasio (%)': [
            round(stats[split]['labels'] / stats[split]['images'] * 100, 1) if stats[split]['images'] > 0 else 0 
            for split in splits
        ]
    }
    split_df = pd.DataFrame(split_data)
    
    # Tampilkan tabel
    display(HTML("<h4>🗃️ Split Dataset</h4>"))
    display(split_df)
    
    # Tampilkan statistik kelas jika ada
    if stats['classes']:
        display(HTML("<h4>🏷️ Distribusi Kelas</h4>"))
        
        # Convert kelas ke DataFrame
        classes_df = pd.DataFrame({
            'Class ID': list(stats['classes'].keys()),
            'Count': list(stats['classes'].values())
        }).sort_values('Count', ascending=False)
        
        display(classes_df)
        
        # Visualisasi dengan bar chart jika tersedia
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 5))
            plt.bar(classes_df['Class ID'], classes_df['Count'])
            plt.title('Distribusi Kelas')
            plt.xlabel('Class ID')
            plt.ylabel('Jumlah')
            plt.tight_layout()
            plt.show()
        except ImportError:
            display(HTML("<p>Matplotlib tidak tersedia, visualisasi bar chart tidak dapat ditampilkan</p>"))