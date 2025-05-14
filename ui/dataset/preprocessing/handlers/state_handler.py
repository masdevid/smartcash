"""
File: smartcash/ui/dataset/preprocessing/handlers/state_handler.py
Deskripsi: Handler state untuk mendeteksi dan mengelola state preprocessing dataset
"""

from typing import Dict, Any
from pathlib import Path
from IPython.display import display, HTML, clear_output
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS

def detect_preprocessing_state(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deteksi state preprocessing dan update UI sesuai keadaan dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    try:
        # Get paths dari UI components
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        # Cek apakah preprocessed_dir sudah ada
        preprocessed_path = Path(preprocessed_dir)
        if not preprocessed_path.exists():
            if logger: logger.info(f"ℹ️ Direktori preprocessing belum ada: {preprocessed_dir}")
            return ui_components
        
        # Cari file di split train/valid/test
        has_processed_data = False
        image_count = 0
        stats = {'total': {'images': 0, 'labels': 0}, 'splits': {}}
        
        # Cek setiap split
        for split in DEFAULT_SPLITS:
            split_dir = preprocessed_path / split
            has_split = split_dir.exists()
            
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            # Count image files di split
            split_images = len(list(images_dir.glob('*.jpg'))) if has_split and images_dir.exists() else 0
            split_labels = len(list(labels_dir.glob('*.txt'))) if has_split and labels_dir.exists() else 0
            
            # Update statistik
            image_count += split_images
            stats['total']['images'] += split_images
            stats['total']['labels'] += split_labels
            stats['splits'][split] = {
                'exists': has_split,
                'images': split_images,
                'labels': split_labels,
                'complete': split_images > 0 and split_labels > 0 and split_images == split_labels
            }
            
            # Cek hasil processing
            if split_images > 0:
                has_processed_data = True
        
        # Update UI status jika preprocessing sudah dilakukan
        if has_processed_data:
            # Tampilkan status informasi
            from smartcash.ui.handlers.status_handler import update_status_panel
            update_status_panel(
                ui_components, 
                "info", 
                f"{ICONS['info']} Ditemukan {image_count} gambar terproses di {preprocessed_dir}"
            )
            
            # Tampilkan summary dengan status data
            generate_preprocessing_summary(ui_components, preprocessed_dir, stats)
            
            # Tampilkan tombol-tombol terkait data
            if 'cleanup_button' in ui_components:
                ui_components['cleanup_button'].layout.display = 'block'
                
            if 'visualization_buttons' in ui_components:
                ui_components['visualization_buttons'].layout.display = 'flex'
            
            if 'summary_container' in ui_components:
                ui_components['summary_container'].layout.display = 'block'
            
            if logger:
                logger.info(f"✅ Preprocessing state berhasil dideteksi: {image_count} gambar terproses")
        
        # Simpan status preprocessing di ui_components
        ui_components['preprocessing_stats'] = stats
        ui_components['preprocessing_done'] = has_processed_data
        
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Error saat deteksi preprocessing state: {str(e)}")
    
    return ui_components

def generate_preprocessing_summary(ui_components: Dict[str, Any], preprocessed_dir: str, stats: Dict[str, Any] = None) -> None:
    """
    Generate dan tampilkan summary preprocessing dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        preprocessed_dir: Direktori dataset preprocessed
        stats: Statistik preprocessing (opsional)
    """
    summary_container = ui_components.get('summary_container')
    if not summary_container:
        return
    
    try:
        # Clear summary container
        with summary_container:
            clear_output(wait=True)
            
            # Jika stats tidak disediakan, dapatkan dari ui_components atau buat baru
            if not stats:
                stats = ui_components.get('preprocessing_stats')
                
            if not stats:
                stats = get_preprocessing_stats(ui_components, preprocessed_dir)
                ui_components['preprocessing_stats'] = stats
            
            # Tampilkan summary dengan format lebih baik
            display(HTML(f"""
            <h3 style="color:{COLORS['dark']}">📊 Preprocessing Summary</h3>
            <div style="padding:10px; background:{COLORS['light']}; border-radius:5px; margin-bottom:10px;">
                <p><strong>📂 Direktori:</strong> {preprocessed_dir}</p>
                <p><strong>🖼️ Total Gambar:</strong> {stats['total']['images']}</p>
                <p><strong>🏷️ Total Label:</strong> {stats['total']['labels']}</p>
                <p><strong>✅ Status:</strong> {
                    "Siap digunakan" if stats['valid'] else "Belum lengkap"}</p>
            </div>
            
            <h4 style="color:{COLORS['dark']}">📊 Detail Split</h4>
            <table style="width:100%; border-collapse:collapse; margin-bottom:20px;">
                <tr style="background:{COLORS['header_bg']};">
                    <th style="padding:8px; text-align:left; border:1px solid #ddd;">Split</th>
                    <th style="padding:8px; text-align:left; border:1px solid #ddd;">Status</th>
                    <th style="padding:8px; text-align:left; border:1px solid #ddd;">Gambar</th>
                    <th style="padding:8px; text-align:left; border:1px solid #ddd;">Label</th>
                </tr>
            """))
            
            # Tampilkan setiap split
            for split, info in stats['splits'].items():
                status = "✅ Lengkap" if info.get('complete', False) else "❌ Tidak Lengkap"
                if not info.get('exists', False):
                    status = "⚠️ Tidak Ada"
                
                display(HTML(f"""
                <tr>
                    <td style="padding:8px; text-align:left; border:1px solid #ddd;">{split}</td>
                    <td style="padding:8px; text-align:left; border:1px solid #ddd;">{status}</td>
                    <td style="padding:8px; text-align:left; border:1px solid #ddd;">{info.get('images', 0)}</td>
                    <td style="padding:8px; text-align:left; border:1px solid #ddd;">{info.get('labels', 0)}</td>
                </tr>
                """))
            
            display(HTML("</table>"))
            
            # Tambahkan informasi tentang langkah selanjutnya
            if stats['valid']:
                display(HTML(f"""
                <div style="padding:10px; background:{COLORS['alert_success_bg']}; color:{COLORS['alert_success_text']}; 
                            border-radius:5px; margin-top:10px; border-left:4px solid {COLORS['alert_success_text']};">
                    <p><strong>{ICONS['success']} Langkah Selanjutnya:</strong> Dataset siap untuk augmentasi atau training.</p>
                </div>
                """))
            else:
                display(HTML(f"""
                <div style="padding:10px; background:{COLORS['alert_warning_bg']}; color:{COLORS['alert_warning_text']}; 
                            border-radius:5px; margin-top:10px; border-left:4px solid {COLORS['alert_warning_text']};">
                    <p><strong>{ICONS['warning']} Perhatian:</strong> Dataset belum lengkap. Jalankan preprocessing untuk semua split.</p>
                </div>
                """))
    
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.error(f"❌ Error saat generate summary: {str(e)}")
        
        with summary_container:
            display(HTML(f"""
            <div style="padding:10px; background:{COLORS['alert_danger_bg']}; color:{COLORS['alert_danger_text']}; 
                      border-radius:5px; border-left:4px solid {COLORS['alert_danger_text']};">
                <p><strong>{ICONS['error']} Error:</strong> Gagal menampilkan summary: {str(e)}</p>
            </div>
            """))

def get_preprocessing_stats(ui_components: Dict[str, Any], preprocessed_dir: str) -> Dict[str, Any]:
    """
    Mendapatkan statistik dataset preprocessing dengan pendekatan one-liner.
    
    Args:
        ui_components: Dictionary komponen UI
        preprocessed_dir: Direktori dataset preprocessed
        
    Returns:
        Dictionary statistik preprocessing
    """
    # Inisialisasi struktur statistik
    stats = {
        'splits': {},
        'total': {'images': 0, 'labels': 0}
    }
    
    # Cek setiap split dengan one-liner untuk deteksi valid/invalid
    for split in DEFAULT_SPLITS:
        split_dir = Path(preprocessed_dir) / split
        if not split_dir.exists():
            stats['splits'][split] = {'exists': False, 'images': 0, 'labels': 0}
            continue
            
        # List comprehension untuk hitung files dengan multiple format support
        images_dir, labels_dir = split_dir / 'images', split_dir / 'labels'
        
        # Count images dengan one-liner dan support multi-format
        image_extensions = ['.jpg', '.png', '.npy']
        num_images = sum([len(list(images_dir.glob(f"*{ext}"))) for ext in image_extensions]) if images_dir.exists() else 0
        
        # Count labels
        num_labels = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
        
        # Update statistik
        stats['splits'][split] = {
            'exists': True,
            'images': num_images,
            'labels': num_labels,
            'complete': num_images > 0 and num_labels > 0 and num_images == num_labels
        }
        
        # Update total
        stats['total']['images'] += num_images
        stats['total']['labels'] += num_labels
    
    # Dataset dianggap valid jika minimal ada 1 split dengan data lengkap (one-liner)
    stats['valid'] = any(split_info.get('complete', False) for split_info in stats['splits'].values())
    
    return stats