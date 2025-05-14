"""
File: smartcash/common/drive_sync.py
Deskripsi: Utilitas untuk sinkronisasi file dengan Google Drive
"""

import os
import shutil
from pathlib import Path
from IPython.display import display, HTML
from tqdm.auto import tqdm

# Cek apakah berjalan di Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def sync_from_drive(source_dir: str, target_dir: str, create_dirs: bool = True):
    """
    Sinkronisasi file dari Google Drive ke direktori lokal.
    
    Args:
        source_dir: Path direktori sumber di Google Drive
        target_dir: Path direktori target di lokal
        create_dirs: Buat direktori jika belum ada
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    if not IN_COLAB:
        print("⚠️ Bukan di lingkungan Google Colab, melewati sinkronisasi Drive")
        return False
    
    try:
        from google.colab import drive
        
        # Mount Google Drive jika belum
        if not os.path.exists('/content/drive'):
            print("🔄 Mounting Google Drive...")
            drive.mount('/content/drive')
        
        # Buat direktori jika belum ada
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        
        if create_dirs:
            source_path.mkdir(parents=True, exist_ok=True)
            target_path.mkdir(parents=True, exist_ok=True)
        
        # Cek apakah ada file di Drive
        if any(source_path.iterdir()):
            print(f"📂 File ditemukan di {source_dir}, menyinkronkan...")
            
            # Salin file dari Drive ke lokal
            for file_path in tqdm(list(source_path.glob('*')), desc="📥 Menyalin dari Drive"):
                target_file = target_path / file_path.name
                if not target_file.exists() or os.path.getsize(file_path) != os.path.getsize(target_file):
                    shutil.copy2(file_path, target_file)
            
            print("✅ Sinkronisasi dari Drive ke lokal selesai")
            return True
        else:
            print(f"ℹ️ Tidak ada file di {source_dir}, akan disinkronkan setelah ada file baru")
            return True
            
    except Exception as e:
        print(f"❌ Error saat sinkronisasi dengan Drive: {str(e)}")
        return False

def sync_to_drive(source_dir: str, target_dir: str, files_info: dict = None, show_success_message: bool = True):
    """
    Sinkronkan file dari direktori lokal ke Google Drive.
    
    Args:
        source_dir: Path direktori sumber di lokal
        target_dir: Path direktori target di Google Drive
        files_info: Informasi file yang akan disinkronkan (opsional)
        show_success_message: Tampilkan pesan sukses HTML
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    if not IN_COLAB:
        print("⚠️ Bukan di lingkungan Google Colab, melewati sinkronisasi Drive")
        return False
        
    try:
        # Pastikan Drive sudah di-mount
        if not os.path.exists('/content/drive'):
            from google.colab import drive
            drive.mount('/content/drive')
        
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Salin file dari lokal ke Drive
        print(f"🔄 Menyinkronkan file ke {target_dir}...")
        
        # Jika ada informasi file spesifik
        if files_info:
            for file_name, file_data in files_info.items():
                file_path = Path(file_data.get('path', ''))
                if file_path.exists():
                    target_file = target_path / file_path.name
                    
                    # Cek apakah perlu disalin (file tidak ada atau ukuran berbeda)
                    if not target_file.exists() or os.path.getsize(file_path) != os.path.getsize(target_file):
                        print(f"📤 Menyalin {file_path.name} ke Drive...")
                        shutil.copy2(file_path, target_file)
                        print(f"✅ {file_path.name} berhasil disimpan di Drive")
        else:
            # Salin semua file dari direktori sumber
            for file_path in source_path.glob('*'):
                if file_path.is_file():
                    target_file = target_path / file_path.name
                    
                    # Cek apakah perlu disalin
                    if not target_file.exists() or os.path.getsize(file_path) != os.path.getsize(target_file):
                        print(f"📤 Menyalin {file_path.name} ke Drive...")
                        shutil.copy2(file_path, target_file)
            
        print("✅ Sinkronisasi ke Drive selesai")
        
        # Tampilkan informasi sukses
        if show_success_message:
            display(HTML(f"""
            <div style='padding:10px;background-color:#e8f5e9;border-left:4px solid #4caf50;margin:10px 0;'>
                <h3 style='margin-top:0'>✅ File berhasil disinkronkan ke Google Drive</h3>
                <p>File tersimpan di: <code>{target_dir}</code></p>
                <p>File akan tersedia saat Colab reset tanpa perlu download ulang.</p>
            </div>
            """))
        
        return True
        
    except Exception as e:
        print(f"❌ Error saat sinkronisasi ke Drive: {str(e)}")
        return False

def sync_models_with_drive(models_dir: str = '/content/models', drive_dir: str = '/content/drive/MyDrive/SmartCash/models', model_info: dict = None):
    """
    Sinkronisasi model dengan Google Drive untuk memastikan persistensi saat Colab reset.
    
    Args:
        models_dir: Direktori lokal untuk model
        drive_dir: Direktori di Drive untuk model
        model_info: Informasi model (opsional)
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    # Sinkronisasi dari Drive ke lokal terlebih dahulu
    sync_from_drive(drive_dir, models_dir)
    
    # Jika ada model_info, sinkronisasi ke Drive
    if model_info:
        return sync_to_drive(models_dir, drive_dir, model_info.get('models', {}))
    
    return True
