"""
File: smartcash/smartcash/ui_handlers/common_utils.py
Author: Alfrida Sabar
Deskripsi: Utilitas umum untuk handler UI yang digunakan di beberapa modul.
           Menggabungkan fungsi yang tumpang tindih dan meningkatkan error handling.
"""

import os
import pickle
import torch
import yaml
import gc
import matplotlib.pyplot as plt
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

@contextmanager
def memory_manager():
    """
    Context manager untuk mengoptimalkan penggunaan memori.
    
    Menggunakan try/finally pattern untuk memastikan cleaning up resources
    bahkan ketika exceptions muncul.
    """
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def is_colab() -> bool:
    """
    Deteksi apakah kode dijalankan di Google Colab.
    
    Returns:
        Boolean yang menunjukkan apakah di Google Colab
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False

def save_config(config: Dict[str, Any], 
                filename: str = 'experiment_config.yaml',
                create_pickle: bool = True,
                logger: Optional[Any] = None) -> bool:
    """
    Simpan konfigurasi ke file yaml dan pickle dengan error handling yang lebih baik.
    
    Args:
        config: Dictionary konfigurasi
        filename: Nama file untuk menyimpan konfigurasi yaml
        create_pickle: Flag untuk membuat file pickle
        logger: Optional logger untuk pesan log
        
    Returns:
        Boolean yang menunjukkan keberhasilan
    """
    try:
        # Pastikan direktori configs ada
        os.makedirs('configs', exist_ok=True)
        
        # Simpan ke yaml
        config_path = os.path.join('configs', filename)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Simpan ke pickle jika diminta
        if create_pickle:
            with open('config.pkl', 'wb') as f:
                pickle.dump(config, f)
        
        if logger:
            logger.info(f"📝 Konfigurasi disimpan ke {config_path}")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
        return False

def load_config(filename: Optional[str] = None,
               fallback_to_pickle: bool = True,
               default_config: Optional[Dict[str, Any]] = None,
               logger: Optional[Any] = None) -> Dict[str, Any]:
    """
    Muat konfigurasi dari file yaml atau pickle dengan prioritas yang jelas.
    
    Args:
        filename: Nama file konfigurasi (optional)
        fallback_to_pickle: Flag untuk memuat dari pickle jika yaml tidak ada
        default_config: Konfigurasi default jika tidak ada file yang ditemukan
        logger: Optional logger untuk pesan log
        
    Returns:
        Dictionary konfigurasi
    """
    config = {}
    
    # Definisikan file yang akan dicoba dimuat
    files_to_try = []
    if filename:
        # If a full path is provided
        if os.path.isabs(filename) or '/' in filename:
            files_to_try.append(filename)
        else:
            # If just a filename, append to configs directory
            files_to_try.append(os.path.join('configs', filename))
    
    # Add default files to try
    files_to_try.extend([
        'configs/experiment_config.yaml',
        'configs/training_config.yaml',
        'configs/base_config.yaml'
    ])
    
    # Coba memuat dari file yaml
    for file_path in files_to_try:
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    config = yaml.safe_load(f)
                if logger:
                    logger.info(f"📝 Konfigurasi dimuat dari {file_path}")
                return config
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Gagal memuat konfigurasi dari {file_path}: {str(e)}")
    
    # Coba memuat dari pickle jika fallback_to_pickle=True
    if fallback_to_pickle and os.path.exists('config.pkl'):
        try:
            with open('config.pkl', 'rb') as f:
                config = pickle.load(f)
            if logger:
                logger.info("📝 Konfigurasi dimuat dari config.pkl")
            return config
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Gagal memuat konfigurasi dari config.pkl: {str(e)}")
    
    # Gunakan konfigurasi default jika semua gagal
    if default_config:
        if logger:
            logger.warning("⚠️ Menggunakan konfigurasi default")
        return default_config
    
    # Jika tidak ada default_config dan semua gagal, kembalikan dictionary kosong
    if logger:
        logger.warning("⚠️ Tidak ada konfigurasi yang dimuat, mengembalikan dictionary kosong")
    return {}

def display_gpu_info(logger: Optional[Any] = None) -> Dict[str, Any]:
    """
    Tampilkan informasi GPU dengan format yang konsisten.
    
    Args:
        logger: Optional logger untuk pesan log
    
    Returns:
        Dictionary berisi informasi GPU atau status CPU
    """
    gpu_info = {'available': torch.cuda.is_available()}
    
    if gpu_info['available']:
        # Tambahkan informasi GPU
        gpu_info.update({
            'name': torch.cuda.get_device_name(0),
            'memory_allocated': torch.cuda.memory_allocated() / (1024**3),
            'memory_reserved': torch.cuda.memory_reserved() / (1024**3),
            'total_memory': torch.cuda.get_device_properties(0).total_memory / (1024**3)
        })
        
        # Log info jika logger tersedia
        if logger:
            logger.info(f"🖥️ GPU: {gpu_info['name']}")
            logger.info(f"🖥️ VRAM: {gpu_info['memory_allocated']:.2f}GB / {gpu_info['total_memory']:.2f}GB")
        
        # Optimize CUDA settings
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
    else:
        if logger:
            logger.info("🖥️ GPU tidak tersedia, menggunakan CPU")
    
    return gpu_info

def create_timestamp_filename(base_name: str, ext: str = 'yaml') -> str:
    """
    Buat nama file dengan timestamp untuk memastikan keunikan.
    
    Args:
        base_name: Nama dasar file
        ext: Ekstensi file
        
    Returns:
        Nama file dengan timestamp
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp}.{ext}"

def plot_metrics(metrics: Dict[str, List], 
                title: str = 'Training Metrics', 
                figsize: Tuple[int, int] = (12, 6)):
    """
    Plot metrics dalam bentuk grafik dengan styling yang konsisten.
    
    Args:
        metrics: Dictionary berisi metrics untuk diplot
        title: Judul plot
        figsize: Ukuran gambar
    """
    plt.figure(figsize=figsize)
    
    # Validasi metrics sebelum plotting
    valid_metrics = {}
    for key, values in metrics.items():
        if isinstance(values, list) and len(values) > 0:
            valid_metrics[key] = values
    
    if not valid_metrics:
        plt.text(0.5, 0.5, 'Tidak ada data metric yang valid', 
                ha='center', va='center', fontsize=14)
        plt.title(title)
        plt.tight_layout()
        plt.show()
        return
    
    # Plot valid metrics
    for key, values in valid_metrics.items():
        if key.lower() == 'epochs':
            continue  # Skip 'epochs' key, used as x-axis below
        
        # Choose x-axis: epochs if available, otherwise use index
        if 'epochs' in valid_metrics and len(valid_metrics['epochs']) == len(values):
            plt.plot(valid_metrics['epochs'], values, 'o-', label=key)
        else:
            plt.plot(range(1, len(values) + 1), values, 'o-', label=key)
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def validate_ui_components(ui_components: Dict[str, Any], 
                         required_components: List[str], 
                         logger: Optional[Any] = None) -> bool:
    """
    Validate that required UI components exist.
    
    Args:
        ui_components: Dictionary of UI components
        required_components: List of required component names
        logger: Optional logger for logging errors
        
    Returns:
        Boolean indicating if all required components are present
    """
    missing_components = [comp for comp in required_components if comp not in ui_components]
    
    if missing_components:
        error_msg = f"Missing UI components: {', '.join(missing_components)}"
        if logger:
            logger.error(f"❌ {error_msg}")
        else:
            print(f"❌ {error_msg}")
        return False
    
    return True