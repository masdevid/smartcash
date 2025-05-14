"""
File: smartcash/dataset/services/downloader/notification_utils.py
Deskripsi: Utilitas untuk menstandarisasi notifikasi observer dalam downloader service dengan penanganan error yang lebih baik
"""

from typing import Dict, Any, Optional
from smartcash.components.observer import notify, EventTopics

def notify_event(
    sender: Any,
    event_type: str,
    observer_manager=None,
    message: Optional[str] = None,
    **kwargs
) -> None:
    """
    Fungsi helper untuk mengirimkan notifikasi observer secara aman dengan penanganan error dan rekursi.
    
    Args:
        sender: Objek pengirim event
        event_type: Tipe event (gunakan konstanta dari EventTopics)
        observer_manager: Observer manager opsional (akan mencoba impor jika None)
        message: Pesan untuk disertakan dengan event
        **kwargs: Parameter tambahan untuk event
    """
    try:
        # Cek jika sudah dalam tahap notifikasi untuk mencegah rekursi
        if hasattr(sender, '_notify_in_progress') and sender._notify_in_progress:
            return
            
        # Set flag untuk mencegah rekursi
        setattr(sender, '_notify_in_progress', True)
        
        # Coba dapatkan EventTopics dari konstanta jika string
        if isinstance(event_type, str):
            try:
                event_const = getattr(EventTopics, event_type, None)
                event_type = event_const if event_const is not None else event_type
            except (ImportError, AttributeError):
                pass
                
        # Pastikan observer_manager tersedia atau gunakan notifikasi langsung
        params = {}
        if message:
            params["message"] = message
            
        # Tambahkan status jika tidak ada dalam kwargs
        if "status" not in kwargs:
            params["status"] = "info"
            
        # Tambahkan kwargs ke params
        params.update(kwargs)
        
        if observer_manager is not None:
            # Jika observer manager diberikan, gunakan itu
            observer_manager.notify(event_type, sender, **params)
        else:
            # Langsung notifikasi via fungsi global
            notify(event_type, sender, **params)
    except Exception as e:
        # Jangan mengganggu proses jika notifikasi gagal
        pass
    finally:
        # Reset flag
        setattr(sender, '_notify_in_progress', False)


# Fungsi helper untuk semua event tipe download
def notify_download(
    event_type: str,
    sender: Any,
    observer_manager=None,
    **kwargs
) -> None:
    """
    Notifikasi event download (start, progress, complete, error) dengan standardisasi parameter.
    
    Args:
        event_type: Tipe event ('start', 'progress', 'complete', 'error')
        sender: Objek pengirim event
        observer_manager: Observer manager opsional
        **kwargs: Parameter tambahan untuk event
    """
    event_mapping = {
        "start": "DOWNLOAD_START",
        "progress": "DOWNLOAD_PROGRESS",
        "complete": "DOWNLOAD_COMPLETE",
        "error": "DOWNLOAD_ERROR"
    }
    
    event_name = event_mapping.get(event_type.lower())
    if not event_name:
        return
        
    # Set status berdasarkan tipe event
    status = "error" if event_type.lower() == "error" else "success" if event_type.lower() == "complete" else "info"
    
    # Tambahkan parameter standar untuk tiap tipe event
    event_params = dict(kwargs)
    
    # Parameter khusus per tipe event
    if event_type.lower() == "start":
        event_params.setdefault("progress", 0)
        event_params.setdefault("total_steps", 5) 
        event_params.setdefault("current_step", 1)
    elif event_type.lower() == "progress":
        # Hitung persentase jika ada progress dan total
        if "progress" in event_params and "total" in event_params:
            progress = event_params["progress"]
            total = event_params["total"]
            if total > 0:
                event_params.setdefault("percentage", int((progress / total) * 100))
    elif event_type.lower() == "complete":
        # Pastikan ada parameter duration jika diperlukan untuk tracking
        if "duration" not in event_params:
            event_params.setdefault("duration", 0)
    
    notify_event(sender, event_name, observer_manager, status=status, **event_params)

# Fungsi helper terintegrasi untuk semua tipe event dari semua layanan
def notify_service_event(
    event_category: str,
    event_type: str,
    sender: Any,
    observer_manager=None,
    **kwargs
) -> None:
    """
    Notifikasi event dari berbagai kategori layanan dengan standardisasi.
    
    Args:
        event_category: Kategori event ('download', 'export', 'backup', 'zip_processing', 'pull_dataset')
        event_type: Tipe event ('start', 'progress', 'complete', 'error')
        sender: Objek pengirim event
        observer_manager: Observer manager opsional
        **kwargs: Parameter tambahan untuk event
    """
    category_mapping = {
        "download": "DOWNLOAD",
        "export": "EXPORT",
        "backup": "BACKUP",
        "zip_processing": "ZIP_PROCESSING",
        "pull_dataset": "PULL_DATASET",
        "zip_import": "ZIP_IMPORT",
        "upload": "UPLOAD",
    }
    
    type_mapping = {
        "start": "START",
        "progress": "PROGRESS",
        "complete": "COMPLETE",
        "error": "ERROR",
    }
    
    category = category_mapping.get(event_category.lower())
    event = type_mapping.get(event_type.lower())
    
    if category and event:
        event_name = f"{category}_{event}"
        status = "error" if event_type.lower() == "error" else "success" if event_type.lower() == "complete" else "info"
        
        # Tambahkan parameter standar untuk tiap tipe event
        event_params = dict(kwargs)
        
        # Parameter khusus per tipe event
        if event_type.lower() == "start":
            event_params.setdefault("progress", 0)
        elif event_type.lower() == "progress":
            # Hitung persentase jika ada progress dan total
            if "progress" in event_params and "total" in event_params:
                progress = event_params["progress"]
                total = event_params["total"]
                if total > 0:
                    event_params.setdefault("percentage", int((progress / total) * 100))
        
        # Panggil notify_event TANPA parameter duplikat status
        notify_event(sender, event_name, observer_manager, **event_params)