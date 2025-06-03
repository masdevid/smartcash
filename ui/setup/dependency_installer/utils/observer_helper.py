"""
File: smartcash/ui/setup/dependency_installer/utils/observer_helper.py
Deskripsi: Helper functions untuk observer notifications dengan silent fail dan pendekatan DRY
"""

from typing import Dict, Any, List, Optional
import time

def notify_observer(observer_manager: Any, event_type: str, data: Optional[Dict[str, Any]] = None, 
                   message: Optional[str] = None, timestamp: Optional[float] = None) -> None:
    """
    Notifikasi observer dengan silent fail dan pendekatan one-liner
    
    Args:
        observer_manager: Observer manager yang akan dinotifikasi
        event_type: Tipe event untuk notifikasi
        data: Data tambahan untuk event
        message: Pesan notifikasi
        timestamp: Timestamp untuk event, default menggunakan time.time()
    """
    if not observer_manager or not hasattr(observer_manager, 'notify'):
        return
    
    # Buat payload dengan data yang diberikan atau default
    payload = data or {}
    
    # Tambahkan message dan timestamp jika diberikan
    if message:
        payload['message'] = message
    
    if timestamp is None:
        payload['timestamp'] = time.time()
    else:
        payload['timestamp'] = timestamp
    
    # Notifikasi dengan silent fail
    try:
        observer_manager.notify(event_type, None, payload)
    except Exception:
        pass  # Silent fail untuk observer notification

def notify_install_start(ui_components: Dict[str, Any], total_packages: int, message: Optional[str] = None) -> None:
    """
    Notify observer tentang start instalasi dengan silent fail
    
    Args:
        ui_components: UI components yang berisi observer_manager
        total_packages: Jumlah total package yang akan diinstall
        message: Pesan notifikasi (opsional)
    """
    observer_manager = ui_components.get('observer_manager')
    if not observer_manager or not hasattr(observer_manager, 'notify'):
        return
    
    # Default message jika tidak disediakan
    if not message:
        message = f"Memulai instalasi {total_packages} package"
    
    # Notify dengan silent fail
    try:
        observer_manager.notify('DEPENDENCY_INSTALL_START', None, {
            'message': message,
            'timestamp': time.time(),
            'total_packages': total_packages
        })
    except Exception:
        pass  # Silent fail untuk observer notification

def notify_install_progress(ui_components: Dict[str, Any], package: str, index: int, total: int) -> None:
    """
    Notifikasi progress instalasi package dengan pendekatan one-liner
    
    Args:
        ui_components: UI components yang berisi observer_manager
        package: Nama package yang sedang diinstall
        index: Index package saat ini (mulai dari 0)
        total: Jumlah total package
    """
    observer_manager = ui_components.get('observer_manager')
    progress_pct = int(((index + 1) / total) * 100) if total > 0 else 0
    
    notify_observer(
        observer_manager=observer_manager,
        event_type='DEPENDENCY_INSTALL_PROGRESS',
        data={
            'package': package,
            'index': index,
            'total': total,
            'progress': progress_pct
        },
        message=f"Menginstall package {index+1}/{total}: {package}"
    )

def notify_install_error(ui_components: Dict[str, Any], package: str, error: str) -> None:
    """
    Notifikasi error instalasi package dengan pendekatan one-liner
    
    Args:
        ui_components: UI components yang berisi observer_manager
        package: Nama package yang gagal diinstall
        error: Pesan error
    """
    observer_manager = ui_components.get('observer_manager')
    
    notify_observer(
        observer_manager=observer_manager,
        event_type='DEPENDENCY_INSTALL_ERROR',
        data={
            'package': package,
            'error': error
        },
        message=f"Gagal install {package}: {error}"
    )

def notify_install_complete(ui_components: Dict[str, Any], stats: Dict[str, Any]) -> None:
    """
    Notify observer tentang complete instalasi dengan silent fail
    
    Args:
        ui_components: UI components yang berisi observer_manager
        stats: Statistik hasil instalasi
    """
    observer_manager = ui_components.get('observer_manager')
    if not observer_manager or not hasattr(observer_manager, 'notify'):
        return
    
    # Format pesan
    message = f"Instalasi selesai: {stats['success']}/{stats['total']} berhasil, {stats['failed']} gagal ({stats['duration']:.1f}s)"
    
    # Notify dengan silent fail
    try:
        observer_manager.notify('DEPENDENCY_INSTALL_COMPLETE', None, {
            'message': message,
            'timestamp': time.time(),
            'duration': stats['duration'],
            'success': stats['success'],
            'failed': stats['failed'],
            'total': stats['total'],
            'stats': stats
        })
    except Exception:
        pass  # Silent fail untuk observer notification

# Fungsi-fungsi observer helper untuk analisis packages

def notify_analyze_start(ui_components: Dict[str, Any], message: Optional[str] = None) -> None:
    """
    Notify observer tentang start analisis dengan silent fail
    
    Args:
        ui_components: UI components yang berisi observer_manager
        message: Pesan notifikasi (opsional)
    """
    observer_manager = ui_components.get('observer_manager')
    if not observer_manager or not hasattr(observer_manager, 'notify'):
        return
    
    # Default message jika tidak disediakan
    if not message:
        message = "Memulai analisis packages terinstall"
    
    # Notify dengan silent fail
    try:
        observer_manager.notify('DEPENDENCY_ANALYZE_START', None, {
            'message': message,
            'timestamp': time.time()
        })
    except Exception:
        pass  # Silent fail untuk observer notification

def notify_analyze_progress(ui_components: Dict[str, Any], progress: int, message: Optional[str] = None) -> None:
    """
    Notify observer tentang progress analisis dengan silent fail
    
    Args:
        ui_components: UI components yang berisi observer_manager
        progress: Nilai progress (0-100)
        message: Pesan notifikasi (opsional)
    """
    observer_manager = ui_components.get('observer_manager')
    if not observer_manager or not hasattr(observer_manager, 'notify'):
        return
    
    # Default message jika tidak disediakan
    if not message:
        message = f"Analisis packages progress: {progress}%"
    
    # Notify dengan silent fail
    try:
        observer_manager.notify('DEPENDENCY_ANALYZE_PROGRESS', None, {
            'message': message,
            'timestamp': time.time(),
            'progress': progress
        })
    except Exception:
        pass  # Silent fail untuk observer notification

def notify_analyze_error(ui_components: Dict[str, Any], error_msg: str) -> None:
    """
    Notify observer tentang error analisis dengan silent fail
    
    Args:
        ui_components: UI components yang berisi observer_manager
        error_msg: Pesan error
    """
    observer_manager = ui_components.get('observer_manager')
    if not observer_manager or not hasattr(observer_manager, 'notify'):
        return
    
    # Format pesan
    message = f"Gagal menganalisis packages: {error_msg}"
    
    # Notify dengan silent fail
    try:
        observer_manager.notify('DEPENDENCY_ANALYZE_ERROR', None, {
            'message': message,
            'timestamp': time.time(),
            'error': error_msg
        })
    except Exception:
        pass  # Silent fail untuk observer notification

def notify_analyze_complete(ui_components: Dict[str, Any], duration: float, result: List[str]) -> None:
    """
    Notify observer tentang complete analisis dengan silent fail
    
    Args:
        ui_components: UI components yang berisi observer_manager
        duration: Durasi analisis dalam detik
        result: Hasil analisis
    """
    observer_manager = ui_components.get('observer_manager')
    if not observer_manager or not hasattr(observer_manager, 'notify'):
        return
    
    # Format pesan
    message = f"Analisis packages selesai ({duration:.1f}s)"
    
    # Notify dengan silent fail
    try:
        observer_manager.notify('DEPENDENCY_ANALYZE_COMPLETE', None, {
            'message': message,
            'timestamp': time.time(),
            'duration': duration,
            'success': True,
            'result': result
        })
    except Exception:
        pass  # Silent fail untuk observer notification
