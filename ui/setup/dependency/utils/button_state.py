"""
File: smartcash/ui/setup/dependency/utils/button_state.py

Utilitas sederhana untuk mengelola state tombol saat operasi berjalan.
"""
from typing import List, Dict, Any, Callable
import functools

def with_button_state(buttons: List[Any]):
    """
    Decorator untuk menangani disable/enable tombol saat operasi berjalan.
    
    Args:
        buttons: Daftar tombol yang akan di-disable selama operasi
        
    Returns:
        Decorator yang akan menangani state tombol
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Disable semua tombol
            for btn in buttons:
                if hasattr(btn, 'disabled'):
                    btn.disabled = True
            
            try:
                # Eksekusi fungsi
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Jika terjadi error, re-raise exception setelah enable tombol
                for btn in buttons:
                    if hasattr(btn, 'disabled'):
                        btn.disabled = False
                raise e
            finally:
                # Pastikan tombol di-enable kembali
                for btn in buttons:
                    if hasattr(btn, 'disabled'):
                        btn.disabled = False
        return wrapper
    return decorator


def create_button_state_handler(buttons: List[Any]) -> Dict[str, Callable]:
    """
    Membuat handler untuk mengelola state tombol.
    
    Args:
        buttons: Daftar tombol yang akan dikelola
        
    Returns:
        Dict berisi fungsi-fungsi untuk mengelola state tombol
    """
    def disable_all():
        """Nonaktifkan semua tombol"""
        for btn in buttons:
            if hasattr(btn, 'disabled'):
                btn.disabled = True
    
    def enable_all():
        """Aktifkan kembali semua tombol"""
        for btn in buttons:
            if hasattr(btn, 'disabled'):
                btn.disabled = False
    
    return {
        'disable_all': disable_all,
        'enable_all': enable_all,
        'with_state': with_button_state(buttons)
    }
