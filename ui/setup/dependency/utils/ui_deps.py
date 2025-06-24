"""
File: smartcash/ui/setup/dependency/utils/ui_deps.py
Deskripsi: Utility untuk manajemen dan validasi dependensi UI components
"""
from typing import Dict, Any, List, Tuple, Optional, Callable, TypeVar, cast, Union
from functools import wraps
import time
from enum import Enum

class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class OperationStatus(str, Enum):
    START = "start"
    SUCCESS = "success"
    FAILURE = "failure"

T = TypeVar('T', bound=Callable[..., Any])

def check_required(ui_components: Dict[str, Any], *required: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Cek komponen UI yang diperlukan dan kembalikan dictionary yang sudah divalidasi.
    
    Args:
        ui_components: Dictionary komponen UI
        *required: Nama-nama komponen yang diperlukan
        
    Returns:
        Tuple[bool, Dict]: (status_valid, components_dict)
    """
    logger = ui_components.get('logger')
    if not logger:
        print("❌ Logger belum diinisialisasi")
        return False, {}
    
    result = {'logger': logger}
    missing = []
    
    for comp in required:
        if comp in ui_components:
            result[comp] = ui_components[comp]
        else:
            missing.append(comp)
    
    if missing:
        logger.error(f"❌ Komponen tidak ditemukan: {', '.join(missing)}")
        return False, {}
        
    return True, result


def requires(*components: str) -> Callable[[T], T]:
    """
    Decorator untuk memvalidasi komponen UI yang diperlukan sebelum menjalankan fungsi.
    
    Contoh:
        @requires('progress_tracker', 'log_output')
        def my_handler(ui_components, ...):
            # ui_components sudah divalidasi
            pass
    """
    def decorator(func: T) -> T:
        @wraps(func)
        def wrapper(ui_components: Dict[str, Any], *args, **kwargs):
            is_valid, comps = check_required(ui_components, *components)
            if not is_valid:
                return None
            return func({**ui_components, **comps}, *args, **kwargs)
        return cast(T, wrapper)
    return decorator


def get_optional(ui_components: Dict[str, Any], *components: str) -> Dict[str, Any]:
    """
    Ambil komponen opsional tanpa validasi ketat.
    Hanya mengembalikan komponen yang ada.
    """
    return {k: v for k, v in ui_components.items() 
            if k in components and v is not None}


def log_operation(operation_name: str, level: LogLevel = LogLevel.INFO, 
                with_status: bool = True, log_args: bool = False):
    """
    Decorator untuk menstandarisasi logging operasi.
    
    Args:
        operation_name: Nama operasi yang akan dilog
        level: Level logging (default: INFO)
        with_status: Apakah menambahkan status START/SUCCESS/FAILURE
        log_args: Apakah menambahkan argumen pemanggilan ke log
    """
    def decorator(func):
        @wraps(func)
        def wrapper(ui_components: Dict[str, Any], *args, **kwargs):
            logger = ui_components.get('logger')
            if not logger:
                return func(ui_components, *args, **kwargs)
                
            # Log start
            start_time = time.time()
            status = OperationStatus.START if with_status else ""
            status_msg = f" {status.upper()}" if status else ""
            
            # Format argumen jika diperlukan
            args_info = ""
            if log_args and (args or kwargs):
                args_str = ", ".join([str(a) for a in args] + 
                                  [f"{k}={v}" for k, v in kwargs.items()])
                args_info = f" with args: {args_str}"
            
            # Log start
            start_msg = f"🚀 {operation_name.upper()}{status_msg}{args_info}"
            getattr(logger, level.value, logger.info)(start_msg)
            
            try:
                # Eksekusi fungsi
                result = func(ui_components, *args, **kwargs)
                
                # Log success
                if with_status:
                    duration = time.time() - start_time
                    success_msg = (
                        f"✅ {operation_name.upper()} {OperationStatus.SUCCESS.upper()} "
                        f"({duration:.2f}s)"
                    )
                    getattr(logger, level.value, logger.info)(success_msg)
                
                return result
                
            except Exception as e:
                # Log error
                if with_status:
                    error_msg = (
                        f"❌ {operation_name.upper()} {OperationStatus.FAILURE.upper()}: "
                        f"{str(e)}"
                    )
                    logger.error(error_msg, exc_info=True)
                raise
                
        return wrapper
    return decorator


def with_logging(operation_name: str = None, level: LogLevel = LogLevel.INFO):
    """
    Decorator untuk menggabungkan requires() dan log_operation()
    """
    def decorator(func):
        # Dapatkan nama operasi dari nama fungsi jika tidak disediakan
        op_name = operation_name or func.__name__.replace('_', ' ').title()
        
        # Gabungkan dengan log_operation
        logged_func = log_operation(op_name, level)(func)
        
        # Pertahankan atribut asli fungsi
        logged_func.__name__ = func.__name__
        logged_func.__doc__ = func.__doc__
        
        return logged_func
    return decorator
