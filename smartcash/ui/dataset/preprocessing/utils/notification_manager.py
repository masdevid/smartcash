"""
File: smartcash/ui/dataset/preprocessing/utils/notification_manager.py
Deskripsi: Manager untuk notifikasi pada modul preprocessing dataset
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message

# Import namespace konstanta
from smartcash.ui.dataset.preprocessing.preprocessing_initializer import PREPROCESSING_LOGGER_NAMESPACE, MODULE_LOGGER_NAME

class NotificationManager:
    """Manager untuk notifikasi pada modul preprocessing dataset."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi NotificationManager.
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self.logger = ui_components.get('logger', get_logger(PREPROCESSING_LOGGER_NAMESPACE))
        self.namespace = PREPROCESSING_LOGGER_NAMESPACE
        self.module_name = MODULE_LOGGER_NAME
        
        # Pastikan fungsi update_status_panel tersedia
        if not self._check_required_functions():
            self.logger.warning(f"[{MODULE_LOGGER_NAME}] ⚠️ Beberapa fungsi yang diperlukan NotificationManager tidak tersedia")
    
    def _check_required_functions(self) -> bool:
        """
        Periksa apakah fungsi-fungsi yang diperlukan tersedia.
        
        Returns:
            bool: True jika semua fungsi tersedia, False jika tidak
        """
        required_functions = [
            'update_status_panel',
            'notify_process_start',
            'notify_process_complete',
            'notify_process_error'
        ]
        
        all_available = True
        for func_name in required_functions:
            if func_name not in self.ui_components and not hasattr(self, func_name):
                self.logger.warning(f"[{MODULE_LOGGER_NAME}] ⚠️ Fungsi {func_name} tidak tersedia untuk NotificationManager")
                all_available = False
                
        return all_available
    
    def update_status(self, status_type: str, message: str) -> None:
        """
        Update status panel dengan pesan dan tipe yang ditentukan.
        
        Args:
            status_type: Tipe pesan ('info', 'success', 'warning', 'error')
            message: Pesan yang akan ditampilkan
        """
        # Import fungsi yang diperlukan
        from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import update_status_panel
        
        try:
            # Update status panel
            update_status_panel(self.ui_components, status_type, message)
            
            # Log pesan ke logger juga
            log_message(self.ui_components, message, status_type)
        except Exception as e:
            # Fallback jika update_status_panel tidak tersedia
            if status_type == 'error':
                self.logger.error(f"[{MODULE_LOGGER_NAME}] {message} (Error update status: {str(e)})")
            elif status_type == 'warning':
                self.logger.warning(f"[{MODULE_LOGGER_NAME}] {message} (Error update status: {str(e)})")
            elif status_type == 'success':
                self.logger.info(f"[{MODULE_LOGGER_NAME}] ✅ {message} (Error update status: {str(e)})")
            else:
                self.logger.info(f"[{MODULE_LOGGER_NAME}] {message} (Error update status: {str(e)})")
    
    def notify_progress(self, **kwargs) -> None:
        """
        Notifikasi update progress untuk preprocessing.
        
        Args:
            **kwargs: Parameter untuk progress tracking
        """
        try:
            # Import fungsi yang diperlukan
            from smartcash.ui.dataset.preprocessing.utils.progress_manager import update_progress
            
            # Ekstrak parameter dari kwargs
            progress = kwargs.get('progress', 0)
            total = kwargs.get('total', 100)
            message = kwargs.get('message', '')
            step_message = kwargs.get('step_message', '')
            
            # Update progress bar
            update_progress(
                self.ui_components, 
                progress, 
                total, 
                overall_message=message,
                step_message=step_message
            )
            
            # Cek jika message ada, log message
            if message:
                # Format kemajuan untuk pesan log
                progress_pct = min(100, int((progress / total) * 100)) if total > 0 else 0
                log_message(
                    self.ui_components,
                    f"Progress: {progress_pct}% - {message}",
                    "info",
                    "🔄"
                )
            
            # Jika observer_manager ada, notifikasi dengan format standar service
            self._notify_service_progress(**kwargs)
            
            # Cek apakah perlu menghentikan proses
            if self.ui_components.get('stop_requested', False):
                return False
                
            return True
            
        except Exception as e:
            log_message(self.ui_components, f"Error saat notify progress: {str(e)}", "warning", "⚠️")
            return True  # Default lanjutkan
    
    def notify_process_start(self, process_name: str, display_info: str, split: Optional[str] = None) -> None:
        """
        Notifikasi bahwa proses telah dimulai.
        
        Args:
            process_name: Nama proses yang dimulai
            display_info: Informasi tambahan untuk ditampilkan
            split: Split dataset yang diproses (opsional)
        """
        # Import fungsi yang diperlukan
        from smartcash.ui.dataset.preprocessing.utils.ui_observers import notify_process_start
        
        try:
            # Set flag yang diperlukan
            self.ui_components['preprocessing_running'] = True
            self.ui_components['stop_requested'] = False
            
            # Format pesan start
            message = f"Memulai {process_name} {display_info}"
            
            # Update status
            self.update_status('info', f"{ICONS['start']} {message}")
            
            # Notifikasi ke UI observer
            notify_process_start(self.ui_components, process_name, display_info, split)
            
            # Log pesan
            log_message(self.ui_components, message, "info", ICONS['start'])
            
            # Notify dengan format standar service
            self._notify_service_event(
                "preprocessing",
                "start",
                message=message,
                progress=0,
                total_steps=5,
                current_step=1,
                split=split
            )
        except Exception as e:
            log_message(self.ui_components, f"Error saat notifikasi proses mulai: {str(e)}", "warning", "⚠️")
    
    def notify_process_complete(self, result: Dict[str, Any], display_info: str) -> None:
        """
        Notifikasi bahwa proses telah selesai dengan sukses.
        
        Args:
            result: Dictionary hasil proses
            display_info: Informasi tambahan untuk ditampilkan
        """
        # Import fungsi yang diperlukan
        from smartcash.ui.dataset.preprocessing.utils.ui_observers import notify_process_complete
        
        try:
            # Format pesan complete
            message = f"Preprocessing {display_info} selesai"
            
            # Update status
            self.update_status('success', f"{ICONS['success']} {message}")
            
            # Notifikasi
            notify_process_complete(self.ui_components, result, display_info)
            
            # Log pesan
            total_processed = result.get('processed', 0)
            total_skipped = result.get('skipped', 0)
            total_failed = result.get('failed', 0)
            log_message(
                self.ui_components, 
                f"{message}: {total_processed} gambar diproses, {total_skipped} dilewati, {total_failed} gagal",
                "success",
                ICONS['success']
            )
            
            # Reset flag running
            self.ui_components['preprocessing_running'] = False
            
            # Notify dengan format standar service
            self._notify_service_event(
                "preprocessing",
                "complete",
                message=message,
                result=result,
                progress=100,
                total=100,
                step="complete"
            )
        except Exception as e:
            log_message(self.ui_components, f"Error saat notifikasi proses selesai: {str(e)}", "warning", "⚠️")
    
    def notify_process_error(self, error_message: str) -> None:
        """
        Notifikasi bahwa proses mengalami error.
        
        Args:
            error_message: Pesan error yang terjadi
        """
        # Import fungsi yang diperlukan
        from smartcash.ui.dataset.preprocessing.utils.ui_observers import notify_process_error
        
        try:
            # Format pesan error
            message = f"Error: {error_message}"
            
            # Update status
            self.update_status('error', f"{ICONS['error']} {message}")
            
            # Notifikasi
            notify_process_error(self.ui_components, error_message)
            
            # Log pesan
            log_message(self.ui_components, message, "error", ICONS['error'])
            
            # Reset flag running
            self.ui_components['preprocessing_running'] = False
            
            # Notify dengan format standar service
            self._notify_service_event(
                "preprocessing",
                "error",
                message=message,
                error=error_message,
                progress=0,
                total=100,
                step="error"
            )
        except Exception as e:
            log_message(self.ui_components, f"Error saat notifikasi error: {str(e)}", "warning", "⚠️")
    
    def notify_process_stop(self, message: str = "Stop oleh pengguna") -> None:
        """
        Notifikasi bahwa proses telah dihentikan oleh pengguna.
        
        Args:
            message: Pesan yang menjelaskan alasan stop
        """
        # Import fungsi yang diperlukan
        from smartcash.ui.dataset.preprocessing.utils.ui_observers import notify_process_stop
        
        try:
            # Format pesan stop
            stop_message = f"Preprocessing dihentikan: {message}"
            
            # Update status
            self.update_status('warning', f"{ICONS['stop']} {stop_message}")
            
            # Set flag stop_requested di UI components
            self.ui_components['stop_requested'] = True
            self.ui_components['preprocessing_running'] = False
            
            # Notifikasi ke observer sistem
            observer_manager = self.ui_components.get('observer_manager')
            if observer_manager and hasattr(observer_manager, 'set_flag'):
                observer_manager.set_flag('stop_requested', True)
            
            # Log pesan
            log_message(self.ui_components, stop_message, "warning", ICONS['stop'])
            
            # Notifikasi ke UI observer
            notify_process_stop(self.ui_components, message)
            
            # Notify dengan format standar service
            self._notify_service_event(
                "preprocessing",
                "stop",
                message=stop_message,
                stop_reason=message,
                progress=0,
                total=100,
                step="stopped"
            )
            
            # Reset UI dalam kasus stop
            try:
                from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import reset_ui_after_preprocessing
                reset_ui_after_preprocessing(self.ui_components)
            except Exception as e:
                log_message(self.ui_components, f"Error saat reset UI setelah stop: {str(e)}", "warning", "⚠️")
                
        except Exception as e:
            log_message(self.ui_components, f"Error saat notifikasi stop: {str(e)}", "warning", "⚠️")
            # Reset UI dalam kasus error
            try:
                from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import reset_ui_after_preprocessing
                reset_ui_after_preprocessing(self.ui_components)
            except Exception:
                pass
    
    def _notify_service_event(self, category: str, event_type: str, **kwargs) -> None:
        """
        Notifikasi event ke observer manager dengan format standar service.
        
        Args:
            category: Kategori event ('preprocessing')
            event_type: Tipe event ('start', 'progress', 'complete', 'error')
            **kwargs: Parameter tambahan untuk event
        """
        try:
            # Dapatkan observer_manager dari UI components
            observer_manager = self.ui_components.get('observer_manager')
            if observer_manager and hasattr(observer_manager, 'notify'):
                try:
                    # Import utilitas notifikasi service
                    from smartcash.dataset.services.downloader.notification_utils import notify_service_event
                    
                    # Notify sesuai standar service
                    notify_service_event(
                        category,
                        event_type,
                        self,
                        observer_manager,
                        **kwargs
                    )
                except ImportError as e:
                    log_message(self.ui_components, f"Tidak dapat mengimpor notify_service_event: {str(e)}", "debug", "ℹ️")
        except Exception as e:
            log_message(self.ui_components, f"Error saat notifikasi service event: {str(e)}", "debug", "ℹ️")
    
    def _notify_service_progress(self, **kwargs) -> None:
        """
        Khusus notifikasi progress ke observer manager dengan format standar service.
        
        Args:
            **kwargs: Parameter tambahan untuk event progress
        """
        # Parameter wajib untuk progress event
        progress_params = {}
        
        # Copy parameter yang relevan
        if 'message' in kwargs:
            progress_params['message'] = kwargs['message']
        if 'step' in kwargs:
            progress_params['step'] = kwargs['step']
            
        # Progress tracking
        if 'progress' in kwargs:
            progress_params['progress'] = kwargs['progress']
        if 'total' in kwargs:
            progress_params['total'] = kwargs['total']
            
        # Split info
        if 'split' in kwargs:
            progress_params['split'] = kwargs['split']
            
        # Tambahkan semua parameter lain yang mungkin diperlukan
        for key, value in kwargs.items():
            if key not in progress_params:
                progress_params[key] = value
                
        # Kirim notifikasi standar
        self._notify_service_event('preprocessing', 'progress', **progress_params)

def get_notification_manager(ui_components: Dict[str, Any]) -> NotificationManager:
    """
    Dapatkan instance NotificationManager yang sudah terinisialisasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        NotificationManager yang sudah terinisialisasi
    """
    # Cek apakah notification_manager sudah ada di UI components
    if 'notification_manager' in ui_components and isinstance(ui_components['notification_manager'], NotificationManager):
        return ui_components['notification_manager']
    
    # Buat notification manager baru
    notification_manager = NotificationManager(ui_components)
    
    # Simpan ke UI components untuk penggunaan berikutnya
    ui_components['notification_manager'] = notification_manager
    
    return notification_manager 