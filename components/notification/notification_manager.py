"""
File: smartcash/components/notification/notification_manager.py
Deskripsi: Manager untuk notifikasi UI dengan integrasi observer pattern dan progress tracking
"""

from typing import Dict, Any, Optional, Callable, List, Union
import time
import threading
from smartcash.components.observer import notify, EventTopics
from smartcash.components.observer.manager_observer import get_observer_manager
from smartcash.components.notification.notification_observer import create_notification_observer

class NotificationManager:
    """
    Manager untuk notifikasi UI dengan integrasi observer pattern.
    Menyediakan interface yang konsisten untuk notifikasi proses, error, dan progress updates.
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """
        Inisialisasi NotificationManager dengan komponen UI.
        
        Args:
            ui_components: Dictionary komponen UI yang berisi logger, progress_tracker, dll.
        """
        self.ui_components = ui_components or {}
        self.logger = ui_components.get('logger') if ui_components else None
        self.observer_manager = None
        
        # Coba dapatkan observer_manager dari ui_components atau buat baru
        if ui_components and 'observer_manager' in ui_components:
            self.observer_manager = ui_components['observer_manager']
        else:
            try:
                self.observer_manager = get_observer_manager()
            except Exception as e:
                if self.logger and hasattr(self.logger, 'debug'):
                    self.logger.debug(f"⚠️ Observer manager tidak tersedia: {str(e)}")
        
        # Thread-safe lock untuk operasi notifikasi
        self._notification_lock = threading.RLock()
        
        # Status notifikasi untuk mencegah duplikasi
        self._notification_status = {
            'last_notification': None,
            'last_timestamp': 0,
            'in_progress': False
        }
    
    def notify_process_start(self, 
                           process_name: str, 
                           message: Optional[str] = None, 
                           **kwargs) -> None:
        """
        Notifikasi awal proses dengan integrasi observer.
        
        Args:
            process_name: Nama proses yang dimulai
            message: Pesan notifikasi
            **kwargs: Parameter tambahan untuk notifikasi
        """
        with self._notification_lock:
            # Cegah duplikasi notifikasi dalam waktu singkat
            current_time = time.time()
            if (self._notification_status['last_notification'] == f"start_{process_name}" and 
                current_time - self._notification_status['last_timestamp'] < 0.5):
                return
                
            self._notification_status['last_notification'] = f"start_{process_name}"
            self._notification_status['last_timestamp'] = current_time
            
            # Default message jika tidak disediakan
            message = message or f"Memulai proses {process_name}"
            
            # Update UI progress jika tersedia
            self._update_ui_progress(0, message, 'info')
            
            # Notifikasi via observer
            self._notify_via_observer(
                f"{process_name.upper()}_START", 
                message=message,
                progress=0,
                total=100,
                status='info',
                timestamp=current_time,
                **kwargs
            )
            
            # Log jika logger tersedia
            if self.logger and hasattr(self.logger, 'info'):
                self.logger.info(f"🚀 {message}")
    
    def notify_process_complete(self, 
                              process_name: str, 
                              message: Optional[str] = None,
                              stats: Optional[Dict[str, Any]] = None,
                              **kwargs) -> None:
        """
        Notifikasi penyelesaian proses dengan integrasi observer.
        
        Args:
            process_name: Nama proses yang selesai
            message: Pesan notifikasi
            stats: Statistik hasil proses
            **kwargs: Parameter tambahan untuk notifikasi
        """
        with self._notification_lock:
            # Cegah duplikasi notifikasi dalam waktu singkat
            current_time = time.time()
            if (self._notification_status['last_notification'] == f"complete_{process_name}" and 
                current_time - self._notification_status['last_timestamp'] < 0.5):
                return
                
            self._notification_status['last_notification'] = f"complete_{process_name}"
            self._notification_status['last_timestamp'] = current_time
            
            # Default message jika tidak disediakan
            message = message or f"Proses {process_name} selesai"
            
            # Update UI progress jika tersedia
            self._update_ui_progress(100, message, 'success')
            
            # Notifikasi via observer
            self._notify_via_observer(
                f"{process_name.upper()}_COMPLETE", 
                message=message,
                progress=100,
                total=100,
                status='success',
                timestamp=current_time,
                stats=stats or {},
                **kwargs
            )
            
            # Log jika logger tersedia
            if self.logger and hasattr(self.logger, 'info'):
                self.logger.info(f"✅ {message}")
                
                # Log statistik jika ada
                if stats:
                    for key, value in stats.items():
                        self.logger.info(f"📊 {key}: {value}")
    
    def notify_process_error(self, 
                           process_name: str, 
                           error_message: str,
                           exception: Optional[Exception] = None,
                           **kwargs) -> None:
        """
        Notifikasi error proses dengan integrasi observer.
        
        Args:
            process_name: Nama proses yang mengalami error
            error_message: Pesan error
            exception: Exception yang terjadi (opsional)
            **kwargs: Parameter tambahan untuk notifikasi
        """
        with self._notification_lock:
            # Cegah duplikasi notifikasi dalam waktu singkat
            current_time = time.time()
            if (self._notification_status['last_notification'] == f"error_{process_name}" and 
                current_time - self._notification_status['last_timestamp'] < 0.5):
                return
                
            self._notification_status['last_notification'] = f"error_{process_name}"
            self._notification_status['last_timestamp'] = current_time
            
            # Format error message
            formatted_message = f"Error: {error_message}"
            if exception:
                formatted_message += f" ({str(exception)})"
            
            # Update UI progress jika tersedia
            self._update_ui_progress(0, formatted_message, 'error')
            
            # Notifikasi via observer
            self._notify_via_observer(
                f"{process_name.upper()}_ERROR", 
                message=formatted_message,
                error_message=error_message,
                error_details=str(exception) if exception else None,
                status='error',
                timestamp=current_time,
                **kwargs
            )
            
            # Log jika logger tersedia
            if self.logger and hasattr(self.logger, 'error'):
                self.logger.error(f"❌ {formatted_message}")
    
    def update_progress(self, 
                       process_name: str,
                       progress: int, 
                       total: int = 100,
                       message: Optional[str] = None,
                       **kwargs) -> None:
        """
        Update progress dengan integrasi observer.
        
        Args:
            process_name: Nama proses
            progress: Nilai progress saat ini
            total: Nilai total progress
            message: Pesan progress
            **kwargs: Parameter tambahan untuk notifikasi
        """
        with self._notification_lock:
            # Hitung persentase
            percentage = int((progress / total) * 100) if total > 0 else 0
            
            # Default message jika tidak disediakan
            message = message or f"Progress {process_name}: {percentage}%"
            
            # Update UI progress jika tersedia
            self._update_ui_progress(percentage, message, 'info')
            
            # Notifikasi via observer (throttle untuk mengurangi overhead)
            current_time = time.time()
            if (self._notification_status['last_notification'] != f"progress_{process_name}" or
                current_time - self._notification_status['last_timestamp'] > 0.2):
                
                self._notification_status['last_notification'] = f"progress_{process_name}"
                self._notification_status['last_timestamp'] = current_time
                
                self._notify_via_observer(
                    f"{process_name.upper()}_PROGRESS", 
                    message=message,
                    progress=progress,
                    total=total,
                    percentage=percentage,
                    status='info',
                    timestamp=current_time,
                    **kwargs
                )
            
            # Log jika logger tersedia dan progress adalah kelipatan 10%
            if self.logger and hasattr(self.logger, 'debug') and percentage % 10 == 0:
                self.logger.debug(f"📊 {message}")
    
    def update_status(self, 
                     message: str,
                     status_type: str = 'info',
                     **kwargs) -> None:
        """
        Update status panel dengan pesan.
        
        Args:
            message: Pesan status
            status_type: Tipe status ('info', 'warning', 'error', 'success')
            **kwargs: Parameter tambahan untuk notifikasi
        """
        with self._notification_lock:
            # Update UI status jika tersedia
            self._update_ui_status(message, status_type)
            
            # Log berdasarkan tipe status
            if self.logger:
                if status_type == 'error' and hasattr(self.logger, 'error'):
                    self.logger.error(f"❌ {message}")
                elif status_type == 'warning' and hasattr(self.logger, 'warning'):
                    self.logger.warning(f"⚠️ {message}")
                elif status_type == 'success' and hasattr(self.logger, 'info'):
                    self.logger.info(f"✅ {message}")
                elif hasattr(self.logger, 'info'):
                    self.logger.info(f"ℹ️ {message}")
    
    def _update_ui_progress(self, progress: int, message: str, status_type: str) -> None:
        """
        Update UI progress dengan berbagai metode yang tersedia.
        
        Args:
            progress: Nilai progress (0-100)
            message: Pesan progress
            status_type: Tipe status ('info', 'warning', 'error', 'success')
        """
        try:
            # Coba update dengan metode terbaru
            if 'update_progress' in self.ui_components and callable(self.ui_components['update_progress']):
                self.ui_components['update_progress']('overall', progress, message, status_type)
            
            # Fallback ke tracker jika tersedia
            elif 'tracker' in self.ui_components and hasattr(self.ui_components['tracker'], 'update'):
                self.ui_components['tracker'].update('overall', progress, message, status_type)
            
            # Fallback ke progress bar jika tersedia
            elif 'progress_bar' in self.ui_components:
                progress_bar = self.ui_components['progress_bar']
                if hasattr(progress_bar, 'value'):
                    progress_bar.value = progress
                if hasattr(progress_bar, 'description'):
                    progress_bar.description = f'Progress: {progress}%'
            
            # Fallback ke progress label jika tersedia
            if 'overall_label' in self.ui_components:
                label = self.ui_components['overall_label']
                if hasattr(label, 'value'):
                    color = self._get_status_color(status_type)
                    label.value = f"<div style='color: {color};'>{message}</div>"
        
        except Exception as e:
            # Silent fail untuk mencegah error UI mengganggu proses utama
            if self.logger and hasattr(self.logger, 'debug'):
                self.logger.debug(f"⚠️ UI progress update error: {str(e)}")
    
    def _update_ui_status(self, message: str, status_type: str) -> None:
        """
        Update UI status panel dengan pesan.
        
        Args:
            message: Pesan status
            status_type: Tipe status ('info', 'warning', 'error', 'success')
        """
        try:
            # Coba update dengan metode terbaru
            if 'update_status' in self.ui_components and callable(self.ui_components['update_status']):
                self.ui_components['update_status'](message, status_type)
            
            # Fallback ke status label jika tersedia
            elif 'status_label' in self.ui_components:
                label = self.ui_components['status_label']
                if hasattr(label, 'value'):
                    color = self._get_status_color(status_type)
                    label.value = f"<div style='color: {color};'>{message}</div>"
            
            # Fallback ke step label jika tersedia
            elif 'step_label' in self.ui_components:
                label = self.ui_components['step_label']
                if hasattr(label, 'value'):
                    color = self._get_status_color(status_type)
                    label.value = f"<div style='color: {color};'>{message}</div>"
        
        except Exception as e:
            # Silent fail untuk mencegah error UI mengganggu proses utama
            if self.logger and hasattr(self.logger, 'debug'):
                self.logger.debug(f"⚠️ UI status update error: {str(e)}")
    
    def _notify_via_observer(self, event_type: str, **kwargs) -> None:
        """
        Notifikasi via observer dengan multiple fallback.
        
        Args:
            event_type: Tipe event
            **kwargs: Parameter untuk event
        """
        try:
            # Cegah rekursi
            if self._notification_status.get('in_progress'):
                return
                
            self._notification_status['in_progress'] = True
            
            # Coba notifikasi via observer_manager dengan NotificationObserver
            if self.observer_manager and hasattr(self.observer_manager, 'register'):
                # Buat callback untuk event ini
                def notification_callback(event_type, sender, **event_kwargs):
                    # Hanya log untuk debugging
                    if self.logger and hasattr(self.logger, 'debug'):
                        self.logger.debug(f"🔔 Notification event: {event_type}")
                
                # Buat observer yang valid dengan adapter
                observer = create_notification_observer(
                    event_type=event_type,
                    callback=notification_callback,
                    name=f"NotificationManager_{event_type}",
                    priority=10
                )
                
                # Register observer ke manager
                try:
                    self.observer_manager.register(observer, event_type)
                    # Notify langsung setelah register
                    self.observer_manager.notify(event_type, None, **kwargs)
                    return
                except Exception as e:
                    if self.logger and hasattr(self.logger, 'debug'):
                        self.logger.debug(f"⚠️ Observer register error: {str(e)}")
            
            # Fallback ke EventDispatcher
            try:
                from smartcash.components.observer import EventDispatcher
                EventDispatcher.notify(event_type, None, **kwargs)
                return
            except (ImportError, AttributeError):
                pass
            
            # Fallback ke notify global
            notify(event_type, None, **kwargs)
            
        except Exception as e:
            # Silent fail untuk mencegah error notifikasi mengganggu proses utama
            if self.logger and hasattr(self.logger, 'debug'):
                self.logger.debug(f"⚠️ Observer notification error: {str(e)}")
        finally:
            self._notification_status['in_progress'] = False
    
    def _get_status_color(self, status_type: str) -> str:
        """
        Dapatkan warna CSS berdasarkan tipe status.
        
        Args:
            status_type: Tipe status ('info', 'warning', 'error', 'success')
            
        Returns:
            Warna CSS
        """
        status_colors = {
            'info': '#007bff',     # Biru
            'warning': '#ffc107',  # Kuning
            'error': '#dc3545',    # Merah
            'success': '#28a745',  # Hijau
            'default': '#6c757d'   # Abu-abu
        }
        return status_colors.get(status_type, status_colors['default'])


def get_notification_manager(ui_components: Optional[Dict[str, Any]] = None) -> NotificationManager:
    """
    Factory function untuk mendapatkan instance NotificationManager.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Instance NotificationManager
    """
    return NotificationManager(ui_components)
