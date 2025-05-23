"""
File: smartcash/ui/dataset/preprocessing/utils/service_integration.py
Deskripsi: Integrasi dengan backend preprocessing service dan observer communication
"""

from typing import Dict, Any, Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor, Future
import json

from smartcash.common.logger import get_logger
from smartcash.common.config import get_config_manager
from smartcash.common.environment import get_environment_manager

logger = get_logger(__name__)

class ServiceIntegrator:
    """Integrator untuk komunikasi dengan backend preprocessing service."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi service integrator.
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self.logger = ui_components.get('logger', logger)
        self.config_manager = get_config_manager()
        self.env_manager = get_environment_manager()
        self.service = None
        self.progress_callback = None
        self.stop_requested = False
        
    def setup_service(self, config: Dict[str, Any]) -> bool:
        """
        Setup preprocessing service dengan konfigurasi.
        
        Args:
            config: Konfigurasi preprocessing
            
        Returns:
            bool: True jika berhasil setup
        """
        try:
            from smartcash.dataset.services.preprocessor.preprocessing_service import PreprocessingService
            
            # Konversi config UI ke format service
            service_config = self._convert_ui_config_to_service(config)
            
            # Buat service instance
            self.service = PreprocessingService(
                config=service_config,
                logger=self.logger,
                observer_manager=self.ui_components.get('observer_manager')
            )
            
            # Setup progress callback
            if hasattr(self.service.preprocessor, 'register_progress_callback'):
                self.service.preprocessor.register_progress_callback(self._progress_callback)
            
            self.logger.info("✅ Preprocessing service berhasil disetup")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error setup preprocessing service: {str(e)}")
            return False
    
    def _convert_ui_config_to_service(self, ui_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Konversi konfigurasi UI ke format backend service.
        
        Args:
            ui_config: Konfigurasi dari UI
            
        Returns:
            Dict: Konfigurasi untuk backend service
        """
        # Extract resolusi
        resolution = ui_config.get('resolution', (640, 640))
        if isinstance(resolution, str) and 'x' in resolution:
            width, height = map(int, resolution.split('x'))
            img_size = (width, height)
        elif isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            img_size = tuple(resolution)
        else:
            img_size = (640, 640)
        
        # Format konfigurasi service
        service_config = {
            'preprocessing': {
                'img_size': img_size,
                'normalize': ui_config.get('normalization', 'minmax') != 'none',
                'normalization': ui_config.get('normalization', 'minmax'),
                'preserve_aspect_ratio': ui_config.get('preserve_aspect_ratio', True),
                'augmentation': ui_config.get('augmentation', False),
                'num_workers': ui_config.get('num_workers', 1),  # Colab safe
                'force_reprocess': ui_config.get('force_reprocess', False),
                'output_dir': ui_config.get('preprocessed_dir', 'data/preprocessed'),
                'validation_items': ui_config.get('validation_items', [])
            },
            'data': {
                'dir': ui_config.get('data_dir', 'data')
            }
        }
        
        return service_config
    
    def _progress_callback(self, **kwargs) -> bool:
        """
        Callback untuk progress tracking dari service.
        
        Args:
            **kwargs: Parameter progress
            
        Returns:
            bool: False jika harus stop, True untuk lanjut
        """
        # Cek stop request
        if self.stop_requested or self.ui_components.get('stop_requested', False):
            self.logger.warning("⏹️ Stop request diterima dari UI")
            return False
        
        # Forward ke UI progress callback
        if self.progress_callback:
            try:
                self.progress_callback(**kwargs)
            except Exception as e:
                self.logger.error(f"❌ Error progress callback: {str(e)}")
        
        return True
    
    def register_progress_callback(self, callback: Callable) -> None:
        """
        Register callback untuk progress updates.
        
        Args:
            callback: Fungsi callback untuk progress
        """
        self.progress_callback = callback
    
    def preprocess_dataset(self, split: str = 'all') -> Optional[Dict[str, Any]]:
        """
        Jalankan preprocessing dataset.
        
        Args:
            split: Split yang akan diproses
            
        Returns:
            Optional[Dict]: Hasil preprocessing atau None jika error
        """
        if not self.service:
            self.logger.error("❌ Service belum disetup")
            return None
        
        try:
            # Reset stop flag
            self.stop_requested = False
            
            # Jalankan preprocessing
            result = self.service.preprocess_dataset(
                split=split,
                show_progress=True,
                force_reprocess=self.service.config.get('preprocessing', {}).get('force_reprocess', False)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error preprocessing dataset: {str(e)}")
            return None
    
    def stop_processing(self) -> None:
        """Stop preprocessing yang sedang berjalan."""
        self.stop_requested = True
        self.ui_components['stop_requested'] = True
        
        # Notify service jika ada
        if self.service and hasattr(self.service, 'stop'):
            self.service.stop()
    
    def get_processing_status(self) -> Dict[str, Any]:
        """
        Dapatkan status preprocessing.
        
        Returns:
            Dict: Status preprocessing
        """
        return {
            'service_ready': self.service is not None,
            'stop_requested': self.stop_requested,
            'ui_stop_requested': self.ui_components.get('stop_requested', False)
        }

def create_service_integrator(ui_components: Dict[str, Any]) -> ServiceIntegrator:
    """
    Factory function untuk membuat service integrator.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        ServiceIntegrator: Instance integrator
    """
    integrator = ServiceIntegrator(ui_components)
    ui_components['service_integrator'] = integrator
    return integrator

def setup_observer_communication(ui_components: Dict[str, Any]) -> None:
    """
    Setup komunikasi observer dengan backend service.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Import event topics
        from smartcash.components.observer.event_topics_observer import EventTopics
        
        observer_manager = ui_components.get('observer_manager')
        if not observer_manager:
            logger.warning("⚠️ Observer manager tidak tersedia")
            return
        
        # Register observer untuk event preprocessing
        def handle_preprocessing_event(event_type: str, sender: Any, **kwargs):
            """Handle preprocessing events dari backend"""
            try:
                if event_type == EventTopics.PREPROCESSING_PROGRESS:
                    # Forward progress ke UI
                    progress = kwargs.get('progress', 0)
                    total = kwargs.get('total', 100)
                    message = kwargs.get('message', '')
                    
                    # Update UI progress
                    if 'notification_manager' in ui_components:
                        ui_components['notification_manager'].notify_progress(
                            progress=progress,
                            total=total,
                            message=message,
                            **kwargs
                        )
                
                elif event_type == EventTopics.PREPROCESSING_START:
                    # Preprocessing dimulai
                    if 'notification_manager' in ui_components:
                        ui_components['notification_manager'].notify_process_start(
                            "preprocessing",
                            kwargs.get('display_info', ''),
                            kwargs.get('split')
                        )
                
                elif event_type == EventTopics.PREPROCESSING_END:
                    # Preprocessing selesai
                    result = kwargs.get('result', {})
                    if 'notification_manager' in ui_components:
                        ui_components['notification_manager'].notify_process_complete(
                            result,
                            kwargs.get('display_info', '')
                        )
                
                elif event_type == EventTopics.PREPROCESSING_ERROR:
                    # Error preprocessing
                    error_msg = kwargs.get('error', 'Unknown error')
                    if 'notification_manager' in ui_components:
                        ui_components['notification_manager'].notify_process_error(error_msg)
                        
            except Exception as e:
                logger.error(f"❌ Error handle preprocessing event: {str(e)}")
        
        # Register observer untuk berbagai event preprocessing
        preprocessing_events = [
            EventTopics.PREPROCESSING_START,
            EventTopics.PREPROCESSING_END,
            EventTopics.PREPROCESSING_PROGRESS,
            EventTopics.PREPROCESSING_ERROR,
            EventTopics.PREPROCESSING_CURRENT_PROGRESS,
            EventTopics.PREPROCESSING_STEP_PROGRESS
        ]
        
        for event in preprocessing_events:
            try:
                observer_manager.register(event, handle_preprocessing_event)
            except Exception as e:
                logger.warning(f"⚠️ Gagal register observer untuk {event}: {str(e)}")
        
        logger.info("✅ Observer communication berhasil disetup")
        
    except ImportError as e:
        logger.warning(f"⚠️ Tidak dapat import EventTopics: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Error setup observer communication: {str(e)}")

def validate_preprocessing_requirements(ui_components: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validasi persyaratan untuk menjalankan preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Tuple[bool, List[str]]: (valid, list_error_messages)
    """
    errors = []
    
    # Cek environment
    env_manager = get_environment_manager()
    if not env_manager.base_dir.exists():
        errors.append("Direktori base tidak ditemukan")
    
    # Cek data directory
    data_dir = ui_components.get('data_dir', 'data')
    if not Path(data_dir).exists():
        errors.append(f"Direktori data tidak ditemukan: {data_dir}")
    
    # Cek config manager
    try:
        config_manager = get_config_manager()
        config = config_manager.get_config()
        if not isinstance(config, dict):
            errors.append("Konfigurasi tidak valid")
    except Exception as e:
        errors.append(f"Error load konfigurasi: {str(e)}")
    
    # Cek backend service availability
    try:
        from smartcash.dataset.services.preprocessor.preprocessing_service import PreprocessingService
    except ImportError:
        errors.append("Backend preprocessing service tidak tersedia")
    
    return len(errors) == 0, errors