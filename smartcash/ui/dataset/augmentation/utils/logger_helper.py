"""
File: smartcash/ui/dataset/augmentation/utils/logger_helper.py
Deskripsi: Helper untuk logging UI augmentasi dengan filtering yang diperbaiki untuk mencegah log hyperparameters tercampur
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.ui_logger import log_to_ui as ui_log
from smartcash.common.logger import get_logger

# Import namespace konstanta
from smartcash.ui.dataset.augmentation.augmentation_initializer import AUGMENTATION_LOGGER_NAMESPACE, MODULE_LOGGER_NAME

# Pattern log yang harus difilter untuk mencegah tercampurnya log dari modul lain
EXCLUDED_LOG_PATTERNS = [
    'hyperparameters', 'optimizer', 'lr_scheduler', 'adam', 'sgd',
    'training', 'epochs', 'batch_size', 'learning_rate',
    'model_config', 'backbone', 'efficientnet', 'yolov5',
    'checkpoint', 'tensorboard', 'wandb', 'mlflow',
    'validation', 'metrics', 'mAP', 'precision', 'recall',
    'loss', 'bbox_loss', 'cls_loss', 'obj_loss'
]

def log_message(ui_components: Dict[str, Any], message: str, level: str = "info", icon: Optional[str] = None) -> None:
    """
    Log pesan ke UI dan logger Python dengan namespace khusus augmentasi dan filtering.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan di-log
        level: Level log (info, warning, error, success)
        icon: Ikon opsional
    """
    # Cek apakah ini adalah augmentasi yang sudah diinisialisasi
    if not is_initialized(ui_components):
        return
    
    # Filter pesan yang mengandung pattern yang tidak relevan dengan augmentasi
    message_lower = message.lower()
    if any(pattern in message_lower for pattern in EXCLUDED_LOG_PATTERNS):
        return
    
    # Filter pesan dari namespace lain
    if _is_from_other_namespace(ui_components, message):
        return
    
    # Pastikan menggunakan logger dengan namespace yang tepat
    logger = ui_components.get('logger') or get_logger(AUGMENTATION_LOGGER_NAMESPACE)
    
    # Log ke UI hanya jika log_output atau output tersedia
    if 'log_output' in ui_components or 'output' in ui_components:
        ui_log(ui_components, message, level, icon)
    
    # Tambahkan prefix untuk memudahkan filtering
    prefixed_message = f"[{MODULE_LOGGER_NAME}] {message}"
    
    # Log ke Python logger
    if logger:
        getattr(logger, level if level != "success" else "info")(
            f"✅ {prefixed_message}" if level == "success" else prefixed_message
        )

def _is_from_other_namespace(ui_components: Dict[str, Any], message: str) -> bool:
    """
    Cek apakah pesan berasal dari namespace lain.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan log
        
    Returns:
        bool: True jika pesan dari namespace lain
    """
    current_namespace = ui_components.get('logger_namespace', AUGMENTATION_LOGGER_NAMESPACE)
    
    # Daftar namespace yang diketahui
    other_namespaces = [
        'smartcash.dataset.preprocessing',
        'smartcash.setup.dependency_installer',
        'smartcash.dataset.download',
        'smartcash.ui.env_config',
        'smartcash.training',
        'smartcash.model'
    ]
    
    # Cek apakah ada indikator namespace lain dalam pesan
    for namespace in other_namespaces:
        if namespace != current_namespace:
            # Ambil bagian terakhir dari namespace sebagai indikator
            namespace_indicator = namespace.split('.')[-1].upper()
            if f"[{namespace_indicator}]" in message:
                return True
    
    return False

def setup_ui_logger(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup UI logger untuk module augmentasi dengan filtering yang diperbaiki.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah ditambahkan logger
    """
    # Setup logger jika belum ada
    if 'logger' not in ui_components:
        ui_components['logger'] = get_logger(AUGMENTATION_LOGGER_NAMESPACE)
    
    # Tambahkan fungsi log_message ke UI components
    ui_components['log_message'] = lambda msg, level="info", icon=None: log_message(ui_components, msg, level, icon)
    
    # Tambahkan flag augmentation_initialized
    ui_components['augmentation_initialized'] = True
    
    # Tambahkan namespace
    ui_components['logger_namespace'] = AUGMENTATION_LOGGER_NAMESPACE
    
    # Setup stdout interceptor dengan filter yang lebih ketat
    _setup_stdout_filter(ui_components)
    
    return ui_components 

def _setup_stdout_filter(ui_components: Dict[str, Any]) -> None:
    """Setup filter untuk stdout agar tidak tercampur dengan log lain."""
    import sys
    from io import StringIO
    
    # Simpan stdout original
    if 'original_stdout' not in ui_components:
        ui_components['original_stdout'] = sys.stdout
    
    class AugmentationStdoutFilter:
        def __init__(self, ui_components):
            self.ui_components = ui_components
            self.original_stdout = ui_components.get('original_stdout', sys.__stdout__)
            
        def write(self, message):
            # Tulis ke stdout original
            self.original_stdout.write(message)
            
            # Filter pesan untuk UI
            if self._should_show_in_ui(message):
                log_message(self.ui_components, message.strip(), "info")
                
        def flush(self):
            self.original_stdout.flush()
            
        def _should_show_in_ui(self, message: str) -> bool:
            """Tentukan apakah pesan harus ditampilkan di UI augmentasi."""
            if not message or not message.strip():
                return False
                
            message_lower = message.lower().strip()
            
            # Filter pattern yang tidak relevan
            if any(pattern in message_lower for pattern in EXCLUDED_LOG_PATTERNS):
                return False
                
            # Hanya tampilkan pesan yang berkaitan dengan augmentasi
            augmentation_keywords = [
                'augment', 'augmentasi', 'variasi', 'kombinasi',
                'posisi', 'pencahayaan', 'flip', 'rotasi', 'scaling',
                'brightness', 'contrast', 'hsv', 'blur', 'noise'
            ]
            
            return any(keyword in message_lower for keyword in augmentation_keywords)
    
    # Pasang filter hanya jika belum dipasang
    if not isinstance(sys.stdout, AugmentationStdoutFilter):
        sys.stdout = AugmentationStdoutFilter(ui_components)

def restore_stdout(ui_components: Dict[str, Any]) -> None:
    """Kembalikan stdout ke kondisi original."""
    import sys
    
    if 'original_stdout' in ui_components:
        sys.stdout = ui_components['original_stdout']
        
def is_initialized(ui_components: Dict[str, Any]) -> bool:
    """
    Cek apakah UI logger sudah diinisialisasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        bool: True jika sudah diinisialisasi
    """
    return ui_components.get('augmentation_initialized', False)