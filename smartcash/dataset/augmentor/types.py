"""
File: smartcash/dataset/augmentor/types.py
Deskripsi: Enhanced type definitions dengan service integration dan UI progress types
"""

from typing import Dict, List, Callable, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

# Type aliases untuk service integration
ProgressCallback = Callable[[str, int, int, str], None]
UIComponents = Dict[str, Any]
ServiceResult = Dict[str, Any]
FileList = List[str]
ClassDistribution = Dict[str, int]
AugmentationTypes = List[str]
PathStr = Union[str, Path]

# Service operation states
class OperationState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class ServiceConfig:
    """Configuration untuk service layer dengan UI integration"""
    ui_components: UIComponents
    config: Dict[str, Any]
    target_split: str = "train"
    progress_callback: Optional[ProgressCallback] = None
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """Extract augmentation config untuk backend"""
        return self.config.get('augmentation', {})
    
    def get_ui_logger(self):
        """Get UI logger dengan fallback"""
        return self.ui_components.get('logger')

@dataclass
class AugConfig:
    """Enhanced konfigurasi augmentasi dengan UI parameter mapping"""
    raw_dir: str = "data"
    aug_dir: str = "data/augmented" 
    prep_dir: str = "data/preprocessed"
    num_variations: int = 2
    target_count: int = 500
    output_prefix: str = "aug"
    process_bboxes: bool = True
    validate_results: bool = False
    target_split: str = "train"
    balance_classes: bool = True
    types: List[str] = None
    
    # Position parameters dari UI
    fliplr: float = 0.5
    degrees: int = 10
    translate: float = 0.1
    scale: float = 0.1
    
    # Lighting parameters dari UI
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    brightness: float = 0.2
    contrast: float = 0.2
    
    def __post_init__(self):
        """Validasi dan default values"""
        if self.num_variations <= 0: self.num_variations = 2
        if self.target_count <= 0: self.target_count = 500
        if not self.types: self.types = ['combined']

@dataclass 
class ProcessingStats:
    """Enhanced statistik hasil pemrosesan dengan service metrics"""
    total_files: int = 0
    processed_files: int = 0
    generated_images: int = 0
    success_rate: float = 0.0
    duration_seconds: float = 0.0
    processing_speed: float = 0.0  # files per second
    
    def calculate_success_rate(self) -> float:
        """Hitung success rate berdasarkan file yang diproses"""
        return (self.processed_files / max(1, self.total_files)) * 100
    
    def calculate_processing_speed(self) -> float:
        """Hitung processing speed"""
        return self.processed_files / max(1, self.duration_seconds)

@dataclass
class ServiceProgress:
    """Progress tracking untuk service operations"""
    operation: str
    current_step: str = ""
    overall_progress: int = 0
    step_progress: int = 0
    current_progress: int = 0
    message: str = ""
    state: OperationState = OperationState.PENDING
    
    def update_overall(self, progress: int, message: str = ""):
        """Update overall progress"""
        self.overall_progress = min(100, max(0, progress))
        if message: self.message = message
    
    def update_step(self, step: str, progress: int, message: str = ""):
        """Update step progress"""
        self.current_step = step
        self.step_progress = min(100, max(0, progress))
        if message: self.message = message
    
    def update_current(self, progress: int, message: str = ""):
        """Update current operation progress"""
        self.current_progress = min(100, max(0, progress))
        if message: self.message = message

@dataclass
class ClassBalanceInfo:
    """Enhanced informasi balancing kelas dengan service integration"""
    class_counts: ClassDistribution
    augmentation_needs: ClassDistribution
    selected_files: FileList
    target_classes: List[int]
    balancing_enabled: bool = False
    priority_order: List[str] = None
    
    def __post_init__(self):
        if not self.priority_order: self.priority_order = []
    
    def get_classes_needing_augmentation(self) -> List[str]:
        """Dapatkan daftar kelas yang perlu augmentasi"""
        return [cls for cls, need in self.augmentation_needs.items() if need > 0]
    
    def get_total_needed(self) -> int:
        """Total sampel yang dibutuhkan"""
        return sum(self.augmentation_needs.values())

@dataclass
class DatasetComparison:
    """Comparison result antara raw dan preprocessed dataset"""
    raw_ready: bool = False
    preprocessed_exists: bool = False
    augmentation_ready: bool = False
    split_comparison: Dict[str, Dict[str, Any]] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if not self.split_comparison: self.split_comparison = {}
        if not self.recommendations: self.recommendations = []

@dataclass
class AugmentationResult:
    """Enhanced hasil augmentasi dengan service metadata"""
    status: str = "pending"
    message: str = ""
    stats: ProcessingStats = None
    class_info: ClassBalanceInfo = None
    output_paths: Dict[str, str] = None
    validation_result: Dict[str, Any] = None
    balance_result: Dict[str, Any] = None
    aug_result: Dict[str, Any] = None
    norm_result: Dict[str, Any] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        """Inisialisasi default values"""
        if self.stats is None: self.stats = ProcessingStats()
        if self.output_paths is None: self.output_paths = {}
        if self.validation_result is None: self.validation_result = {}
        if self.balance_result is None: self.balance_result = {}
        if self.aug_result is None: self.aug_result = {}
        if self.norm_result is None: self.norm_result = {}
    
    def is_success(self) -> bool:
        """Check apakah augmentasi berhasil"""
        return self.status == "success"
    
    def get_summary(self) -> str:
        """Dapatkan ringkasan hasil augmentasi"""
        if self.is_success():
            return f"✅ {self.stats.generated_images} gambar dihasilkan dari {self.stats.processed_files} file"
        return f"❌ {self.message}"
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Summary lengkap pipeline execution"""
        return {
            'status': self.status,
            'total_generated': self.stats.generated_images if self.stats else 0,
            'processing_time': self.processing_time,
            'success_rate': self.stats.success_rate if self.stats else 0,
            'validation_passed': self.validation_result.get('valid', False),
            'balance_enabled': self.balance_result.get('status') != 'skipped',
            'classes_augmented': self.balance_result.get('classes_needing_aug', 0)
        }

# Service communication types
@dataclass
class ServiceMessage:
    """Message format untuk service communication"""
    operation: str
    level: str  # 'overall', 'step', 'current'
    progress: int
    total: int
    message: str
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            import time
            self.timestamp = time.time()

@dataclass
class UIProgressUpdate:
    """Progress update format untuk UI integration"""
    service_message: ServiceMessage
    ui_components: UIComponents
    callback_executed: bool = False
    
    def execute_ui_update(self):
        """Execute UI update berdasarkan service message"""
        if self.callback_executed:
            return
        
        try:
            # Update progress tracker jika ada
            progress_tracker = self.ui_components.get('progress_tracker')
            if progress_tracker and hasattr(progress_tracker, 'update'):
                if self.service_message.level == 'overall':
                    progress_tracker.update_overall(self.service_message.progress, self.service_message.message)
                elif self.service_message.level == 'step':
                    progress_tracker.update_step(self.service_message.progress, self.service_message.message)
                elif self.service_message.level == 'current':
                    progress_tracker.update_current(self.service_message.progress, self.service_message.message)
            
            # Log ke UI logger
            logger = self.ui_components.get('logger')
            if logger and hasattr(logger, 'info'):
                logger.info(self.service_message.message)
            
            self.callback_executed = True
            
        except Exception:
            # Silent fail untuk UI updates
            pass

# Constants untuk currency detection
LAYER_1_CLASSES = list(range(0, 7))    # Banknote detection layer
LAYER_2_CLASSES = list(range(7, 14))   # Nominal detection layer  
LAYER_3_CLASSES = list(range(14, 17))  # Security features (diabaikan untuk balancing)
TARGET_CLASSES = LAYER_1_CLASSES + LAYER_2_CLASSES

# File extensions yang didukung
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
SUPPORTED_LABEL_EXTENSIONS = ['.txt']

# Augmentation pipeline types
DEFAULT_AUGMENTATION_TYPES = ['combined', 'position', 'lighting']
RESEARCH_AUGMENTATION_TYPES = ['combined']  # Untuk penelitian
ALL_AUGMENTATION_TYPES = ['combined', 'position', 'lighting', 'geometric', 'color', 'noise', 'light', 'heavy']

# Service operation mapping
SERVICE_OPERATIONS = {
    'augmentation': {
        'steps': ['validate', 'balance', 'augment', 'normalize', 'complete'],
        'weights': {'validate': 15, 'balance': 10, 'augment': 45, 'normalize': 25, 'complete': 5}
    },
    'check_dataset': {
        'steps': ['locate', 'analyze_raw', 'analyze_preprocessed', 'compare', 'report'],
        'weights': {'locate': 10, 'analyze_raw': 30, 'analyze_preprocessed': 20, 'compare': 20, 'report': 20}
    },
    'cleanup': {
        'steps': ['scan', 'analyze', 'confirm', 'execute', 'verify'],
        'weights': {'scan': 15, 'analyze': 10, 'confirm': 5, 'execute': 60, 'verify': 10}
    }
}

# One-liner factory functions
create_service_config = lambda ui_components, config, target_split='train': ServiceConfig(ui_components, config, target_split)
create_aug_config_from_ui = lambda ui_components: AugConfig(**_extract_aug_params_from_ui(ui_components))
create_progress_tracker = lambda operation: ServiceProgress(operation)
create_service_message = lambda op, level, progress, total, msg: ServiceMessage(op, level, progress, total, msg)

def _extract_aug_params_from_ui(ui_components: UIComponents) -> Dict[str, Any]:
    """Extract augmentation parameters dari UI components"""
    def safe_get(key, default):
        widget = ui_components.get(key)
        return getattr(widget, 'value', default) if widget and hasattr(widget, 'value') else default
    
    return {
        'num_variations': safe_get('num_variations', 3),
        'target_count': safe_get('target_count', 500),
        'output_prefix': safe_get('output_prefix', 'aug'),
        'balance_classes': safe_get('balance_classes', True),
        'target_split': safe_get('target_split', 'train'),
        'types': list(safe_get('augmentation_types', ['combined'])),
        'fliplr': safe_get('fliplr', 0.5),
        'degrees': safe_get('degrees', 10),
        'translate': safe_get('translate', 0.1),
        'scale': safe_get('scale', 0.1),
        'hsv_h': safe_get('hsv_h', 0.015),
        'hsv_s': safe_get('hsv_s', 0.7),
        'brightness': safe_get('brightness', 0.2),
        'contrast': safe_get('contrast', 0.2)
    }

# Validation helpers
def validate_service_config(config: ServiceConfig) -> Dict[str, Any]:
    """Validate service configuration"""
    validation = {'valid': True, 'errors': []}
    
    aug_config = config.get_augmentation_config()
    
    # Basic validations
    if not isinstance(aug_config.get('num_variations', 0), int) or aug_config.get('num_variations', 0) <= 0:
        validation['errors'].append("num_variations harus integer positif")
    
    if not isinstance(aug_config.get('target_count', 0), int) or aug_config.get('target_count', 0) <= 0:
        validation['errors'].append("target_count harus integer positif")
    
    if not aug_config.get('types') or not isinstance(aug_config.get('types'), list):
        validation['errors'].append("types harus list yang tidak kosong")
    
    validation['valid'] = len(validation['errors']) == 0
    return validation

def create_pipeline_summary(result: AugmentationResult) -> str:
    """Create human-readable pipeline summary"""
    if result.is_success():
        summary_parts = [
            f"✅ Pipeline berhasil",
            f"📊 Generated: {result.stats.generated_images if result.stats else 0} images",
            f"⏱️ Time: {result.processing_time:.1f}s"
        ]
        
        if result.balance_result and result.balance_result.get('status') != 'skipped':
            classes_needing = result.balance_result.get('classes_needing_aug', 0)
            summary_parts.append(f"⚖️ Balanced: {classes_needing} classes")
        
        return " | ".join(summary_parts)
    else:
        return f"❌ Pipeline gagal: {result.message}"