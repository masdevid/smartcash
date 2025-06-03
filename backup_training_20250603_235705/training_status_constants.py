"""
File: smartcash/ui/training/constants/training_status_constants.py
Deskripsi: Konstanta untuk status training dan state management UI
"""

from enum import Enum
from typing import Dict, Any, NamedTuple

# ============================================================================
# TRAINING STATUS ENUMS
# ============================================================================

class TrainingStatus(Enum):
    """Status training dengan state management yang jelas"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    ERROR = "error"
    RESUMING = "resuming"

class TrainingPhase(Enum):
    """Phase training untuk progress tracking"""
    SETUP = "setup"
    DATA_LOADING = "data_loading"
    MODEL_PREP = "model_prep"
    TRAINING = "training"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"
    EVALUATION = "evaluation"
    CLEANUP = "cleanup"

class UIState(Enum):
    """State UI components untuk consistent behavior"""
    READY = "ready"
    BUSY = "busy"
    DISABLED = "disabled"
    ERROR = "error"
    LOADING = "loading"

# ============================================================================
# BUTTON STATE CONFIGURATIONS
# ============================================================================

class ButtonConfig(NamedTuple):
    """Configuration untuk button states"""
    description: str
    disabled: bool
    button_style: str
    tooltip: str

# Button configurations untuk berbagai state
BUTTON_CONFIGS = {
    'start_training': {
        TrainingStatus.IDLE: ButtonConfig(
            "🚀 Mulai Training", False, "success", "Mulai proses training model"
        ),
        TrainingStatus.RUNNING: ButtonConfig(
            "🔄 Training...", True, "warning", "Training sedang berjalan"
        ),
        TrainingStatus.PAUSED: ButtonConfig(
            "▶️ Resume Training", False, "info", "Lanjutkan training yang dihentikan sementara"
        ),
        TrainingStatus.ERROR: ButtonConfig(
            "🔄 Retry Training", False, "danger", "Coba ulang training setelah error"
        )
    },
    'stop_training': {
        TrainingStatus.RUNNING: ButtonConfig(
            "⏹️ Stop Training", False, "danger", "Hentikan training dan simpan checkpoint"
        ),
        TrainingStatus.PAUSED: ButtonConfig(
            "⏹️ Stop Training", False, "danger", "Hentikan training sepenuhnya"
        ),
        TrainingStatus.IDLE: ButtonConfig(
            "⏹️ Stop Training", True, "", "Training belum dimulai"
        )
    },
    'reset_metrics': {
        TrainingStatus.IDLE: ButtonConfig(
            "🔄 Reset Metrics", False, "warning", "Reset training metrics dan chart"
        ),
        TrainingStatus.RUNNING: ButtonConfig(
            "🔄 Reset Metrics", True, "", "Tidak dapat reset saat training"
        ),
        TrainingStatus.COMPLETED: ButtonConfig(
            "🔄 Reset Metrics", False, "warning", "Reset metrics untuk training baru"
        )
    }
}

# ============================================================================
# PROGRESS TRACKING CONSTANTS
# ============================================================================

class ProgressType(Enum):
    """Tipe progress untuk tracking berbagai aspek training"""
    OVERALL = "overall"         # Overall training progress
    EPOCH = "epoch"             # Current epoch progress
    BATCH = "batch"             # Current batch progress  
    VALIDATION = "validation"   # Validation progress
    CHECKPOINT = "checkpoint"   # Checkpoint saving progress
    MODEL_LOADING = "loading"   # Model loading progress

# Progress update thresholds (untuk mengurangi UI spam)
PROGRESS_UPDATE_THRESHOLDS = {
    ProgressType.OVERALL: 1,    # Update setiap 1%
    ProgressType.EPOCH: 5,      # Update setiap 5%
    ProgressType.BATCH: 10,     # Update setiap 10%
    ProgressType.VALIDATION: 5, # Update setiap 5%
    ProgressType.CHECKPOINT: 1, # Update setiap 1%
    ProgressType.MODEL_LOADING: 1  # Update setiap 1%
}

# ============================================================================
# UI COMPONENT VISIBILITY RULES
# ============================================================================

# Component visibility berdasarkan training status
COMPONENT_VISIBILITY = {
    'progress_container': {
        TrainingStatus.IDLE: False,
        TrainingStatus.INITIALIZING: True,
        TrainingStatus.RUNNING: True,
        TrainingStatus.PAUSED: True,
        TrainingStatus.STOPPING: True,
        TrainingStatus.COMPLETED: False,
        TrainingStatus.ERROR: False
    },
    'metrics_accordion': {
        TrainingStatus.IDLE: False,
        TrainingStatus.RUNNING: True,
        TrainingStatus.PAUSED: True,
        TrainingStatus.COMPLETED: True,
        TrainingStatus.ERROR: False
    },
    'chart_output': {
        TrainingStatus.RUNNING: True,
        TrainingStatus.PAUSED: True,
        TrainingStatus.COMPLETED: True
    }
}

# ============================================================================
# STATUS MESSAGES & ICONS
# ============================================================================

# Status icons untuk UI feedback
STATUS_ICONS = {
    TrainingStatus.IDLE: "🏁",
    TrainingStatus.INITIALIZING: "🔄",
    TrainingStatus.RUNNING: "🚀",
    TrainingStatus.PAUSED: "⏸️",
    TrainingStatus.STOPPING: "⏹️",
    TrainingStatus.COMPLETED: "✅",
    TrainingStatus.ERROR: "❌",
    TrainingStatus.RESUMING: "▶️"
}

# Status messages template
STATUS_MESSAGES = {
    TrainingStatus.IDLE: "Siap untuk memulai training",
    TrainingStatus.INITIALIZING: "Inisialisasi model dan data...",
    TrainingStatus.RUNNING: "Training sedang berjalan - Epoch {epoch}/{total_epochs}",
    TrainingStatus.PAUSED: "Training dihentikan sementara",
    TrainingStatus.STOPPING: "Menghentikan training dan menyimpan checkpoint...",
    TrainingStatus.COMPLETED: "Training selesai! Best epoch: {best_epoch}",
    TrainingStatus.ERROR: "Error terjadi selama training: {error}",
    TrainingStatus.RESUMING: "Melanjutkan training dari checkpoint..."
}

# ============================================================================
# TRAINING METRICS CONSTANTS
# ============================================================================

class MetricType(Enum):
    """Kategori metrics untuk display dan formatting"""
    LOSS = "loss"
    ACCURACY = "accuracy"
    LEARNING_RATE = "learning_rate"
    TIME = "time"
    DETECTION = "detection"
    CUSTOM = "custom"

# Metric display configurations
METRIC_DISPLAY_CONFIG = {
    MetricType.LOSS: {
        'color': '#dc3545',
        'icon': '📉',
        'precision': 4,
        'format': 'float',
        'lower_is_better': True
    },
    MetricType.ACCURACY: {
        'color': '#28a745',
        'icon': '📈',
        'precision': 3,
        'format': 'percentage',
        'lower_is_better': False
    },
    MetricType.LEARNING_RATE: {
        'color': '#17a2b8',
        'icon': '📊',
        'precision': 6,
        'format': 'scientific',
        'lower_is_better': False
    },
    MetricType.TIME: {
        'color': '#6c757d',
        'icon': '⏱️',
        'precision': 2,
        'format': 'time',
        'lower_is_better': True
    },
    MetricType.DETECTION: {
        'color': '#6f42c1',
        'icon': '🎯',
        'precision': 3,
        'format': 'float',
        'lower_is_better': False
    }
}

# Metric name mappings untuk UI display
METRIC_NAME_MAPPING = {
    'train_loss': 'Training Loss',
    'val_loss': 'Validation Loss',
    'box_loss': 'Box Loss',
    'obj_loss': 'Objectness Loss',
    'cls_loss': 'Classification Loss',
    'learning_rate': 'Learning Rate',
    'lr': 'Learning Rate',
    'epoch_time': 'Epoch Time',
    'batch_time': 'Batch Time',
    'mAP': 'Mean Average Precision',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1_score': 'F1 Score'
}

# ============================================================================
# ERROR HANDLING CONSTANTS
# ============================================================================

class ErrorSeverity(Enum):
    """Tingkat keparahan error untuk handling yang tepat"""
    LOW = "low"           # Warning, training bisa dilanjutkan
    MEDIUM = "medium"     # Error yang memerlukan restart
    HIGH = "high"         # Error fatal, perlu intervention manual
    CRITICAL = "critical" # Error sistem, perlu restart aplikasi

# Error categories dengan severity level
ERROR_CATEGORIES = {
    'model_error': ErrorSeverity.HIGH,
    'data_error': ErrorSeverity.MEDIUM,
    'config_error': ErrorSeverity.MEDIUM,
    'memory_error': ErrorSeverity.HIGH,
    'gpu_error': ErrorSeverity.HIGH,
    'checkpoint_error': ErrorSeverity.MEDIUM,
    'validation_error': ErrorSeverity.LOW,
    'ui_error': ErrorSeverity.LOW,
    'service_error': ErrorSeverity.MEDIUM
}

# Recovery actions berdasarkan error category
ERROR_RECOVERY_ACTIONS = {
    'model_error': [
        "Rebuild model dengan konfigurasi yang benar",
        "Periksa model architecture compatibility",
        "Reset ke model default"
    ],
    'data_error': [
        "Periksa dataset path dan format",
        "Validasi data loader configuration",
        "Gunakan dummy data untuk testing"
    ],
    'config_error': [
        "Refresh konfigurasi dari YAML files",
        "Reset ke default configuration",
        "Periksa format konfigurasi"
    ],
    'memory_error': [
        "Kurangi batch size",
        "Clear GPU memory cache",
        "Restart training session"
    ],
    'gpu_error': [
        "Switch ke CPU mode",
        "Restart CUDA context",
        "Periksa GPU driver"
    ]
}

# ============================================================================
# CHART & VISUALIZATION CONSTANTS
# ============================================================================

# Chart update frequencies (dalam batches)
CHART_UPDATE_FREQUENCY = {
    'real_time': 1,      # Update setiap batch (heavy)
    'frequent': 5,       # Update setiap 5 batch (recommended)
    'moderate': 10,      # Update setiap 10 batch
    'occasional': 20,    # Update setiap 20 batch (light)
}

# Chart data retention limits
CHART_DATA_LIMITS = {
    'max_points': 1000,       # Maximum data points per metric
    'retention_hours': 24,    # Keep data for 24 hours
    'auto_cleanup': True      # Auto cleanup old data
}

# Color schemes untuk charts
CHART_COLOR_SCHEMES = {
    'default': ['#007bff', '#dc3545', '#28a745', '#ffc107', '#17a2b8'],
    'colorblind_friendly': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'high_contrast': ['#000000', '#ff0000', '#00ff00', '#0000ff', '#ffff00']
}

# ============================================================================
# NOTIFICATION CONSTANTS
# ============================================================================

class NotificationType(Enum):
    """Tipe notifikasi untuk training events"""
    TRAINING_START = "training_start"
    EPOCH_COMPLETE = "epoch_complete"
    BEST_MODEL = "best_model"
    TRAINING_COMPLETE = "training_complete"
    ERROR_OCCURRED = "error_occurred"
    CHECKPOINT_SAVED = "checkpoint_saved"
    EARLY_STOPPING = "early_stopping"

# Notification configurations
NOTIFICATION_CONFIG = {
    NotificationType.TRAINING_START: {
        'icon': '🚀',
        'color': '#28a745',
        'duration': 3000,  # ms
        'sound': False
    },
    NotificationType.BEST_MODEL: {
        'icon': '🏆',
        'color': '#ffc107',
        'duration': 5000,
        'sound': True
    },
    NotificationType.TRAINING_COMPLETE: {
        'icon': '🎉',
        'color': '#28a745',
        'duration': 0,  # Persistent
        'sound': True
    },
    NotificationType.ERROR_OCCURRED: {
        'icon': '❌',
        'color': '#dc3545',
        'duration': 0,  # Persistent
        'sound': True
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_button_config(button_name: str, status: TrainingStatus) -> ButtonConfig:
    """Get button configuration untuk status tertentu"""
    button_configs = BUTTON_CONFIGS.get(button_name, {})
    return button_configs.get(status, ButtonConfig("Unknown", True, "", ""))

def get_component_visibility(component_name: str, status: TrainingStatus) -> bool:
    """Check visibility component untuk status tertentu"""
    visibility_rules = COMPONENT_VISIBILITY.get(component_name, {})
    return visibility_rules.get(status, False)

def get_status_message(status: TrainingStatus, **kwargs) -> str:
    """Get formatted status message"""
    template = STATUS_MESSAGES.get(status, "Unknown status")
    try:
        return template.format(**kwargs)
    except KeyError:
        return template

def get_metric_display_config(metric_name: str) -> Dict[str, Any]:
    """Get display configuration untuk metric tertentu"""
    # Determine metric type berdasarkan nama
    if 'loss' in metric_name.lower():
        metric_type = MetricType.LOSS
    elif any(term in metric_name.lower() for term in ['accuracy', 'precision', 'recall', 'f1', 'map']):
        metric_type = MetricType.ACCURACY
    elif 'lr' in metric_name.lower() or 'learning_rate' in metric_name.lower():
        metric_type = MetricType.LEARNING_RATE
    elif 'time' in metric_name.lower():
        metric_type = MetricType.TIME
    elif any(term in metric_name.lower() for term in ['detection', 'bbox', 'iou']):
        metric_type = MetricType.DETECTION
    else:
        metric_type = MetricType.CUSTOM
    
    return METRIC_DISPLAY_CONFIG.get(metric_type, METRIC_DISPLAY_CONFIG[MetricType.CUSTOM])

def format_metric_value(value: float, metric_name: str) -> str:
    """Format metric value untuk display"""
    config = get_metric_display_config(metric_name)
    precision = config['precision']
    format_type = config['format']
    
    if format_type == 'percentage':
        return f"{value * 100:.{precision}f}%"
    elif format_type == 'scientific' and value < 0.01:
        return f"{value:.{precision}e}"
    elif format_type == 'time':
        return f"{value:.{precision}f}s"
    else:
        return f"{value:.{precision}f}"

def get_metric_improvement_indicator(current: float, previous: float, metric_name: str) -> str:
    """Get improvement indicator untuk metric"""
    config = get_metric_display_config(metric_name)
    lower_is_better = config['lower_is_better']
    
    if current == previous:
        return "➡️"
    elif (current < previous and lower_is_better) or (current > previous and not lower_is_better):
        return "⬆️"  # Improvement
    else:
        return "⬇️"  # Degradation

# Export key constants dan functions
__all__ = [
    'TrainingStatus', 'TrainingPhase', 'UIState', 'ProgressType', 'MetricType',
    'ErrorSeverity', 'NotificationType', 'BUTTON_CONFIGS', 'STATUS_ICONS',
    'STATUS_MESSAGES', 'METRIC_DISPLAY_CONFIG', 'ERROR_CATEGORIES',
    'CHART_UPDATE_FREQUENCY', 'get_button_config', 'get_component_visibility',
    'get_status_message', 'format_metric_value', 'get_metric_improvement_indicator'
]