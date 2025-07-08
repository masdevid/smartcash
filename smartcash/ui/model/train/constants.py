"""
File: smartcash/ui/model/train/constants.py
Training module constants and configuration.
"""

from enum import Enum
from typing import Dict, Any, List

class TrainingOperation(Enum):
    """Training operations enumeration."""
    START = "start"
    STOP = "stop"
    RESUME = "resume"

class TrainingPhase(Enum):
    """Training phases enumeration."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"

class MetricType(Enum):
    """Metric types for charts."""
    LOSS = "loss"
    MAP = "map"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    F1 = "f1"

class ChartType(Enum):
    """Chart types for visualization."""
    LINE = "line"
    BAR = "bar"
    AREA = "area"

# Operation progress steps
OPERATION_PROGRESS_STEPS = {
    TrainingOperation.START.value: [
        "Initializing training configuration",
        "Loading and validating model",
        "Preparing data loaders",
        "Setting up optimizer and scheduler",
        "Starting training loop",
        "Training in progress"
    ],
    TrainingOperation.STOP.value: [
        "Stopping current training",
        "Saving current state",
        "Cleanup and finalization"
    ],
    TrainingOperation.RESUME.value: [
        "Loading checkpoint",
        "Restoring training state",
        "Validating loaded model",
        "Resuming training loop"
    ]
}

# Default training configuration
DEFAULT_CONFIG = {
    "training": {
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.001,
        "validation_interval": 1,
        "save_interval": 5,
        "early_stopping": {
            "enabled": True,
            "patience": 15,
            "metric": "val_map50",
            "mode": "max"
        }
    },
    "optimizer": {
        "type": "adam",
        "weight_decay": 0.0005,
        "momentum": 0.9
    },
    "scheduler": {
        "type": "cosine",
        "warmup_epochs": 5,
        "min_lr": 0.00001
    },
    "data": {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True
    },
    "mixed_precision": {
        "enabled": True,
        "opt_level": "O1"
    },
    "paths": {
        "data_dir": "/data",
        "checkpoints_dir": "/data/checkpoints",
        "logs_dir": "/data/logs/training"
    }
}

# UI Configuration
UI_CONFIG = {
    "title": "🚀 Model Training",
    "subtitle": "Train YOLOv5 models with real-time monitoring and metrics visualization",
    "icon": "🤖",
    "theme": "primary"
}

# Chart configuration
CHART_CONFIG = {
    "loss_metrics": {
        "title": "Training Loss",
        "color": "#ff6b6b",
        "metrics": ["train_loss", "val_loss", "obj_loss", "cls_loss", "box_loss"],
        "y_label": "Loss Value"
    },
    "performance_metrics": {
        "title": "Performance Metrics", 
        "color": "#4ecdc4",
        "metrics": ["val_map50", "val_map75", "accuracy", "precision", "f1"],
        "y_label": "Score"
    }
}

# Training status indicators
STATUS_INDICATORS = {
    TrainingPhase.IDLE.value: {
        "icon": "⏸️",
        "color": "gray",
        "message": "Ready to start training"
    },
    TrainingPhase.INITIALIZING.value: {
        "icon": "🔄",
        "color": "blue", 
        "message": "Initializing training components"
    },
    TrainingPhase.TRAINING.value: {
        "icon": "🚀",
        "color": "green",
        "message": "Training in progress"
    },
    TrainingPhase.VALIDATING.value: {
        "icon": "📊",
        "color": "orange",
        "message": "Running validation"
    },
    TrainingPhase.COMPLETED.value: {
        "icon": "✅",
        "color": "success",
        "message": "Training completed successfully"
    },
    TrainingPhase.STOPPED.value: {
        "icon": "🛑",
        "color": "warning",
        "message": "Training stopped by user"
    },
    TrainingPhase.ERROR.value: {
        "icon": "❌",
        "color": "danger",
        "message": "Training error occurred"
    }
}

# Default metrics for tracking
DEFAULT_METRICS = {
    "loss_metrics": {
        "train_loss": 0.0,
        "val_loss": 0.0,
        "obj_loss": 0.0,
        "cls_loss": 0.0,
        "box_loss": 0.0
    },
    "performance_metrics": {
        "val_map50": 0.0,
        "val_map75": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "f1": 0.0
    },
    "training_info": {
        "current_epoch": 0,
        "total_epochs": 0,
        "learning_rate": 0.0,
        "batch_time": 0.0,
        "eta": "00:00:00"
    }
}

# Training configuration validation rules
VALIDATION_RULES = {
    "epochs": {"min": 1, "max": 1000, "type": int},
    "batch_size": {"min": 1, "max": 256, "type": int},
    "learning_rate": {"min": 1e-6, "max": 1.0, "type": float},
    "validation_interval": {"min": 1, "max": 50, "type": int},
    "save_interval": {"min": 1, "max": 100, "type": int},
    "num_workers": {"min": 0, "max": 16, "type": int}
}

# Info content for footer
INFO_CONTENT = {
    "training_tips": [
        "Monitor loss curves for signs of overfitting",
        "Use early stopping to prevent overtraining",
        "Adjust learning rate if loss plateaus",
        "Check validation metrics regularly"
    ],
    "optimization_tips": [
        "Mixed precision training reduces memory usage",
        "Increase batch size for better gradient estimates",
        "Use data parallel training for multiple GPUs",
        "Monitor GPU utilization for optimal performance"
    ]
}