# File: smartcash/utils/observer/__init__.py
# Author: Alfrida Sabar
# Deskripsi: Package initialization untuk observer pattern terkonsolidasi di SmartCash

from smartcash.utils.observer.base_observer import BaseObserver
from smartcash.utils.observer.event_dispatcher import EventDispatcher
from smartcash.utils.observer.event_registry import EventRegistry
from smartcash.utils.observer.observer_manager import ObserverManager
from smartcash.utils.observer.decorators import observable, observe

# Shortcut functions
register = EventDispatcher.register
unregister = EventDispatcher.unregister
notify = EventDispatcher.notify
register_many = EventDispatcher.register_many
unregister_many = EventDispatcher.unregister_many
unregister_all = EventDispatcher.unregister_all

# Standar topik event SmartCash
class EventTopics:
    """Definisi standar topik event dalam SmartCash."""
    
    # Training events
    TRAINING = "training"
    TRAINING_START = "training.start"
    TRAINING_END = "training.end"
    EPOCH_START = "training.epoch.start"
    EPOCH_END = "training.epoch.end"
    BATCH_START = "training.batch.start"
    BATCH_END = "training.batch.end"
    VALIDATION_START = "training.validation.start"
    VALIDATION_END = "training.validation.end"
    
    # Evaluation events
    EVALUATION = "evaluation"
    EVALUATION_START = "evaluation.start"
    EVALUATION_END = "evaluation.end"
    EVALUATION_BATCH = "evaluation.batch"
    
    # Detection events
    DETECTION = "detection"
    DETECTION_START = "detection.start"
    DETECTION_END = "detection.end"
    DETECTION_PROGRESS = "detection.progress"
    
    # Preprocessing events
    PREPROCESSING = "preprocessing"
    PREPROCESSING_START = "preprocessing.start"
    PREPROCESSING_END = "preprocessing.end"
    PREPROCESSING_PROGRESS = "preprocessing.progress"
    VALIDATION_EVENT = "preprocessing.validation"
    AUGMENTATION_EVENT = "preprocessing.augmentation"
    
    # Checkpoint events
    CHECKPOINT = "checkpoint"
    CHECKPOINT_SAVE = "checkpoint.save"
    CHECKPOINT_LOAD = "checkpoint.load"
    
    # Resource events
    RESOURCE = "resource"
    MEMORY_WARNING = "resource.memory.warning"
    GPU_UTILIZATION = "resource.gpu.utilization"
    
    # UI events
    UI = "ui"
    UI_UPDATE = "ui.update"
    UI_REFRESH = "ui.refresh"
    
    # Download events
    DOWNLOAD = "download"
    DOWNLOAD_START = "download.start"
    DOWNLOAD_PROGRESS = "download.progress"
    DOWNLOAD_COMPLETE = "download.complete"
    DOWNLOAD_ERROR = "download.error"
    DOWNLOAD_END = "download.end"
    
    @classmethod
    def get_all_topics(cls):
        """Mendapatkan semua topik event yang didefinisikan."""
        return [v for k, v in cls.__dict__.items() if not k.startswith('_') and isinstance(v, str)]

__all__ = [
    'BaseObserver',
    'EventDispatcher',
    'EventRegistry',
    'ObserverManager',
    'observable',
    'observe',
    'register',
    'unregister',
    'notify',
    'register_many',
    'unregister_many',
    'unregister_all',
    'EventTopics'
]