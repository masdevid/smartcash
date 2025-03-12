# Definisi topik event standar
class EventTopics:
    """Definisi standar topik event dalam SmartCash."""
    
    # Kategori event
    TRAINING = "training"
    EVALUATION = "evaluation"
    DETECTION = "detection"
    PREPROCESSING = "preprocessing"
    CHECKPOINT = "checkpoint"
    RESOURCE = "resource"
    UI = "ui"
    DOWNLOAD = "download"
    DATASET = "dataset"
    AUGMENTATION = "augmentation"
    
    # Training events
    TRAINING_START = "training.start"
    TRAINING_END = "training.end"
    EPOCH_START = "training.epoch.start"
    EPOCH_END = "training.epoch.end"
    BATCH_START = "training.batch.start"
    BATCH_END = "training.batch.end"
    VALIDATION_START = "training.validation.start"
    VALIDATION_PROGRESS = "training.validation.progress"
    VALIDATION_END = "training.validation.end"
    TRAINING_PROGRESS = "training.progress"
    TRAINING_EPOCH_START = "training.epoch.start"
    TRAINING_EPOCH_END = "training.epoch.end"
    TRAINING_BATCH_END = "training.batch.end"
    
    # Evaluation events
    EVALUATION_START = "evaluation.start"
    EVALUATION_END = "evaluation.end"
    EVALUATION_BATCH = "evaluation.batch"
    EVALUATION_BATCH_END = "evaluation.batch.end"
    EVALUATION_PROGRESS = "evaluation.progress"
    
    # Detection events
    DETECTION_START = "detection.start"
    DETECTION_END = "detection.end"
    DETECTION_PROGRESS = "detection.progress"
    OBJECT_DETECTED = "detection.object.detected"
    
    # Preprocessing events
    PREPROCESSING_START = "preprocessing.start"
    PREPROCESSING_END = "preprocessing.end"
    PREPROCESSING_PROGRESS = "preprocessing.progress"
    PREPROCESSING_ERROR = "preprocessing.error"
    VALIDATION_EVENT = "preprocessing.validation"
    AUGMENTATION_EVENT = "preprocessing.augmentation"
    
    # Dataset events
    DATASET_LOAD = "dataset.load"
    DATASET_VALIDATE = "dataset.validate"
    DATASET_SPLIT_START = "dataset.split.start"
    DATASET_SPLIT_END = "dataset.split.end"
    DATASET_SPLIT_ERROR = "dataset.split.error"
    
    # Augmentation events
    AUGMENTATION_START = "augmentation.start"
    AUGMENTATION_END = "augmentation.end"
    AUGMENTATION_PROGRESS = "augmentation.progress"
    AUGMENTATION_ERROR = "augmentation.error"
    
    # Checkpoint events
    CHECKPOINT_SAVE = "checkpoint.save"
    CHECKPOINT_LOAD = "checkpoint.load"
    BEST_MODEL_SAVED = "checkpoint.best_model.saved"
    
    # Resource events
    MEMORY_WARNING = "resource.memory.warning"
    GPU_UTILIZATION = "resource.gpu.utilization"
    
    # UI events
    UI_UPDATE = "ui.update"
    UI_REFRESH = "ui.refresh"
    PROGRESS_UPDATE = "ui.progress.update"
    
    # Download events
    DOWNLOAD_START = "download.start"
    DOWNLOAD_END = "download.end"
    DOWNLOAD_PROGRESS = "download.progress"
    DOWNLOAD_ERROR = "download.error"
    DOWNLOAD_COMPLETE = "download.complete"
    
    @classmethod
    def get_all_topics(cls):
        """Mendapatkan semua topik event yang didefinisikan."""
        return [v for k, v in cls.__dict__.items() if not k.startswith('_') and isinstance(v, str)]