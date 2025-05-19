"""
File: smartcash/components/observer/event_topics_observer.py
Deskripsi: Definisi topik event standar dengan kategori yang lebih terkonsolidasi dan struktur yang lebih efisien
"""

class EventTopics:
    """Definisi standar topik event dalam SmartCash dengan struktur namespace hierarkis."""
    
    # Kategori event utama (namespace level 1)
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
    CONFIG = "config"
    BACKUP = "backup"
    ZIP_PROCESSING = "zip"
    EXPORT = "export"
    RESTORE = "restore"
    PROGRESS = "progress"
    DEPENDENCY = "dependency"
    
    # Progress events - dikonsolidasikan ke namespace sendiri
    PROGRESS_START = "progress.start"
    PROGRESS_UPDATE = "progress.update"
    PROGRESS_COMPLETE = "progress.complete"
    
    # Event hierarkis training - menggunakan format namespace konsisten
    TRAINING_START = f"{TRAINING}.start"
    TRAINING_END = f"{TRAINING}.end"
    TRAINING_PROGRESS = f"{TRAINING}.progress"
    TRAINING_ERROR = f"{TRAINING}.error"
    TRAINING_EPOCH_START = f"{TRAINING}.epoch.start"
    TRAINING_EPOCH_END = f"{TRAINING}.epoch.end"
    TRAINING_BATCH_START = f"{TRAINING}.batch.start"
    TRAINING_BATCH_END = f"{TRAINING}.batch.end"
    TRAINING_VALIDATION_START = f"{TRAINING}.validation.start"
    TRAINING_VALIDATION_PROGRESS = f"{TRAINING}.validation.progress"
    TRAINING_VALIDATION_END = f"{TRAINING}.validation.end"
    
    # Event hierarkis evaluation - menggunakan format namespace konsisten
    EVALUATION_START = f"{EVALUATION}.start"
    EVALUATION_END = f"{EVALUATION}.end"
    EVALUATION_PROGRESS = f"{EVALUATION}.progress"
    EVALUATION_BATCH = f"{EVALUATION}.batch"
    EVALUATION_BATCH_END = f"{EVALUATION}.batch.end"
    
    # Event hierarkis detection - menggunakan format namespace konsisten
    DETECTION_START = f"{DETECTION}.start"
    DETECTION_END = f"{DETECTION}.end"
    DETECTION_PROGRESS = f"{DETECTION}.progress"
    DETECTION_OBJECT = f"{DETECTION}.object.detected"
    
    # Event hierarkis preprocessing - menggunakan format namespace konsisten
    PREPROCESSING_START = f"{PREPROCESSING}.start"
    PREPROCESSING_END = f"{PREPROCESSING}.end"
    PREPROCESSING_PROGRESS = f"{PREPROCESSING}.progress"
    PREPROCESSING_CURRENT_PROGRESS = f"{PREPROCESSING}.current_progress"
    PREPROCESSING_STEP_PROGRESS = f"{PREPROCESSING}.step_progress"
    PREPROCESSING_STEP_CHANGE = f"{PREPROCESSING}.step_change"
    PREPROCESSING_ERROR = f"{PREPROCESSING}.error"
    PREPROCESSING_CLEANUP_START = f"{PREPROCESSING}.cleanup.start"
    PREPROCESSING_CLEANUP_END = f"{PREPROCESSING}.cleanup.end"
    PREPROCESSING_CLEANUP_ERROR = f"{PREPROCESSING}.cleanup.error"
    PREPROCESSING_VALIDATION = f"{PREPROCESSING}.validation"
    PREPROCESSING_AUGMENTATION = f"{PREPROCESSING}.augmentation"
    
    # Event hierarkis dataset - menggunakan format namespace konsisten
    DATASET_LOAD = f"{DATASET}.load"
    DATASET_VALIDATE = f"{DATASET}.validate"
    DATASET_SPLIT_START = f"{DATASET}.split.start"
    DATASET_SPLIT_END = f"{DATASET}.split.end"
    DATASET_SPLIT_ERROR = f"{DATASET}.split.error"
    
    # Event hierarkis augmentation - menggunakan format namespace konsisten
    AUGMENTATION_START = f"{AUGMENTATION}.start"
    AUGMENTATION_END = f"{AUGMENTATION}.end"
    AUGMENTATION_PROGRESS = f"{AUGMENTATION}.progress"
    AUGMENTATION_ERROR = f"{AUGMENTATION}.error"
    AUGMENTATION_CURRENT_PROGRESS = f"{AUGMENTATION}.current_progress"
    AUGMENTATION_CLEANUP_START = f"{AUGMENTATION}.cleanup.start"
    AUGMENTATION_CLEANUP_END = f"{AUGMENTATION}.cleanup.end"
    AUGMENTATION_CLEANUP_ERROR = f"{AUGMENTATION}.cleanup.error"
    
    # Event hierarkis dependency installer - menggunakan format namespace konsisten
    DEPENDENCY_INSTALL_START = f"{DEPENDENCY}.install.start"
    DEPENDENCY_INSTALL_PROGRESS = f"{DEPENDENCY}.install.progress"
    DEPENDENCY_INSTALL_COMPLETE = f"{DEPENDENCY}.install.complete"
    DEPENDENCY_INSTALL_ERROR = f"{DEPENDENCY}.install.error"
    
    # Event hierarkis checkpoint - menggunakan format namespace konsisten
    CHECKPOINT_SAVE = f"{CHECKPOINT}.save"
    CHECKPOINT_LOAD = f"{CHECKPOINT}.load"
    CHECKPOINT_BEST_MODEL_SAVED = f"{CHECKPOINT}.best_model.saved"
    
    # Event hierarkis resource - menggunakan format namespace konsisten
    MEMORY_WARNING = f"{RESOURCE}.memory.warning"
    GPU_UTILIZATION = f"{RESOURCE}.gpu.utilization"
    
    # Event hierarkis UI - menggunakan format namespace konsisten
    UI_UPDATE = f"{UI}.update"
    UI_REFRESH = f"{UI}.refresh"
    
    # Event hierarkis download - menggunakan format namespace konsisten
    DOWNLOAD_START = f"{DOWNLOAD}.start"
    DOWNLOAD_END = f"{DOWNLOAD}.end"
    DOWNLOAD_PROGRESS = f"{DOWNLOAD}.progress"
    DOWNLOAD_ERROR = f"{DOWNLOAD}.error"
    DOWNLOAD_COMPLETE = f"{DOWNLOAD}.complete"
    
    # Event hierarkis export - menggunakan format namespace konsisten
    EXPORT_START = f"{EXPORT}.start"
    EXPORT_PROGRESS = f"{EXPORT}.progress"
    EXPORT_COMPLETE = f"{EXPORT}.complete"
    EXPORT_ERROR = f"{EXPORT}.error"
    
    # Event hierarkis upload - digabungkan dengan namespace export
    UPLOAD_START = f"{EXPORT}.upload.start"
    UPLOAD_PROGRESS = f"{EXPORT}.upload.progress"
    UPLOAD_COMPLETE = f"{EXPORT}.upload.complete"
    UPLOAD_ERROR = f"{EXPORT}.upload.error"
    
    # Event hierarkis backup & restore - menggunakan format namespace konsisten
    BACKUP_START = f"{BACKUP}.start"
    BACKUP_PROGRESS = f"{BACKUP}.progress"
    BACKUP_COMPLETE = f"{BACKUP}.complete"
    BACKUP_ERROR = f"{BACKUP}.error"
    RESTORE_START = f"{RESTORE}.start"
    RESTORE_PROGRESS = f"{RESTORE}.progress"
    RESTORE_COMPLETE = f"{RESTORE}.complete"
    RESTORE_ERROR = f"{RESTORE}.error"
    
    # Event hierarkis ZIP processing - menggunakan format namespace konsisten
    ZIP_PROCESSING_START = f"{ZIP_PROCESSING}.start"
    ZIP_PROCESSING_PROGRESS = f"{ZIP_PROCESSING}.progress"
    ZIP_PROCESSING_COMPLETE = f"{ZIP_PROCESSING}.complete"
    ZIP_PROCESSING_ERROR = f"{ZIP_PROCESSING}.error"
    ZIP_EXTRACT_PROGRESS = f"{ZIP_PROCESSING}.extract.progress"
    ZIP_IMPORT_START = f"{ZIP_PROCESSING}.import.start"
    ZIP_IMPORT_PROGRESS = f"{ZIP_PROCESSING}.import.progress"
    ZIP_IMPORT_COMPLETE = f"{ZIP_PROCESSING}.import.complete"
    ZIP_IMPORT_ERROR = f"{ZIP_PROCESSING}.import.error"
    
    # Event hierarkis pull dataset - dibuat sebagai subnamespace dataset
    PULL_DATASET_START = f"{DATASET}.pull.start"
    PULL_DATASET_PROGRESS = f"{DATASET}.pull.progress"
    PULL_DATASET_COMPLETE = f"{DATASET}.pull.complete"
    PULL_DATASET_ERROR = f"{DATASET}.pull.error"
    
    # Event hierarkis config - menggunakan format namespace konsisten
    CONFIG_UPDATED = f"{CONFIG}.updated"
    CONFIG_LOADED = f"{CONFIG}.loaded"
    CONFIG_RESET = f"{CONFIG}.reset"
    CONFIG_ERROR = f"{CONFIG}.error"
    
    @classmethod
    def get_all_topics(cls):
        """Mendapatkan semua topik event yang didefinisikan dengan one-liner."""
        return [v for k, v in cls.__dict__.items() if not k.startswith('_') and isinstance(v, str)]
    
    @classmethod
    def get_topics_by_namespace(cls, namespace: str):
        """Mendapatkan semua topik yang berada dalam namespace tertentu."""
        return [v for k, v in cls.__dict__.items() if not k.startswith('_') and isinstance(v, str) and v.startswith(f"{namespace}.")]