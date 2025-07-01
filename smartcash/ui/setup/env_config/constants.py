"""
File: smartcash/ui/setup/env_config/constants.py
Deskripsi: Konstanta untuk environment configuration
"""
from pathlib import Path
from enum import Enum, auto
from smartcash.common.constants.paths import (
    COLAB_PATH, DRIVE_PATH, DATA_ROOT, DEFAULT_CONFIG_DIR,
    COLAB_DATA_ROOT, COLAB_CONFIG_PATH, COLAB_DOWNLOADS_PATH,
    COLAB_TRAIN_PATH, COLAB_VALID_PATH, COLAB_TEST_PATH,
    COLAB_PREPROCESSED_PATH, COLAB_AUGMENTED_PATH, COLAB_BACKUP_PATH,
    COLAB_INVALID_PATH, get_paths_for_environment
)
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.core import APP_NAME

# Get environment info
env_manager = get_environment_manager()
env_paths = get_paths_for_environment(
    is_colab=env_manager.is_colab,
    is_drive_mounted=env_manager.is_drive_mounted
)

# Required folders for SmartCash
REQUIRED_FOLDERS = [
    str(Path(COLAB_PATH) / APP_NAME),  # /content/smartcash
    str(Path(COLAB_DATA_ROOT)),  # /content/smartcash/data
    str(Path(COLAB_DOWNLOADS_PATH)),  # /content/smartcash/data/downloads
    str(Path(COLAB_BACKUP_PATH)),  # /content/smartcash/data/backup
    str(Path(COLAB_TRAIN_PATH)),  # /content/smartcash/data/train
    str(Path(COLAB_TEST_PATH)),  # /content/smartcash/data/test
    str(Path(COLAB_VALID_PATH)),  # /content/smartcash/data/valid
    str(Path(COLAB_INVALID_PATH)),  # /content/smartcash/data/invalid
    str(Path(COLAB_PREPROCESSED_PATH)),  # /content/smartcash/data/preprocessed
    str(Path(COLAB_AUGMENTED_PATH)),  # /content/smartcash/data/augmented
    str(Path(COLAB_PATH) / APP_NAME / 'models'),  # /content/smartcash/models
    str(Path(COLAB_CONFIG_PATH)),  # /content/smartcash/configs
    str(Path(COLAB_PATH) / APP_NAME / 'outputs'),  # /content/smartcash/outputs
    str(Path(COLAB_PATH) / APP_NAME / 'logs')  # /content/smartcash/logs
]

# Source directories that need to be created in Google Drive
SOURCE_DIRECTORIES = [
    str(Path(DRIVE_PATH) / DATA_ROOT),  # /content/drive/MyDrive/SmartCash/data
    str(Path(DRIVE_PATH) / 'models'),  # /content/drive/MyDrive/SmartCash/models
    str(Path(DRIVE_PATH) / DEFAULT_CONFIG_DIR),  # /content/drive/MyDrive/SmartCash/configs
    str(Path(DRIVE_PATH) / 'outputs'),  # /content/drive/MyDrive/SmartCash/outputs
    str(Path(DRIVE_PATH) / 'logs')  # /content/drive/MyDrive/SmartCash/logs
]

# Symlink mapping: source -> target
SYMLINK_MAP = {
    str(Path(DRIVE_PATH) / DATA_ROOT): str(Path(COLAB_PATH) / DATA_ROOT),  # data
    str(Path(DRIVE_PATH) / 'models'): str(Path(COLAB_PATH) / 'models'),  # models
    str(Path(DRIVE_PATH) / DEFAULT_CONFIG_DIR): str(Path(COLAB_PATH) / DEFAULT_CONFIG_DIR),  # configs
    str(Path(DRIVE_PATH) / 'outputs'): str(Path(COLAB_PATH) / 'outputs'),  # outputs
    str(Path(DRIVE_PATH) / 'logs'): str(Path(COLAB_PATH) / 'logs')  # logs
}

# Progress steps
PROGRESS_STEPS = [
    "Mounting Google Drive",
    "Creating directories", 
    "Syncing configurations",
    "Verifying setup"
]

# Status messages
STATUS_MESSAGES = {
    'ready': "Siap untuk setup environment",
    'running': "Setup sedang berjalan...",
    'success': "Setup berhasil diselesaikan!",
    'failed': "Setup gagal, silakan coba lagi"
}


class SetupStage(Enum):
    """Stages for setup progress tracking
    
    The stages must proceed in this specific order:
    INIT → DRIVE_MOUNT → SYMLINK_SETUP → FOLDER_SETUP → CONFIG_SYNC → ENV_SETUP → VERIFY → COMPLETE
    Any stage can transition to ERROR state on failure.
    """
    INIT = auto()            # Initial setup and validation
    DRIVE_MOUNT = auto()     # Mount Google Drive if needed
    SYMLINK_SETUP = auto()   # Setup required symlinks (must be before folder creation)
    FOLDER_SETUP = auto()    # Create required folders
    CONFIG_SYNC = auto()     # Sync configuration files
    ENV_SETUP = auto()       # Environment-specific setup
    VERIFY = auto()          # Verify all setup steps completed successfully
    COMPLETE = auto()        # Successful completion
    ERROR = auto()           # Error state

# Stage weights for progress calculation (must sum to 100)
STAGE_WEIGHTS = {
    SetupStage.INIT: 0,          # 0% - Initialization
    SetupStage.DRIVE_MOUNT: 10,  # 10% - Drive mounting
    SetupStage.SYMLINK_SETUP: 20, # 30% - Symlink setup (before folder creation)
    SetupStage.FOLDER_SETUP: 20,  # 50% - Folder creation
    SetupStage.CONFIG_SYNC: 15,   # 65% - Config sync
    SetupStage.ENV_SETUP: 15,     # 80% - Environment setup
    SetupStage.VERIFY: 5,         # 85% - Verification
    SetupStage.COMPLETE: 15,      # 100% - Finalization (weight is not used, just for completion)
    SetupStage.ERROR: 0           # 0% - Error state
}

# Validate stage weights
assert sum(STAGE_WEIGHTS[stage] for stage in SetupStage) == 100, "Stage weights must sum to 100"