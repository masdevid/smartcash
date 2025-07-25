"""
file_path: smartcash/ui/setup/colab/constants.py
Deskripsi: Konstanta untuk proses setup lingkungan khusus di Google Colab.

File ini hasil refaktor dari `smartcash.ui.setup.env_config.constants` agar
penamaan modul lebih merepresentasikan tujuan penggunaannya (Colab). Seluruh
konstanta dan logika bisnis dipertahankan apa adanya untuk menjaga kompatibilitas
behaviour.
"""

from __future__ import annotations

# Kipasi modul pihak ketiga & internal
from pathlib import Path
from enum import Enum, auto

from smartcash.common.constants.paths import (
    COLAB_PATH,
    DRIVE_PATH,
    DATA_ROOT,
    DEFAULT_CONFIG_DIR,
    COLAB_DATA_ROOT,
    COLAB_CONFIG_PATH,
    COLAB_DOWNLOADS_PATH,
    COLAB_TRAIN_PATH,
    COLAB_VALID_PATH,
    COLAB_TEST_PATH,
    COLAB_PREPROCESSED_PATH,
    COLAB_AUGMENTED_PATH,
    COLAB_BACKUP_PATH,
    COLAB_INVALID_PATH,
    get_paths_for_environment,
)
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.core import APP_NAME

# ---------------------------------------------------------------------------
# Environment Aware Paths & Folders
# ---------------------------------------------------------------------------

env_manager = get_environment_manager()

env_paths = get_paths_for_environment(
    is_colab=env_manager.is_colab,
    is_drive_mounted=env_manager.is_drive_mounted,
)

# Folder yang wajib tersedia di runtime Colab SmartCash
REQUIRED_FOLDERS: list[str] = [
    str(Path(COLAB_PATH)),  # /content/
    str(Path(COLAB_DATA_ROOT)),  # /content/data
    str(Path(COLAB_DOWNLOADS_PATH)),  # /content/data/downloads
    str(Path(COLAB_BACKUP_PATH)),  # /content/data/backup
    str(Path(COLAB_TRAIN_PATH)),  # /content/data/train
    str(Path(COLAB_TEST_PATH)),  # /content/data/test
    str(Path(COLAB_VALID_PATH)),  # /content/data/valid
    str(Path(COLAB_INVALID_PATH)),  # /content/data/invalid
    str(Path(COLAB_PREPROCESSED_PATH)),  # /content/data/preprocessed
    str(Path(COLAB_AUGMENTED_PATH)),  # /content/data/augmented
    str(Path(COLAB_DATA_ROOT) / "pretrained"),  # /content/data/pretrained
    str(Path(COLAB_PATH) / "models"),  # /content/models
    str(Path(COLAB_CONFIG_PATH)),  # /content/configs
    str(Path(COLAB_PATH) / "outputs"),  # /content/outputs
    str(Path(COLAB_PATH) / "logs"),  # /content/logs
]

# Direktori sumber di Google Drive yang harus ada
SOURCE_DIRECTORIES: list[str] = [
    str(Path(DRIVE_PATH) / DATA_ROOT),  # /content/drive/MyDrive/SmartCash/data
    str(Path(DRIVE_PATH) / "models"),  # /content/drive/MyDrive/SmartCash/models
    str(Path(DRIVE_PATH) / DEFAULT_CONFIG_DIR),  # /content/drive/MyDrive/SmartCash/configs
    str(Path(DRIVE_PATH) / "outputs"),  # /content/drive/MyDrive/SmartCash/outputs
    str(Path(DRIVE_PATH) / "logs"),  # /content/drive/MyDrive/SmartCash/logs
]

# Pemetaan symlink => {source: target}
SYMLINK_MAP: dict[str, str] = {
    str(Path(DRIVE_PATH) / DATA_ROOT): str(Path(COLAB_PATH) / DATA_ROOT),
    str(Path(DRIVE_PATH) / "models"): str(Path(COLAB_PATH) / "models"),
    str(Path(DRIVE_PATH) / DEFAULT_CONFIG_DIR): str(Path(COLAB_PATH) / DEFAULT_CONFIG_DIR),
    str(Path(DRIVE_PATH) / "outputs"): str(Path(COLAB_PATH) / "outputs"),
    str(Path(DRIVE_PATH) / "logs"): str(Path(COLAB_PATH) / "logs"),
}

# Tahapan progress setup
PROGRESS_STEPS: list[str] = [
    "Mounting Google Drive",
    "Creating directories",
    "Syncing configurations",
    "Verifying setup",
]

# Pesan status user friendly
STATUS_MESSAGES: dict[str, str] = {
    "ready": "Siap untuk setup environment",
    "running": "Setup sedang berjalan...",
    "success": "Setup berhasil diselesaikan!",
    "failed": "Setup gagal, silakan coba lagi",
}


class SetupStage(Enum):
    """Tahapan setup untuk progress tracking.

    Urutan harus berprogres sebagai berikut:
    INIT → DRIVE_MOUNT → SYMLINK_SETUP → FOLDER_SETUP → CONFIG_SYNC → ENV_SETUP → VERIFY → COMPLETE
    Tiap tahap dapat berpindah ke ERROR jika terjadi kegagalan.
    """

    INIT = auto()
    DRIVE_MOUNT = auto()
    SYMLINK_SETUP = auto()
    FOLDER_SETUP = auto()
    CONFIG_SYNC = auto()
    ENV_SETUP = auto()
    VERIFY = auto()
    COMPLETE = auto()
    ERROR = auto()


# Bobot tiap stage (total harus 100)
STAGE_WEIGHTS: dict[SetupStage, int] = {
    SetupStage.INIT: 0,  # 0% - Initialization
    SetupStage.DRIVE_MOUNT: 10,  # 10% - Drive mounting
    SetupStage.SYMLINK_SETUP: 20,  # 30% cumulative - Symlink setup
    SetupStage.FOLDER_SETUP: 20,  # 50% - Folder creation
    SetupStage.CONFIG_SYNC: 15,  # 65% - Config sync
    SetupStage.ENV_SETUP: 15,  # 80% - Environment setup
    SetupStage.VERIFY: 5,  # 85% - Verification
    SetupStage.COMPLETE: 15,  # 100% - Finalization (informative)
    SetupStage.ERROR: 0,  # 0% - Error state
}

# Validasi agar bobot 100
assert (
    sum(STAGE_WEIGHTS[stage] for stage in SetupStage) == 100
), "Stage weights must sum to 100"

# ===========================================================================
# UI Configuration
# ===========================================================================

# UI Configuration
UI_CONFIG = {
    'title': 'Colab Environment Setup',
    'subtitle': 'Configure Google Colab environment for SmartCash',
    'icon': '🚀',
    'module_name': 'colab',
    'parent_module': 'setup',
    'version': '1.0.0'
}

# Button Configuration - Single primary action for Colab setup
BUTTON_CONFIG = {
    'colab_setup': {
        'text': '🚀 Initialize Environment',
        'style': 'primary',
        'tooltip': 'Start Colab environment setup with phases',
        'order': 1
    }
}

# Validation rules for form fields
VALIDATION_RULES = {
    'project_name': {'required': True, 'min_length': 1},
    'environment_type': {'required': True, 'allowed_values': ['colab', 'kaggle', 'local']}
}

# Colab setup phases for UI display
COLAB_PHASES = {
    'init': {'text': 'Menginisialisasi...', 'style': 'primary'},
    'drive': {'text': 'Mounting Google Drive...', 'style': 'primary'},
    'symlink': {'text': 'Menyiapkan symlink...', 'style': 'primary'},
    'folders': {'text': 'Membuat folder...', 'style': 'primary'},
    'config': {'text': 'Menyiapkan konfigurasi...', 'style': 'primary'},
    'env': {'text': 'Menyiapkan environment...', 'style': 'primary'},
    'verify': {'text': 'Memverifikasi setup...', 'style': 'primary'},
    'complete': {'text': 'Setup Selesai!', 'style': 'success'},
    'error': {'text': 'Terjadi Kesalahan', 'style': 'danger'}
}
