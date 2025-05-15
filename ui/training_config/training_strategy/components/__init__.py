"""
File: smartcash/ui/training_config/training_strategy/components/__init__.py
Deskripsi: Package untuk komponen UI strategi pelatihan model
"""

# Import dari file-file komponen yang telah dipecah
from smartcash.ui.training_config.training_strategy.components.utils_components import create_training_strategy_utils_components
from smartcash.ui.training_config.training_strategy.components.validation_components import create_training_strategy_validation_components
from smartcash.ui.training_config.training_strategy.components.multiscale_components import create_training_strategy_multiscale_components
from smartcash.ui.training_config.training_strategy.components.button_components import create_training_strategy_button_components
from smartcash.ui.training_config.training_strategy.components.info_panel_components import create_training_strategy_info_panel
from smartcash.ui.training_config.training_strategy.components.main_components import create_training_strategy_ui_components

# Untuk kompatibilitas dengan kode yang sudah ada
# Jika file training_strategy_components.py masih digunakan di tempat lain
try:
    from smartcash.ui.training_config.training_strategy.components.training_strategy_components import *
except ImportError:
    pass

__all__ = [
    'create_training_strategy_ui_components',
    'create_training_strategy_info_panel',
    'create_training_strategy_utils_components',
    'create_training_strategy_validation_components',
    'create_training_strategy_multiscale_components',
    'create_training_strategy_button_components'
]
