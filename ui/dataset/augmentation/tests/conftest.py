"""
File: smartcash/ui/dataset/augmentation/tests/conftest.py
Deskripsi: Konfigurasi untuk pengujian augmentasi dataset
"""

import sys
import os
from unittest.mock import MagicMock

# Mock modul smartcash.common dan sub-modulnya
def mock_smartcash_common():
    """
    Mock modul smartcash.common dan sub-modulnya yang dibutuhkan untuk pengujian.
    """
    # Buat mock untuk smartcash.common
    if 'smartcash' not in sys.modules:
        sys.modules['smartcash'] = MagicMock()
    
    # Buat mock untuk smartcash.common
    mock_common = MagicMock()
    sys.modules['smartcash'].common = mock_common
    sys.modules['smartcash.common'] = mock_common
    
    # Buat mock untuk smartcash.common.config
    mock_config = MagicMock()
    mock_common.config = mock_config
    sys.modules['smartcash.common.config'] = mock_config
    
    # Buat mock untuk smartcash.common.config.manager
    mock_manager = MagicMock()
    mock_config.manager = mock_manager
    sys.modules['smartcash.common.config.manager'] = mock_manager
    
    # Buat mock untuk ConfigManager
    mock_config_manager = MagicMock()
    mock_manager.ConfigManager = mock_config_manager
    
    # Buat mock untuk smartcash.common.logger
    mock_logger = MagicMock()
    mock_common.logger = mock_logger
    sys.modules['smartcash.common.logger'] = mock_logger
    
    # Buat mock untuk get_logger
    mock_get_logger = MagicMock(return_value=MagicMock())
    mock_logger.get_logger = mock_get_logger
    
    # Buat mock untuk smartcash.common.io
    mock_io = MagicMock()
    mock_common.io = mock_io
    sys.modules['smartcash.common.io'] = mock_io

# Mock modul smartcash.ui.dataset.augmentation.handlers dan sub-modulnya
def mock_augmentation_handlers():
    """
    Mock modul smartcash.ui.dataset.augmentation.handlers dan sub-modulnya yang dibutuhkan untuk pengujian.
    """
    # Pastikan smartcash.ui.dataset.augmentation sudah ada
    if 'smartcash' not in sys.modules:
        sys.modules['smartcash'] = MagicMock()
    if 'smartcash.ui' not in sys.modules:
        sys.modules['smartcash.ui'] = MagicMock()
        sys.modules['smartcash'].ui = sys.modules['smartcash.ui']
    if 'smartcash.ui.dataset' not in sys.modules:
        sys.modules['smartcash.ui.dataset'] = MagicMock()
        sys.modules['smartcash.ui'].dataset = sys.modules['smartcash.ui.dataset']
    if 'smartcash.ui.dataset.augmentation' not in sys.modules:
        sys.modules['smartcash.ui.dataset.augmentation'] = MagicMock()
        sys.modules['smartcash.ui.dataset'].augmentation = sys.modules['smartcash.ui.dataset.augmentation']
    
    # Buat mock untuk handlers
    mock_handlers = MagicMock()
    sys.modules['smartcash.ui.dataset.augmentation.handlers'] = mock_handlers
    sys.modules['smartcash.ui.dataset.augmentation'].handlers = mock_handlers
    
    # Buat mock untuk sub-modul handlers
    handler_modules = [
        'status_handler',
        'button_handlers',
        'execution_handler',
        'initialization_handler',
        'persistence_handler',
        'visualization_handler',
        'config_handler',
        'cleanup_handler',
        'augmentation_service_handler',
        'parameter_handler'
    ]
    
    for module_name in handler_modules:
        mock_module = MagicMock()
        full_module_name = f'smartcash.ui.dataset.augmentation.handlers.{module_name}'
        sys.modules[full_module_name] = mock_module
        setattr(mock_handlers, module_name, mock_module)
        
        # Konfigurasi fungsi-fungsi khusus untuk setiap modul
        if module_name == 'config_handler':
            # Fungsi get_config_from_ui mengembalikan dict
            mock_module.get_config_from_ui = MagicMock(return_value={
                'augmentation': {
                    'enabled': True,
                    'types': ['combined'],
                    'num_variations': 2,
                    'position': {'fliplr': 0.5, 'degrees': 15},
                    'lighting': {'hsv_h': 0.025, 'hsv_s': 0.7},
                    'target_count': 1000,
                    'output_prefix': 'aug'
                }
            })
            
            # Fungsi get_default_augmentation_config mengembalikan dict
            mock_module.get_default_augmentation_config = MagicMock(return_value={
                'augmentation': {
                    'enabled': True,
                    'types': ['combined'],
                    'num_variations': 2,
                    'position': {'fliplr': 0.5, 'degrees': 15},
                    'lighting': {'hsv_h': 0.025, 'hsv_s': 0.7},
                    'target_count': 1000,
                    'output_prefix': 'aug'
                }
            })
            
            # Fungsi load_augmentation_config mengembalikan dict
            mock_module.load_augmentation_config = MagicMock(return_value={
                'augmentation': {
                    'enabled': True,
                    'types': ['combined'],
                    'num_variations': 2,
                    'position': {'fliplr': 0.5, 'degrees': 15},
                    'lighting': {'hsv_h': 0.025, 'hsv_s': 0.7},
                    'target_count': 1000,
                    'output_prefix': 'aug'
                }
            })
            
            # Fungsi save_augmentation_config mengembalikan boolean
            mock_module.save_augmentation_config = MagicMock(return_value=True)
            
            # Fungsi update_ui_from_config tidak mengembalikan nilai tetapi memperbarui UI
            def update_ui_mock(ui_components, config):
                if 'enabled_checkbox' in ui_components:
                    ui_components['enabled_checkbox'].value = True
                if 'types_dropdown' in ui_components:
                    ui_components['types_dropdown'].value = ['combined']
                if 'num_variations_slider' in ui_components:
                    ui_components['num_variations_slider'].value = 2
                return None
            mock_module.update_ui_from_config = MagicMock(side_effect=update_ui_mock)
            
            # Fungsi update_config_from_ui mengembalikan dict
            mock_module.update_config_from_ui = MagicMock(return_value={
                'augmentation': {
                    'enabled': True,
                    'types': ['combined'],
                    'num_variations': 2,
                    'target_count': 1000,
                    'output_prefix': 'aug'
                }
            })
            
        elif module_name == 'persistence_handler':
            # Fungsi load_config_from_file mengembalikan dict
            mock_module.load_config_from_file = MagicMock(return_value={
                'enabled': True,
                'types': ['combined']
            })
            
            # Fungsi register_ui_components tidak mengembalikan nilai
            mock_module.register_ui_components = MagicMock()
            
            # Fungsi reset_config_to_default tidak mengembalikan nilai
            mock_module.reset_config_to_default = MagicMock()
            
            # Buat mock untuk ConfigManager
            mock_config_manager = MagicMock()
            mock_config_manager.get_module_config = MagicMock(return_value={
                'enabled': True,
                'types': ['combined']
            })
            mock_config_manager.save_module_config = MagicMock()
            
            # Pastikan mock_module.ConfigManager mengembalikan instance yang sama
            mock_module.ConfigManager = MagicMock(return_value=mock_config_manager)
            
        elif module_name == 'parameter_handler':
            # Fungsi map_config_to_ui tidak mengembalikan nilai
            mock_module.map_config_to_ui = MagicMock()
            
            # Fungsi map_ui_to_config mengembalikan dict
            mock_module.map_ui_to_config = MagicMock(return_value={
                'augmentation': {
                    'enabled': True,
                    'types': ['combined']
                }
            })
            
            # Fungsi validate_augmentation_params mengembalikan dict dengan status
            mock_module.validate_augmentation_params = MagicMock(return_value={
                'status': 'success',
                'message': 'Parameter valid'
            })

# Mock semua dependensi eksternal sebelum diimpor oleh modul lain
def mock_all_dependencies():
    """
    Mock semua dependensi eksternal yang dibutuhkan untuk pengujian.
    """
    # Mock modul smartcash.common dan sub-modulnya
    mock_smartcash_common()
    
    # Mock modul smartcash.ui.dataset.augmentation.handlers dan sub-modulnya
    mock_augmentation_handlers()
    
    # Daftar modul yang akan di-mock
    modules_to_mock = [
        'cv2',
        'matplotlib',
        'matplotlib.pyplot',
        'seaborn',
        'albumentations',
        'pandas',
        'numpy',
        'torch',
        'torchvision'
    ]
    
    # Mock semua modul
    for module_name in modules_to_mock:
        mock_module = MagicMock()
        sys.modules[module_name] = mock_module
        
        # Tambahkan sub-modul jika diperlukan
        if module_name == 'matplotlib':
            sys.modules['matplotlib.colors'] = MagicMock()
            sys.modules['matplotlib.figure'] = MagicMock()
            sys.modules['matplotlib.image'] = MagicMock()
        
        # Tambahkan fungsi-fungsi khusus jika diperlukan
        if module_name == 'cv2':
            mock_module.imread = MagicMock(return_value=MagicMock())
            mock_module.imwrite = MagicMock(return_value=True)
            mock_module.resize = MagicMock(return_value=MagicMock())
            mock_module.cvtColor = MagicMock(return_value=MagicMock())
            mock_module.COLOR_BGR2RGB = 4
            mock_module.COLOR_RGB2BGR = 4
        
        if module_name == 'matplotlib.pyplot':
            mock_module.figure = MagicMock(return_value=MagicMock())
            mock_module.subplot = MagicMock(return_value=MagicMock())
            mock_module.imshow = MagicMock()
            mock_module.title = MagicMock()
            mock_module.axis = MagicMock()
            mock_module.show = MagicMock()
            mock_module.savefig = MagicMock()
        
        if module_name == 'numpy':
            mock_module.array = MagicMock(return_value=MagicMock())
            mock_module.zeros = MagicMock(return_value=MagicMock())
            mock_module.ones = MagicMock(return_value=MagicMock())
            mock_module.random = MagicMock()
            mock_module.random.rand = MagicMock(return_value=MagicMock())

# Jalankan mock saat modul ini diimpor
mock_all_dependencies()
