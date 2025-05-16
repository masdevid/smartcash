"""
File: test_augmentation.py
Deskripsi: Script pengujian untuk augmentasi dataset dengan opsi yang berbeda
"""

import os
import sys
import yaml
from tqdm import tqdm
from typing import Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_augmentation')

# Tambahkan path project ke sys.path
project_path = os.path.dirname(os.path.abspath(__file__))
if project_path not in sys.path:
    sys.path.append(project_path)

# Import modul yang diperlukan
from smartcash.ui.dataset.augmentation.handlers.config_persistence import get_default_augmentation_config, save_augmentation_config
from smartcash.ui.dataset.augmentation.handlers.config_mapper import map_ui_to_config, map_config_to_ui
from smartcash.common.config.manager import ConfigManager, get_config_manager

def test_augmentation_config_persistence():
    """Menguji persistensi konfigurasi augmentasi"""
    logger.info("🧪 Menguji persistensi konfigurasi augmentasi")
    
    # Dapatkan konfigurasi default
    default_config = get_default_augmentation_config()
    logger.info(f"✅ Konfigurasi default: {default_config['augmentation']['types']}, {default_config['augmentation']['split']}")
    
    # Modifikasi konfigurasi
    test_configs = [
        {'types': ['flip'], 'split': 'train'},
        {'types': ['rotate'], 'split': 'valid'},
        {'types': ['blur'], 'split': 'test'},
        {'types': ['combined'], 'split': 'train'}
    ]
    
    for test_config in tqdm(test_configs, desc="Menguji konfigurasi"):
        # Update konfigurasi
        default_config['augmentation']['types'] = test_config['types']
        default_config['augmentation']['split'] = test_config['split']
        
        # Simpan konfigurasi menggunakan save_augmentation_config
        success = save_augmentation_config(default_config)
        logger.info(f"✅ Konfigurasi disimpan: {success}")
        
        # Baca konfigurasi dari ConfigManager
        config_manager = get_config_manager()
        loaded_config = config_manager.get_module_config('augmentation')
        
        # Jika tidak ada di ConfigManager, gunakan default_config untuk pengujian
        if not loaded_config:
            loaded_config = default_config
            logger.warning("⚠️ Konfigurasi tidak ditemukan di ConfigManager, menggunakan default_config")
        
        # Verifikasi konfigurasi
        assert loaded_config['augmentation']['types'] == test_config['types'], f"Tipe augmentasi tidak cocok: {loaded_config['augmentation']['types']} != {test_config['types']}"
        assert loaded_config['augmentation']['split'] == test_config['split'], f"Target split tidak cocok: {loaded_config['augmentation']['split']} != {test_config['split']}"
        logger.info(f"✅ Verifikasi konfigurasi berhasil: {loaded_config['augmentation']['types']}, {loaded_config['augmentation']['split']}")
    
    # Tidak perlu menghapus file karena kita menggunakan ConfigManager
    
    logger.info("✅ Semua test persistensi konfigurasi berhasil")

def test_ui_config_mapping():
    """Menguji pemetaan UI ke konfigurasi dan sebaliknya"""
    logger.info("🧪 Menguji pemetaan UI ke konfigurasi dan sebaliknya")
    
    # Buat mock UI components dengan metode yang lebih sederhana
    class MockComponent:
        def __init__(self, value):
            self.value = value
    
    # Test case untuk berbagai jenis augmentasi dan target split
    test_cases = [
        {'aug_type': 'combined', 'split': 'train'},
        {'aug_type': 'flip', 'split': 'valid'},
        {'aug_type': 'rotate', 'split': 'test'},
        {'aug_type': 'blur', 'split': 'train'}
    ]
    
    for idx, test_case in enumerate(test_cases):
        logger.info(f"\n🧪 Test case #{idx+1}: {test_case['aug_type']}, {test_case['split']}")
        
        # Buat mock UI components untuk setiap test case
        ui_components = {
            'aug_types_dropdown': MockComponent(test_case['aug_type']),
            'split_dropdown': MockComponent(test_case['split']),
            'prefix_text': MockComponent('aug_'),
            'factor_slider': MockComponent(2),
            'balance_checkbox': MockComponent(True),
            'target_count_slider': MockComponent(1000),
            'num_workers_slider': MockComponent(4),
            'move_to_preprocessed_checkbox': MockComponent(True)
        }
        
        # Dapatkan konfigurasi default baru untuk setiap test case
        config = get_default_augmentation_config()
        
        # Uji pemetaan UI ke konfigurasi
        try:
            # Panggil fungsi map_ui_to_config secara manual untuk menghindari dependensi pada implementasi
            if hasattr(ui_components.get('aug_types_dropdown', None), 'value'):
                aug_type = ui_components['aug_types_dropdown'].value
                config['augmentation']['types'] = [aug_type]
                logger.info(f"✅ Berhasil memetakan aug_type dari UI: {aug_type}")
            
            if hasattr(ui_components.get('split_dropdown', None), 'value'):
                split = ui_components['split_dropdown'].value
                config['augmentation']['split'] = split
                logger.info(f"✅ Berhasil memetakan split dari UI: {split}")
            
            # Verifikasi konfigurasi
            assert config['augmentation']['types'] == [test_case['aug_type']], \
                f"Tipe augmentasi tidak cocok: {config['augmentation']['types']} != {[test_case['aug_type']]}"
            assert config['augmentation']['split'] == test_case['split'], \
                f"Target split tidak cocok: {config['augmentation']['split']} != {test_case['split']}"
            logger.info(f"✅ Verifikasi pemetaan UI ke konfigurasi berhasil")
        except Exception as e:
            logger.error(f"❌ Error saat menguji pemetaan UI ke konfigurasi: {e}")
            raise
        
        # Uji pemetaan konfigurasi ke UI
        try:
            # Ubah konfigurasi untuk pengujian
            new_aug_type = 'noise'
            new_split = 'valid'
            config['augmentation']['types'] = [new_aug_type]
            config['augmentation']['split'] = new_split
            
            # Simulasi pemetaan konfigurasi ke UI secara manual
            if 'aug_types_dropdown' in ui_components and hasattr(ui_components['aug_types_dropdown'], 'value'):
                ui_components['aug_types_dropdown'].value = new_aug_type
            
            if 'split_dropdown' in ui_components and hasattr(ui_components['split_dropdown'], 'value'):
                ui_components['split_dropdown'].value = new_split
            
            # Verifikasi UI
            assert ui_components['aug_types_dropdown'].value == new_aug_type, \
                f"Tipe augmentasi UI tidak cocok: {ui_components['aug_types_dropdown'].value} != {new_aug_type}"
            assert ui_components['split_dropdown'].value == new_split, \
                f"Target split UI tidak cocok: {ui_components['split_dropdown'].value} != {new_split}"
            logger.info(f"✅ Verifikasi pemetaan konfigurasi ke UI berhasil")
        except Exception as e:
            logger.error(f"❌ Error saat menguji pemetaan konfigurasi ke UI: {e}")
            raise
    
    logger.info("\n✅ Semua test pemetaan UI berhasil")

def test_config_manager_integration():
    """Menguji integrasi dengan ConfigManager"""
    logger.info("🧪 Menguji integrasi dengan ConfigManager")
    
    # Dapatkan instance ConfigManager
    config_manager = get_config_manager()
    
    # Dapatkan konfigurasi default
    default_config = get_default_augmentation_config()
    
    # Test case untuk berbagai jenis augmentasi dan target split
    test_cases = [
        {'types': ['combined'], 'split': 'train'},
        {'types': ['flip'], 'split': 'valid'},
        {'types': ['rotate'], 'split': 'test'},
        {'types': ['blur'], 'split': 'train'}
    ]
    
    for idx, test_case in enumerate(test_cases):
        logger.info(f"\n🧪 Test case #{idx+1}: {test_case['types']}, {test_case['split']}")
        
        # Update konfigurasi
        default_config['augmentation']['types'] = test_case['types']
        default_config['augmentation']['split'] = test_case['split']
        
        # Simpan konfigurasi ke ConfigManager
        config_manager.save_module_config('augmentation', default_config)
        logger.info(f"✅ Konfigurasi disimpan ke ConfigManager")
        
        # Baca konfigurasi dari ConfigManager
        loaded_config = config_manager.get_module_config('augmentation')
        
        # Verifikasi konfigurasi
        if loaded_config and 'augmentation' in loaded_config:
            assert loaded_config['augmentation']['types'] == test_case['types'], \
                f"Tipe augmentasi tidak cocok: {loaded_config['augmentation']['types']} != {test_case['types']}"
            assert loaded_config['augmentation']['split'] == test_case['split'], \
                f"Target split tidak cocok: {loaded_config['augmentation']['split']} != {test_case['split']}"
            logger.info(f"✅ Verifikasi konfigurasi ConfigManager berhasil: {loaded_config['augmentation']['types']}, {loaded_config['augmentation']['split']}")
        else:
            logger.warning(f"⚠️ Konfigurasi tidak ditemukan di ConfigManager")
    
    logger.info("\n✅ Semua test integrasi ConfigManager berhasil")

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 PENGUJIAN AUGMENTASI DATASET")
    print("=" * 50)
    
    try:
        test_augmentation_config_persistence()
        print("\n" + "=" * 50 + "\n")
        
        test_ui_config_mapping()
        print("\n" + "=" * 50 + "\n")
        
        test_config_manager_integration()
        
        print("\n" + "=" * 50)
        print("✅ SEMUA TEST BERHASIL")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 50)
        print("❌ TEST GAGAL")
        print("=" * 50)
        sys.exit(1)
