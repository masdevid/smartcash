"""
File: smartcash/ui/dataset/augmentation/handlers/config_mapper.py
Deskripsi: Mapper untuk memetakan nilai UI ke konfigurasi augmentasi dan sebaliknya
"""

from typing import Dict, Any, Optional, List, Union
import ipywidgets as widgets
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.augmentation.handlers.config_validator import validate_augmentation_config

logger = get_logger("augmentation_mapper")

def map_ui_to_config(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Memetakan nilai komponen UI ke konfigurasi augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi yang akan diupdate
        
    Returns:
        Konfigurasi yang sudah diupdate
    """
    # Log konfigurasi awal yang akan diupdate
    logger.debug(f"ℹ️ Konfigurasi awal yang akan diupdate: {config}")
    
    if config is None:
        config = {}
        logger.debug("ℹ️ Konfigurasi awal kosong, membuat konfigurasi baru")
    
    # Pastikan struktur dasar tersedia
    if 'augmentation' not in config:
        config['augmentation'] = {}
        logger.debug("ℹ️ Struktur augmentation tidak ditemukan, membuat baru")
    
    # Simpan referensi ke konfigurasi augmentasi asli
    aug_config = config['augmentation']
    
    try:
        # Dapatkan komponen UI
        aug_options = ui_components.get('aug_options')
        if not aug_options or not hasattr(aug_options, 'children') or len(aug_options.children) < 2:
            logger.warning("⚠️ Komponen aug_options tidak valid, menggunakan konfigurasi yang ada")
            return config
        
        # Dapatkan tab container
        tab = aug_options.children[1]
        if not hasattr(tab, 'children') or len(tab.children) < 2:
            logger.warning("⚠️ Tab container tidak valid, menggunakan konfigurasi yang ada")
            return config
        
        # Dapatkan tab dasar dan lanjutan
        basic_tab = tab.children[0]
        advanced_tab = tab.children[1]
        
        if not hasattr(basic_tab, 'children') or not hasattr(advanced_tab, 'children'):
            logger.warning("⚠️ Tab dasar atau lanjutan tidak valid, menggunakan konfigurasi yang ada")
            return config
        
        # Log nilai UI sebelum ekstraksi
        logger.debug("ℹ️ Mulai ekstraksi nilai dari UI components")
        
        # Ekstrak nilai dari tab dasar
        if len(basic_tab.children) >= 4:
            # Dapatkan jenis augmentasi (children[0]) - sekarang menggunakan SelectMultiple
            if hasattr(basic_tab.children[0], 'value'):
                aug_types_value = list(basic_tab.children[0].value)  # Konversi tuple ke list
                logger.debug(f"ℹ️ Nilai aug_types dari UI: {aug_types_value}")
                
                # Jika tidak ada yang dipilih, gunakan default 'combined'
                if not aug_types_value:
                    aug_types_value = ['combined']
                    logger.debug(f"ℹ️ Tidak ada jenis augmentasi yang dipilih, menggunakan default: {aug_types_value}")
                    
                # Simpan sebagai list untuk kompatibilitas dengan service
                aug_config['types'] = aug_types_value
            else:
                logger.warning("⚠️ Komponen aug_types tidak memiliki atribut value")
                
            # Prefix (children[1])
            if hasattr(basic_tab.children[1], 'value'):
                prefix_value = basic_tab.children[1].value
                logger.debug(f"ℹ️ Nilai prefix dari UI: {prefix_value}")
                aug_config['prefix'] = prefix_value  # Untuk backward compatibility
                aug_config['output_prefix'] = prefix_value  # Untuk service
            else:
                logger.warning("⚠️ Komponen prefix tidak memiliki atribut value")
            
            # Jumlah variasi (children[2])
            if hasattr(basic_tab.children[2], 'value'):
                try:
                    factor_value = int(basic_tab.children[2].value)
                    logger.debug(f"ℹ️ Nilai factor dari UI: {factor_value}")
                    aug_config['factor'] = factor_value  # Untuk backward compatibility
                    aug_config['num_variations'] = factor_value  # Untuk service
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Gagal mengkonversi nilai factor: {e}")
            else:
                logger.warning("⚠️ Komponen factor tidak memiliki atribut value")
        else:
            logger.warning(f"⚠️ Tab dasar tidak memiliki cukup children: {len(basic_tab.children) if hasattr(basic_tab, 'children') else 'tidak ada children'}")
        
        # Ekstrak nilai dari tab lanjutan
        if len(advanced_tab.children) >= 4:
            # Balance kelas (children[0])
            if hasattr(advanced_tab.children[0], 'value'):
                try:
                    balance_value = bool(advanced_tab.children[0].value)
                    logger.debug(f"ℹ️ Nilai balance dari UI: {balance_value}")
                    aug_config['balance_classes'] = balance_value  # Untuk backward compatibility
                    aug_config['target_balance'] = balance_value  # Untuk service
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Gagal mengkonversi nilai balance: {e}")
            else:
                logger.warning("⚠️ Komponen balance tidak memiliki atribut value")
            
            # Target count (children[1])
            if hasattr(advanced_tab.children[1], 'value'):
                try:
                    target_count = int(advanced_tab.children[1].value)
                    logger.debug(f"ℹ️ Nilai target_count dari UI: {target_count}")
                    aug_config['target_count'] = target_count
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Gagal mengkonversi nilai target_count: {e}")
            else:
                logger.warning("⚠️ Komponen target_count tidak memiliki atribut value")
            
            # Num workers (children[2])
            if hasattr(advanced_tab.children[2], 'value'):
                try:
                    num_workers = int(advanced_tab.children[2].value)
                    logger.debug(f"ℹ️ Nilai num_workers dari UI: {num_workers}")
                    aug_config['num_workers'] = num_workers
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Gagal mengkonversi nilai num_workers: {e}")
            else:
                logger.warning("⚠️ Komponen num_workers tidak memiliki atribut value")
            
            # Move to preprocessed (children[3])
            if hasattr(advanced_tab.children[3], 'value'):
                try:
                    move_to_preprocessed = bool(advanced_tab.children[3].value)
                    logger.debug(f"ℹ️ Nilai move_to_preprocessed dari UI: {move_to_preprocessed}")
                    aug_config['move_to_preprocessed'] = move_to_preprocessed
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Gagal mengkonversi nilai move_to_preprocessed: {e}")
            else:
                logger.warning("⚠️ Komponen move_to_preprocessed tidak memiliki atribut value")
        else:
            logger.warning(f"⚠️ Tab lanjutan tidak memiliki cukup children: {len(advanced_tab.children) if hasattr(advanced_tab, 'children') else 'tidak ada children'}")
        
        # Ekstrak nilai jenis augmentasi (jika tersedia)
        aug_types_dropdown = ui_components.get('aug_types_dropdown')
        if aug_types_dropdown and hasattr(aug_types_dropdown, 'value'):
            try:
                aug_types = aug_types_dropdown.value
                logger.debug(f"ℹ️ Nilai aug_types dari UI: {aug_types}")
                # Konversi ke list jika string
                if isinstance(aug_types, str):
                    aug_types = [aug_types]
                aug_config['types'] = aug_types
            except Exception as e:
                logger.warning(f"⚠️ Gagal mengkonversi nilai aug_types: {e}")
                aug_config['types'] = ['combined']
        else:
            logger.debug("ℹ️ Komponen aug_types_dropdown tidak ditemukan, menggunakan nilai default")
            aug_config['types'] = ['combined']
            
        # Ekstrak nilai target split (jika tersedia)
        split_dropdown = ui_components.get('split_dropdown')
        if split_dropdown and hasattr(split_dropdown, 'value'):
            try:
                split = split_dropdown.value
                logger.debug(f"ℹ️ Nilai split dari UI: {split}")
                aug_config['split'] = split
            except Exception as e:
                logger.warning(f"⚠️ Gagal mengkonversi nilai split: {e}")
                aug_config['split'] = 'train'
        else:
            logger.debug("ℹ️ Komponen split_dropdown tidak ditemukan, menggunakan nilai default")
            aug_config['split'] = 'train'
            
        # Nilai tetap untuk service
        aug_config['process_bboxes'] = True
        aug_config['validate_results'] = True
        aug_config['resume'] = False
        
        # Pastikan data path tersedia
        if 'data' not in config:
            config['data'] = {}
        if 'dataset_path' not in config['data']:
            config['data']['dataset_path'] = ui_components.get('data_dir', 'data/preprocessed')
        
        # Validasi konfigurasi
        try:
            from smartcash.ui.dataset.augmentation.handlers.config_validator import validate_augmentation_config
            config = validate_augmentation_config(config)
        except Exception as e:
            logger.warning(f"⚠️ Gagal validasi konfigurasi: {e}")
        
        # Log konfigurasi yang dihasilkan
        logger.debug(f"ℹ️ Konfigurasi augmentasi setelah mapping: {config}")
        
    except Exception as e:
        logger.error(f"❌ Error saat memetakan UI ke config: {str(e)}")
        logger.debug(f"ℹ️ Detail error: {e}", exc_info=True)
    
    return config

def map_config_to_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Memetakan nilai konfigurasi augmentasi ke komponen UI.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi augmentasi
        
    Returns:
        Dictionary komponen UI yang sudah diupdate
    """
    try:
        # Log konfigurasi yang akan digunakan untuk memperbarui UI
        logger.debug(f"ℹ️ Konfigurasi yang akan digunakan untuk memperbarui UI: {config}")
        
        # Pastikan config valid
        if not config or not isinstance(config, dict) or 'augmentation' not in config:
            logger.warning("⚠️ Konfigurasi tidak valid, tidak dapat memperbarui UI")
            return ui_components
        
        aug_config = config['augmentation']
        logger.debug(f"ℹ️ Konfigurasi augmentasi: {aug_config}")
        
        # Dapatkan komponen UI
        aug_options = ui_components.get('aug_options')
        if not aug_options or not hasattr(aug_options, 'children') or len(aug_options.children) < 2:
            logger.warning("⚠️ Komponen aug_options tidak valid, tidak dapat memperbarui UI")
            return ui_components
        
        # Dapatkan tab container
        tab = aug_options.children[1]
        if not hasattr(tab, 'children') or len(tab.children) < 2:
            logger.warning("⚠️ Tab container tidak valid, tidak dapat memperbarui UI")
            return ui_components
        
        # Dapatkan tab dasar dan lanjutan
        basic_tab = tab.children[0]
        advanced_tab = tab.children[1]
        
        if not hasattr(basic_tab, 'children') or not hasattr(advanced_tab, 'children'):
            logger.warning("⚠️ Tab dasar atau lanjutan tidak valid, tidak dapat memperbarui UI")
            return ui_components
        
        # Log struktur UI sebelum pembaruan
        logger.debug(f"ℹ️ Tab dasar memiliki {len(basic_tab.children) if hasattr(basic_tab, 'children') else 0} children")
        logger.debug(f"ℹ️ Tab lanjutan memiliki {len(advanced_tab.children) if hasattr(advanced_tab, 'children') else 0} children")
        
        # Update nilai di tab dasar
        if len(basic_tab.children) >= 4:
            # Prefix (children[1]) - Prioritaskan output_prefix untuk service
            if hasattr(basic_tab.children[1], 'value'):
                prefix_value = aug_config.get('output_prefix', aug_config.get('prefix', 'aug_'))
                logger.debug(f"ℹ️ Memperbarui nilai prefix UI ke: {prefix_value}")
                basic_tab.children[1].value = prefix_value
            else:
                logger.warning("⚠️ Komponen prefix tidak memiliki atribut value")
            
            # Jumlah variasi (children[2]) - Prioritaskan num_variations untuk service
            if hasattr(basic_tab.children[2], 'value'):
                factor_value = aug_config.get('num_variations', aug_config.get('factor', 2))
                logger.debug(f"ℹ️ Memperbarui nilai factor UI ke: {factor_value}")
                basic_tab.children[2].value = factor_value
            else:
                logger.warning("⚠️ Komponen factor tidak memiliki atribut value")
        else:
            logger.warning(f"⚠️ Tab dasar tidak memiliki cukup children: {len(basic_tab.children) if hasattr(basic_tab, 'children') else 'tidak ada children'}")
        
        # Update nilai di tab lanjutan
        if len(advanced_tab.children) >= 4:
            # Balance kelas (children[0]) - Prioritaskan target_balance untuk service
            if hasattr(advanced_tab.children[0], 'value'):
                balance_value = aug_config.get('target_balance', aug_config.get('balance_classes', True))
                logger.debug(f"ℹ️ Memperbarui nilai balance UI ke: {balance_value}")
                advanced_tab.children[0].value = balance_value
            else:
                logger.warning("⚠️ Komponen balance tidak memiliki atribut value")
            
            # Target count (children[1])
            if hasattr(advanced_tab.children[1], 'value'):
                target_count = aug_config.get('target_count', 1000)
                logger.debug(f"ℹ️ Memperbarui nilai target_count UI ke: {target_count}")
                advanced_tab.children[1].value = target_count
            else:
                logger.warning("⚠️ Komponen target_count tidak memiliki atribut value")
            
            # Num workers (children[2])
            if hasattr(advanced_tab.children[2], 'value'):
                num_workers = aug_config.get('num_workers', 4)
                logger.debug(f"ℹ️ Memperbarui nilai num_workers UI ke: {num_workers}")
                advanced_tab.children[2].value = num_workers
            else:
                logger.warning("⚠️ Komponen num_workers tidak memiliki atribut value")
            
            # Move to preprocessed (children[3])
            if hasattr(advanced_tab.children[3], 'value'):
                move_to_preprocessed = aug_config.get('move_to_preprocessed', True)
                logger.debug(f"ℹ️ Memperbarui nilai move_to_preprocessed UI ke: {move_to_preprocessed}")
                advanced_tab.children[3].value = move_to_preprocessed
            else:
                logger.warning("⚠️ Komponen move_to_preprocessed tidak memiliki atribut value")
        else:
            logger.warning(f"⚠️ Tab lanjutan tidak memiliki cukup children: {len(advanced_tab.children) if hasattr(advanced_tab, 'children') else 'tidak ada children'}")
        
        # Update jenis augmentasi select multiple (jika tersedia)
        aug_types_select = ui_components.get('aug_types_select')
        if aug_types_select and hasattr(aug_types_select, 'value'):
            try:
                aug_types = aug_config.get('types', ['combined'])
                # Pastikan aug_types adalah list
                if not isinstance(aug_types, list):
                    aug_types = ['combined']
                    
                # Pastikan semua nilai dalam aug_types valid (ada dalam opsi)
                valid_options = [opt[1] for opt in aug_types_select.options]
                valid_aug_types = [t for t in aug_types if t in valid_options]
                
                # Jika tidak ada jenis yang valid, gunakan default
                if not valid_aug_types:
                    valid_aug_types = ['combined']
                    
                logger.debug(f"ℹ️ Memperbarui nilai aug_types UI ke: {valid_aug_types}")
                aug_types_select.value = tuple(valid_aug_types)  # SelectMultiple menggunakan tuple
            except Exception as e:
                logger.warning(f"⚠️ Gagal memperbarui nilai aug_types: {e}")
                logger.debug(f"Detail error: {e}", exc_info=True)
        else:
            logger.debug("ℹ️ Komponen aug_types_select tidak ditemukan")
            
        # Update target split dropdown (jika tersedia)
        split_dropdown = ui_components.get('split_dropdown')
        if split_dropdown and hasattr(split_dropdown, 'value'):
            try:
                split = aug_config.get('split', 'train')
                logger.debug(f"ℹ️ Memperbarui nilai split UI ke: {split}")
                split_dropdown.value = split
            except Exception as e:
                logger.warning(f"⚠️ Gagal memperbarui nilai split: {e}")
        else:
            logger.debug("ℹ️ Komponen split_dropdown tidak ditemukan")
        
        # Pastikan data path tersedia
        if 'data' not in config:
            config['data'] = {}
        if 'dataset_path' not in config['data']:
            config['data']['dataset_path'] = ui_components.get('data_dir', 'data/preprocessed')
        
        # Simpan referensi config ke ui_components
        ui_components['config'] = config
        
        # Log konfigurasi yang dihasilkan
        logger.debug(f"ℹ️ UI berhasil diperbarui dari konfigurasi: {config}")
        
    except Exception as e:
        logger.error(f"❌ Error saat memetakan config ke UI: {str(e)}")
        logger.debug(f"ℹ️ Detail error: {e}", exc_info=True)
    
    return ui_components

def extract_augmentation_params(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ekstrak parameter augmentasi dari komponen UI untuk digunakan dalam proses augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary parameter augmentasi
    """
    # Dapatkan konfigurasi
    config = ui_components.get('config', {})
    aug_config = config.get('augmentation', {})
    
    # Parameter default
    params = {
        'split': 'train',
        'augmentation_types': ['combined'],
        'num_variations': 2,
        'output_prefix': 'aug_',
        'validate_results': True,
        'process_bboxes': True,
        'target_balance': True,
        'num_workers': 4,
        'move_to_preprocessed': True,
        'target_count': 1000
    }
    
    # Update dari konfigurasi
    if aug_config:
        # Prioritaskan parameter yang digunakan oleh service
        params['num_variations'] = aug_config.get('num_variations', aug_config.get('factor', params['num_variations']))
        params['output_prefix'] = aug_config.get('output_prefix', aug_config.get('prefix', params['output_prefix']))
        params['validate_results'] = aug_config.get('validate_results', params['validate_results'])
        params['process_bboxes'] = aug_config.get('process_bboxes', params['process_bboxes'])
        params['target_balance'] = aug_config.get('target_balance', aug_config.get('balance_classes', params['target_balance']))
        params['num_workers'] = aug_config.get('num_workers', params['num_workers'])
        params['move_to_preprocessed'] = aug_config.get('move_to_preprocessed', params['move_to_preprocessed'])
        params['target_count'] = aug_config.get('target_count', params['target_count'])
    
    # Log parameter yang dihasilkan
    logger.debug(f"ℹ️ Parameter augmentasi yang diekstrak: {params}")
    
    return params
