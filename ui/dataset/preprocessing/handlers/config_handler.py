"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Fixed config handler yang mempertahankan struktur preprocessing_config.yaml dengan inheritance dan merge yang tepat
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
from smartcash.ui.dataset.preprocessing.handlers.config_updater import update_preprocessing_ui
from smartcash.common.config.manager import get_config_manager

class PreprocessingConfigHandler(ConfigHandler):
    """Config handler untuk preprocessing dengan struktur yang konsisten dengan preprocessing_config.yaml"""
    
    def __init__(self, module_name: str = 'preprocessing', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = 'preprocessing_config.yaml'
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari preprocessing UI components"""
        return extract_preprocessing_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config dengan struktur lengkap"""
        update_preprocessing_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config sesuai preprocessing_config.yaml structure"""
        from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
        return get_default_preprocessing_config()
    
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """Load config dengan inheritance dari base_config.yaml"""
        try:
            filename = config_filename or self.config_filename
            
            # Load preprocessing_config.yaml
            preprocessing_config = self.config_manager.load_config(filename) or {}
            
            # Load base_config.yaml jika ada inheritance
            base_inheritance = preprocessing_config.get('_base_')
            if base_inheritance:
                base_config = self.config_manager.load_config(base_inheritance) or {}
                # Merge base dengan preprocessing override
                merged_config = self._merge_with_inheritance(base_config, preprocessing_config)
            else:
                merged_config = preprocessing_config
            
            # Fallback ke default jika kosong
            if not merged_config:
                self.logger.warning("⚠️ Config kosong, menggunakan default")
                merged_config = self.get_default_config()
            
            return merged_config
            
        except Exception as e:
            self.logger.error(f"❌ Error loading config: {str(e)}")
            return self.get_default_config()
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Save config dengan mempertahankan struktur preprocessing_config.yaml"""
        try:
            filename = config_filename or self.config_filename
            
            # Extract UI values
            ui_config = self.extract_config(ui_components)
            
            # Load existing preprocessing_config untuk preserve non-UI fields
            existing_config = self.config_manager.load_config(filename) or {}
            
            # Create final config dengan struktur preprocessing_config.yaml
            final_config = self._create_preprocessing_config_structure(ui_config, existing_config)
            
            # Validate config structure
            validation = self.validate_config(final_config)
            if not validation['valid']:
                self.logger.error(f"❌ Config tidak valid: {'; '.join(validation['errors'])}")
                return False
            
            # Save dengan struktur yang tepat
            success = self.config_manager.save_config(final_config, filename)
            
            if success:
                self.logger.success(f"✅ Config preprocessing tersimpan ke {filename}")
                return True
            else:
                self.logger.error(f"❌ Gagal menyimpan config ke {filename}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error saving preprocessing config: {str(e)}")
            return False
    
    def _merge_with_inheritance(self, base_config: Dict[str, Any], preprocessing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge base_config dengan preprocessing_config override"""
        import copy
        
        # Start dengan base config
        merged = copy.deepcopy(base_config)
        
        # Override dengan preprocessing_config values (kecuali _base_)
        for key, value in preprocessing_config.items():
            if key == '_base_':
                continue  # Skip inheritance marker
            
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Deep merge untuk nested dictionaries
                merged[key] = self._deep_merge_dicts(merged[key], value)
            else:
                # Direct override untuk non-dict values
                merged[key] = value
        
        return merged
    
    def _deep_merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        import copy
        
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _create_preprocessing_config_structure(self, ui_config: Dict[str, Any], existing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create config dengan struktur sesuai preprocessing_config.yaml"""
        
        # Preserve inheritance marker
        base_inheritance = existing_config.get('_base_', 'base_config.yaml')
        
        # Extract sections dari UI config
        preprocessing_ui = ui_config.get('preprocessing', {})
        performance_ui = ui_config.get('performance', {})
        cleanup_ui = ui_config.get('cleanup', {})
        
        # Build final config sesuai preprocessing_config.yaml structure
        final_config = {
            # Inheritance marker
            '_base_': base_inheritance,
            
            # Override preprocessing section dengan UI values
            'preprocessing': {
                # Basic settings dari existing atau default
                'output_dir': existing_config.get('preprocessing', {}).get('output_dir', 'data/preprocessed'),
                'save_visualizations': existing_config.get('preprocessing', {}).get('save_visualizations', False),
                'vis_dir': existing_config.get('preprocessing', {}).get('vis_dir', 'visualizations/preprocessing'),
                'sample_size': existing_config.get('preprocessing', {}).get('sample_size', 0),
                
                # UI controlled settings
                'target_split': preprocessing_ui.get('target_split', 'all'),
                'force_reprocess': preprocessing_ui.get('force_reprocess', False),
                
                # Complete normalization structure dari UI
                'normalization': {
                    'enabled': preprocessing_ui.get('normalization', {}).get('enabled', True),
                    'method': preprocessing_ui.get('normalization', {}).get('method', 'minmax'),
                    'target_size': preprocessing_ui.get('normalization', {}).get('target_size', [640, 640]),
                    'preserve_aspect_ratio': preprocessing_ui.get('normalization', {}).get('preserve_aspect_ratio', True),
                    'normalize_pixel_values': preprocessing_ui.get('normalization', {}).get('normalize_pixel_values', True),
                    'pixel_range': preprocessing_ui.get('normalization', {}).get('pixel_range', [0, 1])
                },
                
                # Preserve complex structures dari existing config
                'validate': existing_config.get('preprocessing', {}).get('validate', {
                    'enabled': True,
                    'fix_issues': True,
                    'move_invalid': True,
                    'visualize': False,
                    'check_image_quality': True,
                    'check_labels': True,
                    'check_coordinates': True,
                    'check_uuid_consistency': True
                }),
                'analysis': existing_config.get('preprocessing', {}).get('analysis', {
                    'enabled': False,
                    'class_balance': True,
                    'image_size_distribution': True,
                    'bbox_statistics': True,
                    'layer_balance': True
                }),
                'balance': existing_config.get('preprocessing', {}).get('balance', {
                    'enabled': False,
                    'target_distribution': 'auto',
                    'methods': {
                        'undersampling': False,
                        'oversampling': True,
                        'augmentation': True
                    },
                    'min_samples_per_class': 100,
                    'max_samples_per_class': 1000
                })
            },
            
            # Override performance section dengan UI values
            'performance': {
                'num_workers': performance_ui.get('num_workers', 8),
                'batch_size': existing_config.get('performance', {}).get('batch_size', 32),
                'use_gpu': existing_config.get('performance', {}).get('use_gpu', True),
                'compression_level': existing_config.get('performance', {}).get('compression_level', 90),
                'max_memory_usage_gb': existing_config.get('performance', {}).get('max_memory_usage_gb', 4.0),
                'use_mixed_precision': existing_config.get('performance', {}).get('use_mixed_precision', True)
            },
            
            # Preserve cleanup section dari existing atau default
            'cleanup': existing_config.get('cleanup', {
                'augmentation_patterns': [
                    'aug_.*',
                    '.*_augmented.*',
                    '.*_modified.*',
                    '.*_processed.*',
                    '.*_norm.*'
                ],
                'ignored_patterns': [
                    '.*\.gitkeep',
                    '.*\.DS_Store',
                    '.*\.gitignore'
                ],
                'backup_dir': 'data/backup/preprocessing',
                'backup_enabled': False,
                'auto_cleanup_preprocessed': False
            })
        }
        
        # Preserve metadata
        final_config['config_version'] = ui_config.get('config_version', '1.0')
        final_config['updated_at'] = ui_config.get('updated_at')
        
        return final_config
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced validation untuk struktur preprocessing_config.yaml"""
        errors = []
        warnings = []
        
        # Validate inheritance
        if '_base_' not in config:
            warnings.append("Inheritance marker '_base_' tidak ditemukan")
        
        # Validate preprocessing section
        preprocessing = config.get('preprocessing', {})
        if not preprocessing:
            errors.append("Section 'preprocessing' wajib ada")
        else:
            # Validate normalization
            normalization = preprocessing.get('normalization', {})
            if not normalization:
                errors.append("Normalization config wajib ada")
            else:
                method = normalization.get('method', 'minmax')
                if method not in ['minmax', 'standard', 'none']:
                    errors.append("Metode normalisasi tidak valid")
                
                target_size = normalization.get('target_size', [640, 640])
                if not isinstance(target_size, list) or len(target_size) != 2:
                    errors.append("Target size harus berupa list [width, height]")
            
            # Validate target_split
            target_split = preprocessing.get('target_split', 'all')
            if target_split not in ['all', 'train', 'valid', 'test']:
                warnings.append("Target split tidak standar")
        
        # Validate performance section
        performance = config.get('performance', {})
        if performance:
            num_workers = performance.get('num_workers', 8)
            if not isinstance(num_workers, int) or num_workers < 1 or num_workers > 16:
                warnings.append("Number of workers sebaiknya antara 1-16")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }