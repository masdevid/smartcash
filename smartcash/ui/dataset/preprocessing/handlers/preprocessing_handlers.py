"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Fixed unified handlers dengan safe operations dan proper progress tracking
"""

from typing import Dict, Any
from smartcash.dataset.preprocessor.core.preprocessing_manager import PreprocessingManager
from smartcash.dataset.preprocessor.operations.dataset_checker import DatasetChecker
from smartcash.dataset.preprocessor.operations.cleanup_executor import CleanupExecutor
from smartcash.ui.components.confirmation_dialog import create_destructive_confirmation
from smartcash.common.config.manager import get_config_manager
from smartcash.common.utils.one_liner_fixes import safe_operation_or_none, safe_widget_operation

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup unified handlers dengan safe progress callbacks"""
    
    # Setup safe progress callback
    def create_progress_callback():
        def progress_callback(**kwargs):
            progress = kwargs.get('progress', 0)
            message = kwargs.get('message', 'Processing...')
            level = kwargs.get('type', 'level1')
            
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker:
                # Safe API calls dengan fallback
                if level in ['overall', 'level1']:
                    safe_operation_or_none(lambda: progress_tracker.update_overall(progress, message, kwargs.get('color', None)))
                elif level in ['step', 'level2']:
                    safe_operation_or_none(lambda: progress_tracker.update_current(progress, message, kwargs.get('color', None)))
                else:
                    safe_operation_or_none(lambda: progress_tracker.update(level, progress, message, kwargs.get('color', None)))
            else:
                # Fallback untuk compatibility
                update_fn = ui_components.get('update_progress')
                safe_operation_or_none(lambda: update_fn('overall' if level in ['overall', 'level1'] else 'step', progress, message) if update_fn else None)
        
        return progress_callback
    
    ui_components['progress_callback'] = create_progress_callback()
    
    # Setup handlers dengan safe operations
    safe_operation_or_none(lambda: setup_preprocessing_handler(ui_components, config))
    safe_operation_or_none(lambda: setup_check_handler(ui_components, config))
    safe_operation_or_none(lambda: setup_cleanup_handler(ui_components, config))
    safe_operation_or_none(lambda: setup_config_handlers(ui_components, config))
    
    return ui_components

def setup_preprocessing_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup preprocessing handler dengan safe state management"""
    
    def execute_preprocessing(button=None):
        button_manager = _get_button_manager(ui_components)
        logger = ui_components.get('logger')
        
        _clear_outputs(ui_components)
        safe_operation_or_none(lambda: button_manager.disable_buttons('preprocess_button'))
        
        try:
            # Safe dataset validation
            valid, msg = _validate_dataset_ready(config, logger)
            if not valid:
                _update_status_panel(ui_components, f"❌ {msg}", "error")
                return
            
            logger and logger.info("🚀 Memulai preprocessing dataset")
            
            # Safe progress tracker show
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker:
                preprocessing_steps = ["prepare", "process", "verify"]
                step_weights = {"prepare": 20, "process": 60, "verify": 20}
                safe_operation_or_none(lambda: progress_tracker.show("Preprocessing Dataset", preprocessing_steps, step_weights))
            else:
                safe_operation_or_none(lambda: ui_components.get('show_for_operation', lambda x: None)('preprocessing'))
            
            # Extract config dan execute preprocessing
            params = _extract_processing_params(ui_components)
            processing_config = {**config, 'preprocessing': {**config.get('preprocessing', {}), **params}}
            
            manager = safe_operation_or_none(lambda: PreprocessingManager(processing_config, logger))
            if not manager:
                raise Exception("Failed to create preprocessing manager")
            
            safe_operation_or_none(lambda: manager.register_progress_callback(ui_components['progress_callback']))
            
            result = safe_operation_or_none(lambda: manager.preprocess_with_uuid_consistency(
                split=params.get('split', 'all'),
                force_reprocess=params.get('force_reprocess', False)
            )) or {'success': False, 'message': 'Preprocessing operation failed'}
            
            if result.get('success', False):
                total = result.get('total_images', 0)
                time_taken = result.get('processing_time', 0)
                
                # Safe progress completion
                progress_tracker = ui_components.get('progress_tracker')
                if progress_tracker and hasattr(progress_tracker, 'complete'):
                    safe_operation_or_none(lambda: progress_tracker.complete(
                        f"Preprocessing selesai: {total:,} gambar dalam {time_taken:.1f}s"
                    ))
                else:
                    safe_operation_or_none(lambda: ui_components.get('complete_operation', lambda x: None)(
                        f"Preprocessing selesai: {total:,} gambar dalam {time_taken:.1f}s"
                    ))
                
                _update_status_panel(ui_components, f"✅ Preprocessing berhasil: {total:,} gambar", "success")
            else:
                raise Exception(result.get('message', 'Unknown preprocessing error'))
                
        except Exception as e:
            error_msg = f"Preprocessing gagal: {str(e)}"
            logger and logger.error(f"💥 {error_msg}")
            
            # Safe error handling
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker and hasattr(progress_tracker, 'error'):
                safe_operation_or_none(lambda: progress_tracker.error(error_msg))
            else:
                safe_operation_or_none(lambda: ui_components.get('error_operation', lambda x: None)(error_msg))
            
            _update_status_panel(ui_components, error_msg, "error")
        
        finally:
            safe_operation_or_none(lambda: button_manager.enable_buttons())
    
    # Safe button binding
    preprocess_button = ui_components.get('preprocess_button')
    safe_widget_operation(preprocess_button, 'on_click', execute_preprocessing)

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup dataset checker dengan safe file scanning"""
    
    def execute_check(button=None):
        button_manager = _get_button_manager(ui_components)
        logger = ui_components.get('logger')
        
        _clear_outputs(ui_components)
        safe_operation_or_none(lambda: button_manager.disable_buttons('check_button'))
        
        try:
            logger and logger.info("🔍 Checking dataset")
            safe_operation_or_none(lambda: ui_components.get('show_for_operation', lambda x: None)('check'))
            
            # Safe source dataset check
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker:
                safe_operation_or_none(lambda: progress_tracker.update('level1', 30, "🔍 Checking source dataset"))
            else:
                safe_operation_or_none(lambda: ui_components.get('update_progress', lambda *a: None)('overall', 30, "Checking source dataset"))
            
            source_valid, source_msg = _validate_dataset_ready(config, logger)
            
            # Safe preprocessed check
            if progress_tracker:
                safe_operation_or_none(lambda: progress_tracker.update('level1', 70, "📁 Checking preprocessed dataset"))
            else:
                safe_operation_or_none(lambda: ui_components.get('update_progress', lambda *a: None)('overall', 70, "Checking preprocessed dataset"))
            
            preprocessed_exists, preprocessed_count = _check_preprocessed_exists(config)
            
            # Display results safely
            if source_valid:
                logger and logger.success(f"✅ {source_msg}")
                msg_parts = source_msg.split(': ')
                display_msg = msg_parts[1] if len(msg_parts) > 1 else source_msg
                _update_status_panel(ui_components, f"Dataset siap: {display_msg}", "success")
            else:
                logger and logger.error(f"❌ {source_msg}")
                _update_status_panel(ui_components, f"❌ {source_msg}", "error")
                return
            
            if preprocessed_exists:
                logger and logger.success(f"💾 Preprocessed dataset: {preprocessed_count:,} gambar")
                safe_operation_or_none(lambda: _show_preprocessed_breakdown(ui_components, config, logger))
            else:
                logger and logger.info("ℹ️ Belum ada preprocessed dataset")
            
            # Safe progress completion
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker and hasattr(progress_tracker, 'complete'):
                safe_operation_or_none(lambda: progress_tracker.complete("Dataset check selesai"))
            else:
                safe_operation_or_none(lambda: ui_components.get('complete_operation', lambda x: None)("Dataset check selesai"))
            
        except Exception as e:
            error_msg = f"Check gagal: {str(e)}"
            logger and logger.error(f"💥 {error_msg}")
            safe_operation_or_none(lambda: ui_components.get('error_operation', lambda x: None)(error_msg))
        
        finally:
            safe_operation_or_none(lambda: button_manager.enable_buttons())
    
    # Safe button binding
    check_button = ui_components.get('check_button')
    safe_widget_operation(check_button, 'on_click', execute_check)

def _show_preprocessed_breakdown(ui_components: Dict[str, Any], config: Dict[str, Any], logger):
    """Show detailed preprocessed breakdown dengan safe operations"""
    def show_operation():
        from pathlib import Path
        from IPython.display import display, HTML
        
        preprocessed_dir = Path(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed'))
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        breakdown = {}
        total_images = 0
        
        for split in ['train', 'valid', 'test']:
            split_images_dir = preprocessed_dir / split / 'images'
            if split_images_dir.exists():
                split_files = [f for f in split_images_dir.glob('*.*') if f.suffix.lower() in image_extensions]
                count = len(split_files)
                if count > 0:
                    breakdown[split] = count
                    total_images += count
                    logger and logger.info(f"📂 {split}: {count:,} gambar preprocessed")
        
        # Safe display detailed report
        if ui_components.get('log_output') and breakdown:
            report_html = f"""
            <div style="background:#f0f8ff;padding:10px;border-radius:5px;margin:10px 0;">
                <strong>📊 Preprocessed Dataset Breakdown:</strong><br>
                <ul style="margin:8px 0;padding-left:20px;">
            """
            
            for split, count in breakdown.items():
                percentage = (count / total_images * 100) if total_images > 0 else 0
                report_html += f"<li><strong>{split}:</strong> {count:,} gambar ({percentage:.1f}%)</li>"
            
            report_html += f"""
                </ul>
                <div style="margin-top:8px;padding:6px;background:#e7f3ff;border-radius:3px;">
                    <strong>Total:</strong> {total_images:,} gambar preprocessed siap untuk training
                </div>
            </div>
            """
            
            with ui_components['log_output']:
                display(HTML(report_html))
    
    safe_operation_or_none(show_operation)

def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup cleanup handler dengan safe confirmation"""
    
    def execute_cleanup(button=None):
        _clear_outputs(ui_components)
        
        # Safe check existing preprocessed data
        has_data, count = _check_preprocessed_exists(config)
        if not has_data:
            _update_status_panel(ui_components, "ℹ️ Tidak ada data preprocessed untuk dibersihkan", "info")
            return
        
        def confirmed_cleanup():
            button_manager = _get_button_manager(ui_components)
            logger = ui_components.get('logger')
            
            safe_operation_or_none(lambda: button_manager.disable_buttons('cleanup_button'))
            
            try:
                logger and logger.info("🧹 Cleanup preprocessed data")
                safe_operation_or_none(lambda: ui_components.get('show_for_operation', lambda x: None)('cleanup'))
                
                executor = safe_operation_or_none(lambda: CleanupExecutor(config, logger))
                if not executor:
                    raise Exception("Failed to create cleanup executor")
                
                safe_operation_or_none(lambda: executor.register_progress_callback(ui_components['progress_callback']))
                
                result = safe_operation_or_none(lambda: executor.cleanup_preprocessed_data(safe_mode=True)) or {
                    'success': False, 'message': 'Cleanup operation failed'
                }
                
                if result.get('success', False):
                    stats = result.get('stats', {})
                    files_removed = stats.get('files_removed', 0)
                    
                    # Safe progress completion
                    progress_tracker = ui_components.get('progress_tracker')
                    if progress_tracker and hasattr(progress_tracker, 'complete'):
                        safe_operation_or_none(lambda: progress_tracker.complete(
                            f"Cleanup selesai: {files_removed:,} file dihapus"
                        ))
                    
                    _update_status_panel(ui_components, f"🧹 Cleanup berhasil: {files_removed:,} file", "success")
                else:
                    raise Exception(result.get('message', 'Unknown cleanup error'))
                    
            except Exception as e:
                error_msg = f"Cleanup gagal: {str(e)}"
                logger and logger.error(f"💥 {error_msg}")
                
                # Safe error handling
                progress_tracker = ui_components.get('progress_tracker')
                if progress_tracker and hasattr(progress_tracker, 'error'):
                    safe_operation_or_none(lambda: progress_tracker.error(error_msg))
                else:
                    safe_operation_or_none(lambda: ui_components.get('error_operation', lambda x: None)(error_msg))
            
            finally:
                safe_operation_or_none(lambda: button_manager.enable_buttons())
        
        # Safe confirmation dialog
        def show_confirmation():
            confirmation_area = ui_components.get('confirmation_area')
            if confirmation_area:
                from IPython.display import display, clear_output
                with confirmation_area:
                    clear_output(wait=True)
                    dialog = create_destructive_confirmation(
                        title="⚠️ Konfirmasi Cleanup Dataset",
                        message=f"Operasi ini akan menghapus {count:,} file preprocessed.\n\nData asli tetap aman. Lanjutkan?",
                        on_confirm=lambda b: (confirmed_cleanup(), _clear_outputs(ui_components)),
                        on_cancel=lambda b: _clear_outputs(ui_components),
                        item_name="data preprocessed"
                    )
                    display(dialog)
        
        safe_operation_or_none(show_confirmation)
    
    # Safe button binding
    cleanup_button = ui_components.get('cleanup_button')
    safe_widget_operation(cleanup_button, 'on_click', execute_cleanup)

def setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup config save/reset handlers dengan safe operations"""
    config_manager = get_config_manager()
    
    def save_config(button=None):
        def save_operation():
            _clear_outputs(ui_components)
            params = _extract_processing_params(ui_components)
            save_success = config_manager.save_config({'preprocessing': params}, 'preprocessing_config')
            status = "✅ Konfigurasi tersimpan" if save_success else "❌ Gagal simpan konfigurasi"
            _update_status_panel(ui_components, status, "success" if save_success else "error")
        
        safe_operation_or_none(save_operation)
    
    def reset_config(button=None):
        def reset_operation():
            _clear_outputs(ui_components)
            _apply_default_config(ui_components)
            _update_status_panel(ui_components, "🔄 Konfigurasi direset ke default", "info")
        
        safe_operation_or_none(reset_operation)
    
    # Safe button binding
    save_button = ui_components.get('save_button')
    reset_button = ui_components.get('reset_button')
    safe_widget_operation(save_button, 'on_click', save_config)
    safe_widget_operation(reset_button, 'on_click', reset_config)

# Safe validation helpers
def _validate_dataset_ready(config: Dict[str, Any], logger) -> tuple[bool, str]:
    """Safe dataset validation dengan error handling yang lebih baik"""
    def validate_operation():
        from smartcash.dataset.utils.path_validator import get_path_validator
        from pathlib import Path
        import os
        
        # Pastikan config dan data_dir valid
        if not config or not isinstance(config, dict):
            return False, "Konfigurasi tidak valid"
            
        data_dir = config.get('data', {}).get('dir', 'data')
        if not data_dir or not isinstance(data_dir, str):
            return False, "Path dataset tidak valid"
        
        # Validasi path dataset secara manual terlebih dahulu
        data_path = Path(data_dir)
        if not data_path.exists():
            return False, f"Directory dataset tidak ditemukan: {data_dir}"
            
        # Periksa struktur dasar dataset
        required_splits = ['train']
        missing_splits = []
        
        for split in required_splits:
            split_path = data_path / split
            if not split_path.exists():
                missing_splits.append(split)
                continue
                
            images_dir = split_path / 'images'
            labels_dir = split_path / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                missing_splits.append(f"{split} (images/labels)")
        
        if missing_splits:
            return False, f"Struktur dataset tidak lengkap: {', '.join(missing_splits)} tidak ditemukan"
        
        # Gunakan validator untuk analisis lebih detail
        try:
            validator = get_path_validator(logger)
            result = validator.validate_dataset_structure(data_dir)
            
            if not result.get('valid', False):
                issues = result.get('issues', ['Unknown error'])
                first_issue = issues[0] if issues else 'No images found'
                return False, f"Dataset tidak valid: {first_issue}"
            
            total_images = result.get('total_images', 0)
            if total_images == 0:
                return False, "Dataset tidak memiliki gambar"
                
            return True, f"Dataset siap: {total_images:,} gambar"
            
        except Exception as e:
            error_msg = str(e)
            logger and logger.error(f"❌ Error validasi dataset: {error_msg}")
            return False, f"Error validasi dataset: {error_msg[:100]}..."
    
    try:
        return safe_operation_or_none(validate_operation) or (False, "Error validating dataset")
    except Exception as e:
        logger and logger.error(f"❌ Unexpected error in dataset validation: {str(e)}")
        return False, f"Error validasi dataset: {str(e)[:100]}..."

def _check_preprocessed_exists(config: Dict[str, Any]) -> tuple[bool, int]:
    """Safe check existing preprocessed data"""
    def check_operation():
        from pathlib import Path
        
        preprocessed_dir = Path(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed'))
        
        if not preprocessed_dir.exists():
            return False, 0
        
        total_files = 0
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for split in ['train', 'valid', 'test']:
            split_images_dir = preprocessed_dir / split / 'images'
            if split_images_dir.exists():
                split_files = [f for f in split_images_dir.glob('*.*') if f.suffix.lower() in image_extensions]
                total_files += len(split_files)
        
        return total_files > 0, total_files
    
    return safe_operation_or_none(check_operation) or (False, 0)

# Safe UI state management
class SimpleButtonManager:
    """Safe button state management"""
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.disabled_buttons = []
    
    def disable_buttons(self, exclude_button: str = None):
        """Safe disable buttons except exclude_button"""
        buttons = ['preprocess_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
        for btn_key in buttons:
            if btn_key != exclude_button and btn_key in self.ui_components:
                btn = self.ui_components[btn_key]
                if btn and hasattr(btn, 'disabled') and not btn.disabled:
                    safe_widget_operation(btn, 'disabled', True)
                    self.disabled_buttons.append(btn_key)
    
    def enable_buttons(self):
        """Safe re-enable previously disabled buttons"""
        for btn_key in self.disabled_buttons:
            if btn_key in self.ui_components:
                btn = self.ui_components[btn_key]
                if btn and hasattr(btn, 'disabled'):
                    safe_widget_operation(btn, 'disabled', False)
        self.disabled_buttons.clear()

def _get_button_manager(ui_components: Dict[str, Any]) -> SimpleButtonManager:
    """Get safe button manager instance"""
    if 'simple_button_manager' not in ui_components:
        ui_components['simple_button_manager'] = SimpleButtonManager(ui_components)
    return ui_components['simple_button_manager']

# Safe helper functions
def _extract_processing_params(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Safe extract parameters dari UI components"""
    def extract_operation():
        resolution = getattr(ui_components.get('resolution_dropdown'), 'value', '640x640')
        width, height = map(int, resolution.split('x')) if 'x' in resolution else (640, 640)
        
        return {
            'img_size': [width, height],
            'normalize': getattr(ui_components.get('normalization_dropdown'), 'value', 'minmax') != 'none',
            'num_workers': getattr(ui_components.get('worker_slider'), 'value', 4),
            'split': getattr(ui_components.get('split_dropdown'), 'value', 'all'),
            'force_reprocess': False
        }
    
    return safe_operation_or_none(extract_operation) or {
        'img_size': [640, 640], 'normalize': True, 'num_workers': 4, 'split': 'all', 'force_reprocess': False
    }

def _apply_default_config(ui_components: Dict[str, Any]):
    """Safe apply default config ke UI"""
    def apply_operation():
        widgets = [
            ('resolution_dropdown', '640x640'),
            ('normalization_dropdown', 'minmax'),
            ('worker_slider', 4),
            ('split_dropdown', 'all')
        ]
        
        for widget_key, default_value in widgets:
            widget = ui_components.get(widget_key)
            if widget and hasattr(widget, 'value'):
                safe_widget_operation(widget, 'value', default_value)
    
    safe_operation_or_none(apply_operation)

def _clear_outputs(ui_components: Dict[str, Any]):
    """Safe clear UI outputs"""
    def clear_operation():
        for key in ['log_output', 'status', 'confirmation_area']:
            widget = ui_components.get(key)
            if widget and hasattr(widget, 'clear_output'):
                safe_widget_operation(widget, 'clear_output', wait=True)
    
    safe_operation_or_none(clear_operation)

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info"):
    """Safe update status panel"""
    def update_operation():
        from smartcash.ui.components.status_panel import update_status_panel
        status_panel = ui_components.get('status_panel')
        if status_panel:
            update_status_panel(status_panel, message, status_type)
    
    safe_operation_or_none(update_operation)