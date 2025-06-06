"""
File: smartcash/ui/dataset/preprocessing/handlers/preprocessing_handlers.py
Deskripsi: Unified handlers yang terintegrasi dengan dataset/preprocessors untuk eliminasi duplikasi
"""

from typing import Dict, Any
from smartcash.dataset.preprocessor.core.preprocessing_manager import PreprocessingManager
from smartcash.dataset.preprocessor.operations.dataset_checker import DatasetChecker
from smartcash.dataset.preprocessor.operations.cleanup_executor import CleanupExecutor
from smartcash.ui.components.confirmation_dialog import create_destructive_confirmation
from smartcash.ui.utils.button_state_manager import get_button_state_manager
from smartcash.common.config.manager import get_config_manager

def setup_preprocessing_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup unified handlers dengan integrasi preprocessors"""
    
    # Setup progress callback dengan API yang benar
    def create_progress_callback():
        def progress_callback(**kwargs):
            progress = kwargs.get('progress', 0)
            message = kwargs.get('message', 'Processing...')
            level = kwargs.get('type', 'level1')
            
            # Menggunakan progress_tracker object jika tersedia
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker:
                # Mapping level untuk API yang benar
                if level == 'overall' or level == 'level1':
                    # Gunakan update_overall untuk level1/overall
                    progress_tracker.update_overall(progress, message, kwargs.get('color', None))
                elif level == 'step' or level == 'level2':
                    # Gunakan update_current untuk level2/step
                    progress_tracker.update_current(progress, message, kwargs.get('color', None))
                else:
                    # Fallback ke update untuk kompatibilitas
                    progress_tracker.update(level, progress, message, kwargs.get('color', None))
            else:
                # Fallback untuk kompatibilitas dengan implementasi lama
                update_fn = ui_components.get('update_progress')
                if update_fn:
                    update_fn('overall' if level in ['overall', 'level1'] else 'step', progress, message)
        return progress_callback
    
    ui_components['progress_callback'] = create_progress_callback()
    
    # Setup handlers
    setup_preprocessing_handler(ui_components, config)
    setup_check_handler(ui_components, config)
    setup_cleanup_handler(ui_components, config)
    setup_config_handlers(ui_components, config)
    
    return ui_components

def setup_preprocessing_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup preprocessing handler dengan integrated state management"""
    
    def execute_preprocessing(button=None):
        button_manager = _get_button_manager(ui_components)
        logger = ui_components.get('logger')
        
        _clear_outputs(ui_components)
        button_manager.disable_buttons('preprocess_button')
        
        try:
            # Validate dataset ready
            valid, msg = _validate_dataset_ready(config, logger)
            if not valid:
                _update_status_panel(ui_components, f"❌ {msg}", "error")
                return
            
            logger and logger.info("🚀 Memulai preprocessing dataset")
            
            # Show progress tracker dengan API yang benar
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker:
                # Gunakan metode show dengan parameter yang benar
                preprocessing_steps = ["prepare", "process", "verify"]
                step_weights = {"prepare": 20, "process": 60, "verify": 20}
                progress_tracker.show("Preprocessing Dataset", preprocessing_steps, step_weights)
            else:
                # Fallback ke metode lama
                ui_components.get('show_for_operation', lambda x: None)('preprocessing')
            
            # Extract config dan execute
            params = _extract_processing_params(ui_components)
            processing_config = {**config, 'preprocessing': {**config.get('preprocessing', {}), **params}}
            
            manager = PreprocessingManager(processing_config, logger)
            manager.register_progress_callback(ui_components['progress_callback'])
            
            result = manager.preprocess_with_uuid_consistency(
                split=params.get('split', 'all'),
                force_reprocess=params.get('force_reprocess', False)
            )
            
            if result['success']:
                total = result.get('total_images', 0)
                time_taken = result.get('processing_time', 0)
                # Complete operation dengan API yang benar
                progress_tracker = ui_components.get('progress_tracker')
                if progress_tracker and hasattr(progress_tracker, 'complete'):
                    progress_tracker.complete(
                        f"Preprocessing selesai: {total:,} gambar dalam {time_taken:.1f}s"
                    )
                else:
                    # Fallback ke metode lama
                    ui_components.get('complete_operation', lambda x: None)(
                        f"Preprocessing selesai: {total:,} gambar dalam {time_taken:.1f}s"
                    )
                _update_status_panel(ui_components, f"✅ Preprocessing berhasil: {total:,} gambar", "success")
            else:
                raise Exception(result['message'])
                
        except Exception as e:
            error_msg = f"Preprocessing gagal: {str(e)}"
            logger and logger.error(f"💥 {error_msg}")
            # Error operation dengan API yang benar
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker and hasattr(progress_tracker, 'error'):
                progress_tracker.error(error_msg)
            else:
                # Fallback ke metode lama
                ui_components.get('error_operation', lambda x: None)(error_msg)
            _update_status_panel(ui_components, error_msg, "error")
        
        finally:
            button_manager.enable_buttons()
    
    ui_components['preprocess_button'].on_click(execute_preprocessing)

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup dataset checker dengan direct file scanning"""
    
    def execute_check(button=None):
        button_manager = _get_button_manager(ui_components)
        logger = ui_components.get('logger')
        
        _clear_outputs(ui_components)
        button_manager.disable_buttons('check_button')
        
        try:
            logger and logger.info("🔍 Checking dataset")
            ui_components.get('show_for_operation', lambda x: None)('check')
            
            # Check source dataset
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker:
                progress_tracker.update('level1', 30, "🔍 Checking source dataset")
            else:
                ui_components.get('update_progress', lambda *a: None)('overall', 30, "Checking source dataset")
            
            source_valid, source_msg = _validate_dataset_ready(config, logger)
            
            # Check preprocessed dengan direct scanning
            if progress_tracker:
                progress_tracker.update('level1', 70, "📁 Checking preprocessed dataset")
            else:
                ui_components.get('update_progress', lambda *a: None)('overall', 70, "Checking preprocessed dataset")
            preprocessed_exists, preprocessed_count = _check_preprocessed_exists(config)
            
            # Display results
            if source_valid:
                logger and logger.success(f"✅ {source_msg}")
                _update_status_panel(ui_components, f"Dataset siap: {source_msg.split(': ')[1]}", "success")
            else:
                logger and logger.error(f"❌ {source_msg}")
                _update_status_panel(ui_components, f"❌ {source_msg}", "error")
                return
            
            if preprocessed_exists:
                logger and logger.success(f"💾 Preprocessed dataset: {preprocessed_count:,} gambar")
                
                # Show detailed breakdown
                _show_preprocessed_breakdown(ui_components, config, logger)
            else:
                logger and logger.info("ℹ️ Belum ada preprocessed dataset")
            
            # Complete operation dengan API yang benar
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker and hasattr(progress_tracker, 'complete'):
                progress_tracker.complete("Dataset check selesai")
            else:
                # Fallback ke metode lama
                ui_components.get('complete_operation', lambda x: None)("Dataset check selesai")
            
        except Exception as e:
            error_msg = f"Check gagal: {str(e)}"
            logger and logger.error(f"💥 {error_msg}")
            ui_components.get('error_operation', lambda x: None)(error_msg)
        
        finally:
            button_manager.enable_buttons()
    
    ui_components['check_button'].on_click(execute_check)

def _show_preprocessed_breakdown(ui_components: Dict[str, Any], config: Dict[str, Any], logger):
    """Show detailed preprocessed breakdown"""
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
    
    # Display detailed report
    if 'log_output' in ui_components and breakdown:
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

def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup cleanup handler dengan integrated confirmation"""
    
    def execute_cleanup(button=None):
        _clear_outputs(ui_components)
        
        # Check existing preprocessed data
        has_data, count = _check_preprocessed_exists(config)
        if not has_data:
            _update_status_panel(ui_components, "ℹ️ Tidak ada data preprocessed untuk dibersihkan", "info")
            return
        
        def confirmed_cleanup():
            button_manager = _get_button_manager(ui_components)
            logger = ui_components.get('logger')
            
            button_manager.disable_buttons('cleanup_button')
            
            try:
                logger and logger.info("🧹 Cleanup preprocessed data")
                ui_components.get('show_for_operation', lambda x: None)('cleanup')
                
                executor = CleanupExecutor(config, logger)
                executor.register_progress_callback(ui_components['progress_callback'])
                
                result = executor.cleanup_preprocessed_data(safe_mode=True)
                
                if result['success']:
                    stats = result.get('stats', {})
                    files_removed = stats.get('files_removed', 0)
                    # Complete operation dengan API yang benar
                    progress_tracker = ui_components.get('progress_tracker')
                    if progress_tracker and hasattr(progress_tracker, 'complete'):
                        progress_tracker.complete(
                        f"Cleanup selesai: {files_removed:,} file dihapus"
                    )
                    _update_status_panel(ui_components, f"🧹 Cleanup berhasil: {files_removed:,} file", "success")
                else:
                    raise Exception(result['message'])
                    
            except Exception as e:
                error_msg = f"Cleanup gagal: {str(e)}"
                logger and logger.error(f"💥 {error_msg}")
                # Error operation dengan API yang benar
                progress_tracker = ui_components.get('progress_tracker')
                if progress_tracker and hasattr(progress_tracker, 'error'):
                    progress_tracker.error(error_msg)
                else:
                    # Fallback ke metode lama
                    ui_components.get('error_operation', lambda x: None)(error_msg)
            
            finally:
                button_manager.enable_buttons()
        
        # Show confirmation dengan informasi detail
        if 'confirmation_area' in ui_components:
            from IPython.display import display, clear_output
            with ui_components['confirmation_area']:
                clear_output(wait=True)
                dialog = create_destructive_confirmation(
                    title="⚠️ Konfirmasi Cleanup Dataset",
                    message=f"Operasi ini akan menghapus {count:,} file preprocessed.\n\nData asli tetap aman. Lanjutkan?",
                    on_confirm=lambda b: (confirmed_cleanup(), _clear_outputs(ui_components)),
                    on_cancel=lambda b: _clear_outputs(ui_components),
                    item_name="data preprocessed"
                )
                display(dialog)
    
    ui_components['cleanup_button'].on_click(execute_cleanup)

def setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup config save/reset handlers"""
    config_manager = get_config_manager()
    
    def save_config(button=None):
        try:
            _clear_outputs(ui_components)
            params = _extract_processing_params(ui_components)
            save_success = config_manager.save_config({'preprocessing': params}, 'preprocessing_config')
            status = "✅ Konfigurasi tersimpan" if save_success else "❌ Gagal simpan konfigurasi"
            _update_status_panel(ui_components, status, "success" if save_success else "error")
        except Exception as e:
            _update_status_panel(ui_components, f"❌ Error: {str(e)}", "error")
    
    def reset_config(button=None):
        try:
            _clear_outputs(ui_components)
            _apply_default_config(ui_components)
            _update_status_panel(ui_components, "🔄 Konfigurasi direset ke default", "info")
        except Exception as e:
            _update_status_panel(ui_components, f"❌ Error: {str(e)}", "error")
    
    ui_components['save_button'].on_click(save_config)
    ui_components['reset_button'].on_click(reset_config)

# Integrated validation helpers (replacing validation_helper.py)
def _validate_dataset_ready(config: Dict[str, Any], logger) -> tuple[bool, str]:
    """Integrated dataset validation"""
    from smartcash.dataset.utils.path_validator import get_path_validator
    
    data_dir = config.get('data', {}).get('dir', 'data')
    validator = get_path_validator(logger)
    result = validator.validate_dataset_structure(data_dir)
    
    if not result['valid'] or result['total_images'] == 0:
        return False, f"Dataset tidak valid: {result.get('issues', ['Unknown error'])[0] if result.get('issues') else 'No images found'}"
    
    return True, f"Dataset siap: {result['total_images']:,} gambar"

def _check_preprocessed_exists(config: Dict[str, Any]) -> tuple[bool, int]:
    """Check existing preprocessed data dengan direct file scanning"""
    from pathlib import Path
    
    preprocessed_dir = Path(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed'))
    
    if not preprocessed_dir.exists():
        return False, 0
    
    total_files = 0
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Scan splits: train, valid, test
    for split in ['train', 'valid', 'test']:
        split_images_dir = preprocessed_dir / split / 'images'
        if split_images_dir.exists():
            split_files = [f for f in split_images_dir.glob('*.*') if f.suffix.lower() in image_extensions]
            total_files += len(split_files)
    
    return total_files > 0, total_files

# Integrated UI state management (replacing ui_state_manager.py)
class SimpleButtonManager:
    """Simplified button state management"""
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.disabled_buttons = []
    
    def disable_buttons(self, exclude_button: str = None):
        """Disable buttons except exclude_button"""
        buttons = ['preprocess_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
        for btn_key in buttons:
            if btn_key != exclude_button and btn_key in self.ui_components:
                btn = self.ui_components[btn_key]
                if btn and not btn.disabled:
                    btn.disabled = True
                    self.disabled_buttons.append(btn_key)
    
    def enable_buttons(self):
        """Re-enable previously disabled buttons"""
        for btn_key in self.disabled_buttons:
            if btn_key in self.ui_components:
                btn = self.ui_components[btn_key]
                if btn:
                    btn.disabled = False
        self.disabled_buttons.clear()

def _get_button_manager(ui_components: Dict[str, Any]) -> SimpleButtonManager:
    """Get button manager instance"""
    if 'simple_button_manager' not in ui_components:
        ui_components['simple_button_manager'] = SimpleButtonManager(ui_components)
    return ui_components['simple_button_manager']

# Helper functions
def _extract_processing_params(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameters dari UI components"""
    resolution = getattr(ui_components.get('resolution_dropdown'), 'value', '640x640')
    width, height = map(int, resolution.split('x'))
    
    return {
        'img_size': [width, height],
        'normalize': getattr(ui_components.get('normalization_dropdown'), 'value', 'minmax') != 'none',
        'num_workers': getattr(ui_components.get('worker_slider'), 'value', 4),
        'split': getattr(ui_components.get('split_dropdown'), 'value', 'all'),
        'force_reprocess': False
    }

def _apply_default_config(ui_components: Dict[str, Any]):
    """Apply default config ke UI"""
    if 'resolution_dropdown' in ui_components:
        ui_components['resolution_dropdown'].value = '640x640'
    if 'normalization_dropdown' in ui_components:
        ui_components['normalization_dropdown'].value = 'minmax'
    if 'worker_slider' in ui_components:
        ui_components['worker_slider'].value = 4
    if 'split_dropdown' in ui_components:
        ui_components['split_dropdown'].value = 'all'

def _display_check_results(ui_components: Dict[str, Any], source_result: Dict[str, Any], 
                          preprocessed_result: Dict[str, Any], logger):
    """Display check results - REMOVED (replaced by direct scanning)"""
    pass  # Functionality moved to setup_check_handler

def _clear_outputs(ui_components: Dict[str, Any]):
    """Clear UI outputs"""
    for key in ['log_output', 'status', 'confirmation_area']:
        widget = ui_components.get(key)
        if widget and hasattr(widget, 'clear_output'):
            widget.clear_output(wait=True)

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info"):
    """Update status panel"""
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components:
        update_status_panel(ui_components['status_panel'], message, status_type)