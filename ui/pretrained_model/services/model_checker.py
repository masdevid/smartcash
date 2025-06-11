"""
File: smartcash/ui/pretrained_model/services/model_checker.py
Deskripsi: Service untuk check existing models dengan progress tracker integration
"""

from pathlib import Path
from typing import Dict, Any, List
from smartcash.ui.pretrained_model.utils.model_utils import ModelUtils, SimpleProgressDelegate

class ModelChecker:
    """Service untuk check existing models dengan UI progress tracker integration"""
    
    def __init__(self, ui_components: Dict[str, Any], logger=None):
        self.ui_components, self.logger = ui_components, logger
        self.progress_delegate = SimpleProgressDelegate(ui_components)
        self.config = ModelUtils.get_models_from_ui_config(ui_components)
        self.models_dir = Path(self.config['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def check_all_models(self) -> Dict[str, Any]:
        """Check semua model dengan UI progress tracking"""
        try:
            self._start_check_operation()
            
            model_names = list(self.config['models'].keys())
            existing_models, missing_models, model_details = [], [], {}
            
            self.logger and self.logger.info(f"🔍 Checking {len(model_names)} models di {self.models_dir}")
            
            for i, model_name in enumerate(model_names):
                self._update_check_progress(i, len(model_names), model_name)
                
                model_detail = self._check_single_model(model_name)
                model_details[model_name] = model_detail
                
                (existing_models.append(model_name) if model_detail['exists'] and model_detail['valid'] 
                 else missing_models.append(model_name))
                
                self._log_model_status(model_detail)
            
            self._complete_check_operation(len(existing_models), len(model_names))
            
            return {
                'existing_models': existing_models, 'missing_models': missing_models,
                'total_count': len(model_names), 'existing_count': len(existing_models),
                'missing_count': len(missing_models), 'model_details': model_details,
                'models_dir': str(self.models_dir)
            }
            
        except Exception as e:
            self.logger and self.logger.error(f"💥 Error checking models: {str(e)}")
            return {'existing_models': [], 'missing_models': list(self.config['models'].keys()),
                   'total_count': len(self.config['models']), 'existing_count': 0,
                   'missing_count': len(self.config['models']), 'error': str(e), 'models_dir': str(self.models_dir)}
    
    def _start_check_operation(self) -> None:
        """Start check operation dengan progress tracker"""
        tracker = self.ui_components.get('tracker')
        tracker and tracker.show("Model Check")
        self._safe_update_progress(5, "Memulai pemeriksaan model")
    
    def _update_check_progress(self, current: int, total: int, model_name: str) -> None:
        """Update progress untuk current check"""
        progress = int(20 + (current / total) * 70) if total > 0 else 20  # 20-90%
        self._safe_update_progress(progress, f"Check {model_name} ({current+1}/{total})")
    
    def _complete_check_operation(self, existing_count: int, total_count: int) -> None:
        """Complete check operation"""
        summary_msg = f"Check selesai: {existing_count}/{total_count} model tersedia"
        self._safe_update_progress(100, summary_msg)
        
        tracker = self.ui_components.get('tracker')
        tracker and tracker.complete(summary_msg)
    
    def _safe_update_progress(self, progress: int, message: str) -> None:
        """Safe update progress dengan fallback"""
        update_fn = self.ui_components.get('update_primary')
        if update_fn:
            update_fn(progress, message)
        else:
            tracker = self.ui_components.get('tracker')
            tracker and tracker.update_primary(progress, message)
    
    def _check_single_model(self, model_name: str) -> Dict[str, Any]:
        """Check single model dengan detailed validation"""
        try:
            config = self.config['models'][model_name]
            file_path = self.models_dir / config['filename']
            
            model_detail = {
                'name': model_name, 'config_name': config.get('name', model_name),
                'filename': config.get('filename', ''), 'file_path': str(file_path),
                'expected_min_size': config.get('min_size_mb', 1) * 1024 * 1024,
                'exists': file_path.exists(), 'valid': False, 'actual_size': 0,
                'size_check': False, 'readable': False
            }
            
            if file_path.exists():
                try:
                    stat = file_path.stat()
                    model_detail.update({
                        'actual_size': stat.st_size,
                        'size_check': stat.st_size >= model_detail['expected_min_size'],
                        'readable': self._test_file_readable(file_path)
                    })
                    model_detail['valid'] = model_detail['size_check'] and model_detail['readable']
                except (PermissionError, OSError) as e:
                    model_detail.update({'readable': False, 'error': str(e)})
            
            return model_detail
            
        except Exception as e:
            return {'name': model_name, 'exists': False, 'valid': False, 'error': str(e)}
    
    def _test_file_readable(self, file_path: Path) -> bool:
        """Test if file is readable dengan one-liner"""
        try:
            with open(file_path, 'rb') as f: f.read(1024)
            return True
        except (PermissionError, OSError):
            return False
    
    def _log_model_status(self, model_detail: Dict[str, Any]) -> None:
        """Log model status dengan conditional logging"""
        if model_detail['exists'] and model_detail['valid']:
            size_str = ModelUtils.format_file_size(model_detail['actual_size'])
            self.logger and self.logger.info(f"✅ {model_detail['config_name']}: {size_str}")
        else:
            reason = self._get_missing_reason(model_detail)
            self.logger and self.logger.info(f"❌ {model_detail['config_name']}: {reason}")
    
    def _get_missing_reason(self, model_detail: Dict[str, Any]) -> str:
        """Get human-readable reason untuk missing/invalid model"""
        return ("file tidak ditemukan" if not model_detail['exists']
               else f"ukuran tidak valid ({model_detail.get('actual_size', 0) / (1024 * 1024):.1f}MB < {model_detail.get('expected_min_size', 0) / (1024 * 1024):.1f}MB)" if not model_detail.get('size_check', False)
               else "file tidak dapat dibaca" if not model_detail.get('readable', False)
               else "validasi gagal")