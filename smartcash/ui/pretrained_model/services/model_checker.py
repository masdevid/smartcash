"""
File: smartcash/ui/pretrained_model/services/model_checker.py
Deskripsi: Service khusus untuk check existing models dengan comprehensive progress tracking
"""

from pathlib import Path
from typing import Dict, Any, List
from smartcash.ui.pretrained_model.utils.model_utils import ModelUtils, ProgressTracker

class ModelChecker:
    """Service untuk check existing models dengan detailed progress dan validation"""
    
    def __init__(self, config: Dict[str, Any], logger=None, progress_tracker: ProgressTracker = None):
        self.config = config
        self.logger = logger
        self.progress_tracker = progress_tracker
        self.models_dir = Path(config.get('models_dir', '/content/models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def check_all_models(self) -> Dict[str, Any]:
        """Check semua model dengan step-by-step progress dan detailed validation"""
        try:
            if self.progress_tracker:
                self.progress_tracker.next_step('CHECK_MODELS', "Memeriksa model tersedia")
            
            model_names = ModelUtils.get_all_model_names()
            existing_models = []
            missing_models = []
            model_details = {}
            
            self.logger and self.logger.info(f"🔍 Checking {len(model_names)} models di {self.models_dir}")
            
            for i, model_name in enumerate(model_names):
                if self.progress_tracker:
                    self.progress_tracker.update_current_step(
                        (i * 100) // len(model_names),
                        f"Check {model_name} ({i+1}/{len(model_names)})"
                    )
                
                model_detail = self._check_single_model(model_name)
                model_details[model_name] = model_detail
                
                if model_detail['exists'] and model_detail['valid']:
                    existing_models.append(model_name)
                    config = ModelUtils.get_model_config(model_name)
                    size_str = ModelUtils.format_file_size(model_detail['actual_size'])
                    self.logger and self.logger.info(f"✅ {config['name']}: {size_str}")
                else:
                    missing_models.append(model_name)
                    config = ModelUtils.get_model_config(model_name)
                    reason = self._get_missing_reason(model_detail)
                    self.logger and self.logger.info(f"❌ {config['name']}: {reason}")
            
            # Summary info
            total_count = len(model_names)
            existing_count = len(existing_models)
            missing_count = len(missing_models)
            
            if self.progress_tracker:
                summary_msg = f"Check selesai: {existing_count}/{total_count} model tersedia"
                self.progress_tracker.update_current_step(100, summary_msg)
            
            return {
                'existing_models': existing_models,
                'missing_models': missing_models,
                'total_count': total_count,
                'existing_count': existing_count,
                'missing_count': missing_count,
                'model_details': model_details,
                'models_dir': str(self.models_dir)
            }
            
        except Exception as e:
            self.logger and self.logger.error(f"💥 Error checking models: {str(e)}")
            return {
                'existing_models': [],
                'missing_models': ModelUtils.get_all_model_names(),
                'total_count': len(ModelUtils.get_all_model_names()),
                'existing_count': 0,
                'missing_count': len(ModelUtils.get_all_model_names()),
                'error': str(e),
                'models_dir': str(self.models_dir)
            }
    
    def _check_single_model(self, model_name: str) -> Dict[str, Any]:
        """Check single model dengan detailed info"""
        try:
            config = ModelUtils.get_model_config(model_name)
            file_path = ModelUtils.get_model_file_path(model_name, str(self.models_dir))
            
            model_detail = {
                'name': model_name,
                'config_name': config.get('name', model_name),
                'filename': config.get('filename', ''),
                'expected_min_size': config.get('min_size_mb', 1) * 1024 * 1024,
                'file_path': str(file_path),
                'exists': file_path.exists(),
                'valid': False,
                'actual_size': 0,
                'size_check': False,
                'readable': False
            }
            
            if file_path.exists():
                try:
                    # Check file stats
                    stat = file_path.stat()
                    model_detail['actual_size'] = stat.st_size
                    model_detail['size_check'] = stat.st_size >= model_detail['expected_min_size']
                    
                    # Check if file is readable
                    with open(file_path, 'rb') as f:
                        f.read(1024)  # Try to read first 1KB
                    model_detail['readable'] = True
                    
                    # Overall validation
                    model_detail['valid'] = model_detail['size_check'] and model_detail['readable']
                    
                except (PermissionError, OSError) as e:
                    model_detail['readable'] = False
                    model_detail['error'] = str(e)
            
            return model_detail
            
        except Exception as e:
            return {
                'name': model_name,
                'exists': False,
                'valid': False,
                'error': str(e)
            }
    
    def _get_missing_reason(self, model_detail: Dict[str, Any]) -> str:
        """Get human-readable reason untuk missing/invalid model"""
        if not model_detail['exists']:
            return "file tidak ditemukan"
        elif not model_detail.get('size_check', False):
            actual_mb = model_detail.get('actual_size', 0) / (1024 * 1024)
            expected_mb = model_detail.get('expected_min_size', 0) / (1024 * 1024)
            return f"ukuran tidak valid ({actual_mb:.1f}MB < {expected_mb:.1f}MB)"
        elif not model_detail.get('readable', False):
            return "file tidak dapat dibaca"
        else:
            return "validasi gagal"
    
    def get_model_status_summary(self) -> str:
        """Get summary status dalam format human-readable"""
        result = self.check_all_models()
        existing_count = result.get('existing_count', 0)
        total_count = result.get('total_count', 0)
        
        if existing_count == total_count:
            return f"✅ Semua {total_count} model tersedia dan valid"
        elif existing_count == 0:
            return f"❌ Tidak ada model yang tersedia ({total_count} model perlu diunduh)"
        else:
            missing_count = total_count - existing_count
            return f"⚠️ {existing_count}/{total_count} model tersedia ({missing_count} perlu diunduh)"