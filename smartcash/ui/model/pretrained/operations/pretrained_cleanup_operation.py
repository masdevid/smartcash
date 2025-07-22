"""
Cleanup operation for pretrained models.
"""

import os
from typing import Dict, Any

from .pretrained_base_operation import PretrainedBaseOperation


class PretrainedCleanupOperation(PretrainedBaseOperation):
    """Cleanup operation for pretrained models."""
    
    def execute_operation(self) -> Dict[str, Any]:
        """Execute cleanup operation with actual file cleanup."""
        try:
            self.log(f"🗑️ Scanning for corrupted files in {self.models_dir}", 'info')
            
            # Find corrupted/invalid files
            validation_results = self.validate_downloaded_models(self.models_dir)
            invalid_files = []
            space_freed = 0
            removed_models = []
            
            for model_name, result in validation_results.items():
                if not result.get('valid', False):
                    file_path = result.get('file_path')
                    if file_path and os.path.exists(file_path):
                        try:
                            file_size = os.path.getsize(file_path)
                            file_size_mb = file_size / (1024 * 1024)
                            
                            # More informative log with model name and size
                            self.log(f"🗑️ Pretrained model {model_name} removed ({file_size_mb:.1f}MB)", 'info')
                            os.remove(file_path)
                            
                            invalid_files.append(file_path)
                            removed_models.append(f"{model_name} ({file_size_mb:.1f}MB)")
                            space_freed += file_size
                        except Exception as e:
                            self.log(f"⚠️ Failed to remove {model_name}: {e}", 'warning')
            
            # Clean up empty directories
            self.log("📁 Organizing model directory", 'info')
            if os.path.exists(self.models_dir):
                try:
                    # Remove empty subdirectories
                    for root, dirs, _ in os.walk(self.models_dir, topdown=False):
                        for dir_name in dirs:
                            dir_path = os.path.join(root, dir_name)
                            if not os.listdir(dir_path):
                                os.rmdir(dir_path)
                                self.log(f"🗑️ Removed empty directory: {dir_path}", 'info')
                except Exception as e:
                    self.log(f"⚠️ Error organizing directories: {e}", 'warning')
            
            space_freed_mb = space_freed / (1024 * 1024)
            
            if removed_models:
                self.log(f"✅ Cleanup complete: removed {len(removed_models)} invalid models, freed {space_freed_mb:.1f}MB", 'success')
                self.log(f"🗑️ Removed models: {', '.join(removed_models)}", 'info')
            else:
                self.log("✅ Cleanup complete: no invalid models found", 'success')
            
            return {
                'success': True,
                'message': f'Cleanup completed. Removed {len(invalid_files)} invalid files, freed {space_freed_mb:.2f}MB',
                'files_cleaned': len(invalid_files),
                'space_freed': f'{space_freed_mb:.2f}MB',
                'invalid_files_removed': invalid_files,
                'removed_models': removed_models
            }
            
        except Exception as e:
            self.log(f"❌ Error in cleanup operation: {e}", 'error')
            return {'success': False, 'error': str(e)}