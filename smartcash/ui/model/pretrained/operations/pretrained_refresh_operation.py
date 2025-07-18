"""
Refresh operation for pretrained models.
"""

import datetime
from typing import Dict, Any

from .pretrained_base_operation import PretrainedBaseOperation


class PretrainedRefreshOperation(PretrainedBaseOperation):
    """Refresh operation for pretrained models."""
    
    def execute_operation(self) -> Dict[str, Any]:
        """Execute refresh operation with actual directory scanning."""
        try:
            self.log(f"🔄 Refreshing model status in {self.models_dir}", 'info')
            
            # Scan model directory
            self.log("📁 Scanning model directory", 'info')
            models_found = self.scan_model_directory(self.models_dir)
            
            # Update model information
            self.log("📊 Updating model information", 'info')
            validation_results = self.validate_downloaded_models(self.models_dir)
            
            self.log("✅ Refresh complete", 'success')
            
            return {
                'success': True,
                'message': f'Refresh completed. Found {len(models_found)} model files',
                'models_found': models_found,
                'validation_results': validation_results,
                'refresh_time': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.log(f"❌ Error in refresh operation: {e}", 'error')
            return {'success': False, 'error': str(e)}