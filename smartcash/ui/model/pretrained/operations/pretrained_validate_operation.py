"""
Validate operation for pretrained models.
"""

from typing import Dict, Any

from .pretrained_base_operation import PretrainedBaseOperation


class PretrainedValidateOperation(PretrainedBaseOperation):
    """Validate operation for pretrained models."""
    
    def execute_operation(self) -> Dict[str, Any]:
        """Execute validation operation with actual model validation."""
        try:
            self.log(f"🔍 Checking model files in {self.models_dir}", 'info')
            validation_results = self.validate_downloaded_models(self.models_dir)
            
            self.log("✅ Validation complete", 'success')
            
            # Count valid models
            valid_models = [model for model, result in validation_results.items() if result.get('valid', False)]
            invalid_models = [model for model, result in validation_results.items() if not result.get('valid', False)]
            
            if invalid_models:
                self.log(f"⚠️ Invalid models found: {', '.join(invalid_models)}", 'warning')
            
            return {
                'success': True,
                'message': f'Validation completed. Valid: {len(valid_models)}, Invalid: {len(invalid_models)}',
                'models_validated': list(validation_results.keys()),
                'validation_results': validation_results,
                'valid_models': valid_models,
                'invalid_models': invalid_models
            }
            
        except Exception as e:
            self.log(f"❌ Error in validate operation: {e}", 'error')
            return {'success': False, 'error': str(e)}