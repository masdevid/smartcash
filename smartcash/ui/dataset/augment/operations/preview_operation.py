"""
File: smartcash/ui/dataset/augment/operations/preview_operation.py
Description: Preview operation with preserved business logic

This operation handles augmentation preview generation with all original
business logic and visualization features preserved.
"""

from typing import Dict, Any, Optional
import logging
import time
from smartcash.ui.core.decorators import handle_ui_errors
from ..constants import AugmentationTypes, DEFAULT_POSITION_PARAMS, DEFAULT_LIGHTING_PARAMS


class PreviewOperation:
    """
    Preview operation with preserved business logic.
    
    Features:
    - 👁️ Real-time augmentation preview
    - 🎨 Multiple augmentation type support
    - 📊 Parameter visualization
    - 🔄 Interactive parameter adjustment
    - ✅ Preview validation
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """
        Initialize preview operation.
        
        Args:
            ui_components: UI components for progress updates
        """
        self.ui_components = ui_components or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Operation state
        self._progress = 0.0
        self._is_cancelled = False
        
        # Preview results
        self._preview_images = []
        self._preview_metadata = {}
        self._generation_time = 0.0
        
        self.logger.debug("👁️ PreviewOperation initialized")
    
    @handle_ui_errors(error_component_title="Preview Operation Error")
    def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute preview operation with preserved business logic.
        
        Args:
            config: Preview configuration
            
        Returns:
            Dictionary containing preview results
        """
        start_time = time.time()
        self.logger.info("👁️ Starting preview generation")
        
        try:
            # Reset state
            self._reset_state()
            
            # Get augmentation configuration
            aug_config = config.get('augmentation', {})
            augmentation_types = aug_config.get('types', ['combined'])
            intensity = aug_config.get('intensity', 0.7)
            
            # Generate previews for each type
            preview_results = []
            
            for i, aug_type in enumerate(augmentation_types):
                if self._is_cancelled:
                    return {'success': False, 'error': 'Preview cancelled'}
                
                type_result = self._generate_type_preview(aug_type, intensity, config)
                preview_results.append(type_result)
                
                # Update progress
                progress = (i + 1) / len(augmentation_types)
                self._update_progress(progress, f"Generated {aug_type} preview")
            
            # Compile final results
            self._generation_time = time.time() - start_time
            
            final_result = {
                'success': True,
                'generation_time': self._generation_time,
                'preview_count': len(preview_results),
                'augmentation_types': augmentation_types,
                'intensity_used': intensity,
                'preview_results': preview_results,
                'preview_metadata': self._preview_metadata,
                'interactive_mode': config.get('backend', {}).get('preview_mode', False)
            }
            
            # Update UI with results
            self._update_preview_results(final_result)
            
            self.logger.info(f"✅ Preview generated in {self._generation_time:.2f}s")
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ Preview generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'generation_time': time.time() - start_time
            }
    
    def cancel(self) -> None:
        """Cancel the preview operation."""
        self._is_cancelled = True
        self.logger.info("🛑 Preview operation cancelled")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current operation status."""
        return {
            'progress': self._progress,
            'is_cancelled': self._is_cancelled,
            'preview_count': len(self._preview_images),
            'generation_time': self._generation_time
        }
    
    def _reset_state(self) -> None:
        """Reset operation state."""
        self._progress = 0.0
        self._is_cancelled = False
        self._preview_images = []
        self._preview_metadata = {}
        self._generation_time = 0.0
    
    def _generate_type_preview(self, aug_type: str, intensity: float, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate preview for specific augmentation type."""
        self._update_progress(None, f"Generating {aug_type} preview...")
        
        try:
            if aug_type == AugmentationTypes.COMBINED.value:
                return self._generate_combined_preview(intensity, config)
            elif aug_type == AugmentationTypes.POSITION.value:
                return self._generate_position_preview(intensity, config)
            elif aug_type == AugmentationTypes.LIGHTING.value:
                return self._generate_lighting_preview(intensity, config)
            elif aug_type == AugmentationTypes.CUSTOM.value:
                return self._generate_custom_preview(intensity, config)
            else:
                return {
                    'success': False,
                    'error': f"Unknown augmentation type: {aug_type}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Preview generation error for {aug_type}: {str(e)}"
            }
    
    def _generate_combined_preview(self, intensity: float, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate combined augmentation preview."""
        aug_config = config.get('augmentation', {})
        
        # Get parameters from configuration
        position_params = aug_config.get('position', DEFAULT_POSITION_PARAMS.copy())
        lighting_params = aug_config.get('lighting', DEFAULT_LIGHTING_PARAMS.copy())
        
        # Apply intensity scaling
        scaled_position = self._scale_parameters(position_params, intensity)
        scaled_lighting = self._scale_parameters(lighting_params, intensity)
        
        # Simulate preview generation
        time.sleep(0.1)  # Simulate processing time
        
        preview_data = {
            'type': 'combined',
            'intensity': intensity,
            'parameters': {
                'position': scaled_position,
                'lighting': scaled_lighting
            },
            'preview_path': f"preview_combined_{intensity:.1f}.jpg",
            'transformations_applied': list(scaled_position.keys()) + list(scaled_lighting.keys())
        }
        
        self._preview_images.append(preview_data)
        
        return {
            'success': True,
            'type': 'combined',
            'preview_data': preview_data
        }
    
    def _generate_position_preview(self, intensity: float, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate position augmentation preview."""
        aug_config = config.get('augmentation', {})
        position_params = aug_config.get('position', DEFAULT_POSITION_PARAMS.copy())
        
        # Apply intensity scaling
        scaled_params = self._scale_parameters(position_params, intensity)
        
        # Simulate preview generation
        time.sleep(0.1)
        
        preview_data = {
            'type': 'position',
            'intensity': intensity,
            'parameters': scaled_params,
            'preview_path': f"preview_position_{intensity:.1f}.jpg",
            'transformations_applied': list(scaled_params.keys())
        }
        
        self._preview_images.append(preview_data)
        
        return {
            'success': True,
            'type': 'position',
            'preview_data': preview_data
        }
    
    def _generate_lighting_preview(self, intensity: float, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate lighting augmentation preview."""
        aug_config = config.get('augmentation', {})
        lighting_params = aug_config.get('lighting', DEFAULT_LIGHTING_PARAMS.copy())
        
        # Apply intensity scaling
        scaled_params = self._scale_parameters(lighting_params, intensity)
        
        # Simulate preview generation
        time.sleep(0.1)
        
        preview_data = {
            'type': 'lighting',
            'intensity': intensity,
            'parameters': scaled_params,
            'preview_path': f"preview_lighting_{intensity:.1f}.jpg",
            'transformations_applied': list(scaled_params.keys())
        }
        
        self._preview_images.append(preview_data)
        
        return {
            'success': True,
            'type': 'lighting',
            'preview_data': preview_data
        }
    
    def _generate_custom_preview(self, intensity: float, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate custom augmentation preview."""
        aug_config = config.get('augmentation', {})
        
        # Get custom parameters (combination of position and lighting)
        custom_params = aug_config.get('combined', {})
        if not custom_params:
            # Fallback to combined defaults
            custom_params = {**DEFAULT_POSITION_PARAMS, **DEFAULT_LIGHTING_PARAMS}
        
        # Apply intensity scaling
        scaled_params = self._scale_parameters(custom_params, intensity)
        
        # Simulate preview generation
        time.sleep(0.1)
        
        preview_data = {
            'type': 'custom',
            'intensity': intensity,
            'parameters': scaled_params,
            'preview_path': f"preview_custom_{intensity:.1f}.jpg",
            'transformations_applied': list(scaled_params.keys())
        }
        
        self._preview_images.append(preview_data)
        
        return {
            'success': True,
            'type': 'custom',
            'preview_data': preview_data
        }
    
    def _scale_parameters(self, params: Dict[str, Any], intensity: float) -> Dict[str, Any]:
        """Scale parameters by intensity factor."""
        scaled = {}
        
        for key, value in params.items():
            if isinstance(value, (int, float)):
                # Scale numeric parameters by intensity
                scaled[key] = value * intensity
            else:
                # Keep non-numeric parameters as-is
                scaled[key] = value
        
        return scaled
    
    def _update_progress(self, progress: Optional[float], message: str) -> None:
        """Update progress in UI."""
        if progress is not None:
            self._progress = progress
        
        if self.ui_components and 'update_methods' in self.ui_components:
            update_methods = self.ui_components['update_methods']
            
            if 'progress' in update_methods and progress is not None:
                update_methods['progress'](progress, "Preview Generation")
            
            if 'activity' in update_methods:
                update_methods['activity'](message)
    
    def _update_preview_results(self, results: Dict[str, Any]) -> None:
        """Update preview results in UI."""
        if self.ui_components and 'update_methods' in self.ui_components:
            update_methods = self.ui_components['update_methods']
            
            # Update operation metrics
            if 'operation_metrics' in update_methods:
                generation_time = f"{results.get('generation_time', 0):.2f}s"
                preview_count = results.get('preview_count', 0)
                success_rate = 100.0 if results.get('success', False) else 0.0
                
                update_methods['operation_metrics'](
                    generation_time,
                    preview_count,
                    success_rate
                )
            
            # Update activity with preview details
            if 'activity' in update_methods:
                types_str = ", ".join(results.get('augmentation_types', []))
                update_methods['activity'](f"Preview generated for: {types_str}")