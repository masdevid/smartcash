"""
Symlink Operation (Optimized) - Enhanced Mixin Integration
Create symbolic links with backend service integration.
"""

import os
from typing import Dict, Any, Optional, Callable, List
from smartcash.ui.components.operation_container import OperationContainer
from .base_colab_operation import BaseColabOperation


class SymlinkOperation(BaseColabOperation):
    """Optimized symlink operation with enhanced mixin integration."""
    
    def __init__(self, operation_name: str, config: Dict[str, Any], operation_container: Optional[OperationContainer] = None, **kwargs):
        super().__init__(operation_name, config, operation_container, **kwargs)
    
    def get_operations(self) -> Dict[str, Callable]:
        return {'create_symlinks': self.execute_create_symlinks}
    
    def execute_create_symlinks(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Create symbolic links with enhanced backend service integration."""
        def execute_operation():
            progress_steps = self.get_progress_steps('symlink')
            
            # Step 1: Initialize Symlink Creation Process (Direct Implementation)
            self.update_progress_safe(progress_callback, progress_steps[0]['progress'], progress_steps[0]['message'], progress_steps[0].get('phase_progress', 0))
            
            # Direct symlink creation without backend services
            self.log_info("Starting symlink creation process...")
            init_result = {'success': True, 'message': 'Symlink creation initialized directly'}
            
            # Step 2: Prepare Symlink Mappings
            self.update_progress_safe(progress_callback, progress_steps[1]['progress'], progress_steps[1]['message'], progress_steps[1].get('phase_progress', 0))
            
            symlink_mappings = self._prepare_symlink_mappings()
            if not symlink_mappings:
                return self.create_error_result("No symlink mappings configured")
            
            # Step 3: Create Symlinks
            self.update_progress_safe(progress_callback, progress_steps[2]['progress'], progress_steps[2]['message'], progress_steps[2].get('phase_progress', 0))
            
            creation_results = []
            total_links = len(symlink_mappings)
            
            for i, (source, target) in enumerate(symlink_mappings.items()):
                # Update progress for each symlink
                link_progress = int(progress_steps[2]['progress'] + (progress_steps[3]['progress'] - progress_steps[2]['progress']) * (i / total_links))
                self.update_progress_safe(progress_callback, link_progress, f"Creating symlink: {os.path.basename(target)}", 
                                        int(progress_steps[2].get('phase_progress', 0) + (progress_steps[3].get('phase_progress', 0) - progress_steps[2].get('phase_progress', 0)) * (i / total_links)))
                
                result = self._create_single_symlink(source, target)
                creation_results.append(result)
                
                if not result['success']:
                    self.log_warning(f"Symlink creation failed: {result['error']}")
            
            # Step 4: Verification and Sync
            self.update_progress_safe(progress_callback, progress_steps[3]['progress'], progress_steps[3]['message'], progress_steps[3].get('phase_progress', 0))
            
            # Verify all symlinks
            verification_results = self._verify_symlinks(symlink_mappings)
            
            # Log symlink creation status (no complex module sync needed)
            successful_links = [r for r in creation_results if r['success']]
            self.log_info(f"Symlinks created: {len(successful_links)}/{total_links}")
            sync_result = {'success': True, 'message': f'Created {len(successful_links)} symlinks'}
            
            return self.create_success_result(
                f'Created {len(successful_links)}/{total_links} symlinks',
                creation_results=creation_results,
                verification_results=verification_results,
                sync_result=sync_result,
                total_links=total_links,
                successful_links=len(successful_links)
            )
        
        return self.execute_with_error_handling(execute_operation)
    
    def _prepare_symlink_mappings(self) -> Dict[str, str]:
        """Prepare symlink mappings from configuration."""
        try:
            paths_config = self.config.get('paths', {})
            drive_base = paths_config.get('drive_base', '/content/drive/MyDrive/SmartCash')
            colab_base = paths_config.get('colab_base', '/content')
            
            # Standard symlink mappings
            mappings = {
                f"{drive_base}/data": f"{colab_base}/data",
                f"{drive_base}/models": f"{colab_base}/models", 
                f"{drive_base}/configs": f"{colab_base}/configs",
                f"{drive_base}/outputs": f"{colab_base}/outputs",
                f"{drive_base}/logs": f"{colab_base}/logs"
            }
            
            # Filter to only existing source directories
            filtered_mappings = {}
            for source, target in mappings.items():
                if os.path.exists(source):
                    filtered_mappings[source] = target
                else:
                    self.log_info(f"Source directory does not exist, skipping: {source}")
            
            return filtered_mappings
        except Exception as e:
            self.log_error(f"Failed to prepare symlink mappings: {e}")
            return {}
    
    def _create_single_symlink(self, source: str, target: str) -> Dict[str, Any]:
        """Create a single symlink with enhanced error handling."""
        try:
            # Check if source exists
            if not os.path.exists(source):
                return {'success': False, 'source': source, 'target': target, 'error': 'Source does not exist'}
            
            # Create target directory if needed
            target_dir = os.path.dirname(target)
            os.makedirs(target_dir, exist_ok=True)
            
            # Remove existing target if it exists
            if os.path.exists(target) or os.path.islink(target):
                if os.path.islink(target):
                    os.unlink(target)
                    self.log_info(f"Removed existing symlink: {target}")
                else:
                    return {'success': False, 'source': source, 'target': target, 'error': 'Target exists and is not a symlink'}
            
            # Create symlink
            os.symlink(source, target)
            
            # Verify symlink creation
            if os.path.islink(target) and os.readlink(target) == source:
                self.log_success(f"Symlink created: {target} -> {source}")
                return {'success': True, 'source': source, 'target': target}
            else:
                return {'success': False, 'source': source, 'target': target, 'error': 'Symlink verification failed'}
                
        except Exception as e:
            return {'success': False, 'source': source, 'target': target, 'error': str(e)}
    
    def _verify_symlinks(self, mappings: Dict[str, str]) -> Dict[str, Any]:
        """Verify all symlinks with cross-module validation."""
        try:
            verified_links = []
            broken_links = []
            
            for source, target in mappings.items():
                if os.path.islink(target):
                    if os.readlink(target) == source and os.path.exists(source):
                        verified_links.append({'source': source, 'target': target, 'status': 'verified'})
                    else:
                        broken_links.append({'source': source, 'target': target, 'status': 'broken'})
                else:
                    broken_links.append({'source': source, 'target': target, 'status': 'missing'})
            
            # Cross-module verification
            cross_verification = self.validate_cross_module_configs(['model', 'dataset'])
            
            return {
                'verified_links': verified_links,
                'broken_links': broken_links,
                'total_checked': len(mappings),
                'verification_success': len(broken_links) == 0,
                'cross_module_verification': cross_verification
            }
        except Exception as e:
            return {'verification_error': str(e), 'verification_success': False}
    
    def get_progress_steps(self, operation_type: str = 'symlink') -> list:
        """Get optimized progress steps for symlink operation."""
        return [
            {'progress': 10, 'message': '🔧 Initializing services...', 'phase_progress': 25},
            {'progress': 25, 'message': '📋 Preparing symlink mappings...', 'phase_progress': 40},
            {'progress': 60, 'message': '🔗 Creating symlinks...', 'phase_progress': 75},
            {'progress': 100, 'message': '✅ Symlinks verified', 'phase_progress': 100}
        ]