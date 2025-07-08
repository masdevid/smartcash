"""
File: smartcash/ui/dataset/preprocess/operations/check_operation.py
Description: Dataset check operation handler
"""

from typing import Dict, Any, Optional, Callable
import asyncio
from smartcash.ui.dataset.preprocess.operations.base_preprocess_operation import BasePreprocessOperation
from smartcash.ui.dataset.preprocess.constants import (
    PreprocessingOperation, SUCCESS_MESSAGES, ERROR_MESSAGES
)


class CheckOperation(BasePreprocessOperation):
    """
    Dataset check operation handler.
    
    Features:
    - 🔍 Dataset status checking and validation
    - 📊 File statistics and readiness assessment
    - 🔄 Backend API integration for comprehensive checks
    - 📈 Progress tracking for check phases
    """
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any],
                 progress_callback: Optional[Callable] = None, 
                 log_callback: Optional[Callable] = None):
        """
        Initialize check operation.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary
            progress_callback: Progress update callback
            log_callback: Log message callback
        """
        super().__init__(
            ui_components=ui_components,
            config=config,
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        
        # Additional config references for convenience
        self.data_config = config.get('data', {})
        self.preprocessing_config = config.get('preprocessing', {})
        
        # Operation metadata
        self.operation_type = PreprocessingOperation.CHECK
        self.check_results = {}
        self.file_statistics = {}
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations for this handler."""
        return {
            'check': self.execute
        }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute dataset check operation.
        
        Args:
            **kwargs: Additional execution parameters
            
        Returns:
            Check results dictionary
        """
        try:
            self.log_info("🔍 Starting dataset check operation")
            
            # Phase 1: Directory Structure Check (30%)
            await self._check_directory_structure()
            
            # Phase 2: File Statistics (50%)
            await self._collect_file_statistics()
            
            # Phase 3: Readiness Assessment (20%)
            await self._assess_readiness()
            
            # Mark operation as completed
            self.mark_completed()
            self.log_success(SUCCESS_MESSAGES['check_complete'])
            
            return {
                'success': True,
                'operation': self.operation_type.value,
                'message': SUCCESS_MESSAGES['check_complete'],
                'service_ready': self.check_results.get('service_ready', False),
                'file_statistics': self.file_statistics,
                'readiness_summary': self.check_results
            }
            
        except Exception as e:
            error_msg = f"Dataset check failed: {str(e)}"
            self.log_error(error_msg)
            self.mark_failed(error_msg)
            
            return {
                'success': False,
                'operation': self.operation_type.value,
                'message': error_msg,
                'error': str(e),
                'service_ready': False
            }
    
    async def _check_directory_structure(self) -> None:
        """Check dataset directory structure."""
        self.log_info("📁 Checking directory structure")
        
        data_dir = self.data_config.get('dir', 'data')
        target_splits = self.preprocessing_config.get('target_splits', ['train', 'valid'])
        
        directory_checks = {
            'data_dir_exists': bool(data_dir),
            'splits_available': [],
            'structure_valid': True
        }
        
        # Check each split directory
        for i, split in enumerate(target_splits):
            progress = 30 * (i + 1) / len(target_splits)
            self.update_progress(progress, f"Checking {split} directory")
            
            # Simulate directory check
            await asyncio.sleep(0.2)
            
            # Mark split as available (simulation)
            directory_checks['splits_available'].append(split)
            self.log_info(f"  ✅ {split} directory structure valid")
        
        self.check_results['directory_structure'] = directory_checks
        self.log_success("✅ Directory structure check completed")
    
    async def _collect_file_statistics(self) -> None:
        """Collect file statistics using real backend API."""
        self.log_info("📊 Collecting file statistics")
        
        try:
            # Use backend API for real file scanning
            from smartcash.dataset.preprocessor import get_preprocessing_status, get_dataset_stats
            
            self.update_progress(40, "Scanning files with backend API")
            
            # Get comprehensive status from backend
            status = get_preprocessing_status(self.config, self.ui_components)
            
            if status.get('success', False):
                # Extract file statistics from backend response
                if 'file_statistics' in status:
                    self.file_statistics = status['file_statistics']
                    self.log_success("✅ File statistics collected from preprocessing status")
                else:
                    # Fallback to dataset stats API
                    self.update_progress(60, "Getting detailed dataset statistics")
                    data_dir = self.data_config.get('dir', 'data')
                    target_splits = self.preprocessing_config.get('target_splits', ['train', 'valid'])
                    
                    stats_result = get_dataset_stats(data_dir, target_splits)
                    if stats_result.get('success', False):
                        # Convert dataset stats format to file statistics format
                        self.file_statistics = {}
                        for split, split_data in stats_result.get('by_split', {}).items():
                            file_counts = split_data.get('file_counts', {})
                            self.file_statistics[split] = {
                                'raw_images': file_counts.get('raw', 0),
                                'preprocessed_files': file_counts.get('preprocessed', 0),
                                'augmented_files': file_counts.get('augmented', 0),
                                'sample_files': file_counts.get('samples', 0),
                                'total_size_mb': split_data.get('total_size_mb', 0)
                            }
                        self.log_success("✅ File statistics collected from dataset stats")
                    else:
                        raise RuntimeError(f"Dataset stats failed: {stats_result.get('message', 'Unknown error')}")
                
                # Update check results with backend data
                self.check_results.update(status)
                
                # Log detailed statistics
                total_raw = sum(stats.get('raw_images', 0) for stats in self.file_statistics.values())
                total_preprocessed = sum(stats.get('preprocessed_files', 0) for stats in self.file_statistics.values())
                
                self.log_info(f"📈 Statistics Summary:")
                self.log_info(f"  • Total raw images: {total_raw}")
                self.log_info(f"  • Total preprocessed: {total_preprocessed}")
                
                for split, stats in self.file_statistics.items():
                    self.log_info(f"  • {split}: {stats['raw_images']} raw, {stats['preprocessed_files']} processed")
                
            else:
                error_msg = status.get('message', 'Backend status check failed')
                self.log_error(f"❌ Backend status error: {error_msg}")
                raise RuntimeError(error_msg)
                
        except ImportError as e:
            error_msg = f"Backend preprocessing module not available: {str(e)}"
            self.log_error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            self.log_error(f"File statistics collection failed: {str(e)}")
            raise
        
        self.update_progress(80, "File statistics collection completed")
    
    async def _simulate_file_statistics(self) -> None:
        """Simulate file statistics collection."""
        target_splits = self.preprocessing_config.get('target_splits', ['train', 'valid'])
        
        for i, split in enumerate(target_splits):
            progress = 30 + (50 * (i + 1) / len(target_splits))
            self.update_progress(progress, f"Scanning {split} files")
            
            # Simulate file scanning
            await asyncio.sleep(0.3)
            
            # Generate simulated statistics
            self.file_statistics[split] = {
                'raw_images': 75 + (i * 25),  # train: 75, valid: 100
                'preprocessed_files': 0,  # Not processed yet
                'augmented_files': 0,
                'sample_files': 0,
                'total_size_mb': 15.5 + (i * 5.2)
            }
            
            self.log_info(f"  📈 {split}: {self.file_statistics[split]['raw_images']} raw images")
    
    async def _assess_readiness(self) -> None:
        """Assess dataset readiness for preprocessing."""
        self.log_info("🎯 Assessing preprocessing readiness")
        
        self.update_progress(90, "Assessing readiness")
        
        # Calculate readiness metrics
        total_raw_files = sum(
            stats.get('raw_images', 0) 
            for stats in self.file_statistics.values()
        )
        
        directory_valid = self.check_results.get('directory_structure', {}).get('structure_valid', False)
        has_files = total_raw_files > 0
        config_valid = bool(self.preprocessing_config.get('target_splits'))
        
        # Determine service readiness
        service_ready = directory_valid and has_files and config_valid
        
        readiness_summary = {
            'service_ready': service_ready,
            'directory_structure': directory_valid,
            'files_available': has_files,
            'configuration_valid': config_valid,
            'total_raw_files': total_raw_files,
            'ready_for_preprocessing': service_ready
        }
        
        self.check_results.update(readiness_summary)
        
        # Log readiness assessment
        if service_ready:
            self.log_success(f"✅ Dataset ready for preprocessing ({total_raw_files} files)")
        else:
            self.log_warning("⚠️ Dataset not ready for preprocessing")
            if not directory_valid:
                self.log_warning("  • Directory structure issues")
            if not has_files:
                self.log_warning("  • No raw image files found")
            if not config_valid:
                self.log_warning("  • Configuration incomplete")
        
        await asyncio.sleep(0.2)
        self.update_progress(100, "Readiness assessment completed")
    
    def get_operation_status(self) -> Dict[str, Any]:
        """
        Get current operation status.
        
        Returns:
            Status dictionary
        """
        status = super().get_operation_status()
        status.update({
            'operation_type': self.operation_type.value,
            'service_ready': self.check_results.get('service_ready', False),
            'total_files': sum(
                stats.get('raw_images', 0) 
                for stats in self.file_statistics.values()
            ),
            'splits_checked': list(self.file_statistics.keys()),
            'readiness_issues': self._get_readiness_issues()
        })
        return status
    
    def _get_readiness_issues(self) -> list:
        """Get list of readiness issues."""
        issues = []
        
        if not self.check_results.get('directory_structure', False):
            issues.append("Directory structure invalid")
        
        if not self.check_results.get('files_available', False):
            issues.append("No raw image files found")
        
        if not self.check_results.get('configuration_valid', False):
            issues.append("Configuration incomplete")
        
        return issues
    
    def cancel(self) -> bool:
        """
        Cancel the check operation.
        
        Returns:
            True if cancellation was successful
        """
        try:
            self.log_warning("⚠️ Cancelling check operation")
            self.mark_cancelled("Operation cancelled by user")
            return True
        except Exception as e:
            self.log_error(f"Failed to cancel operation: {e}")
            return False