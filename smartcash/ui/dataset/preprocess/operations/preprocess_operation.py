"""
File: smartcash/ui/dataset/preprocess/operations/preprocess_operation.py
Description: Preprocessing operation handler
"""

from typing import Dict, Any, Optional, Callable
import asyncio
from smartcash.ui.dataset.preprocess.operations.base_preprocess_operation import BasePreprocessOperation
from smartcash.ui.dataset.preprocess.constants import (
    PreprocessingOperation, ProcessingPhase, PROGRESS_PHASES, SUCCESS_MESSAGES, ERROR_MESSAGES
)


class PreprocessOperation(BasePreprocessOperation):
    """
    Preprocessing operation handler.
    
    Features:
    - 🎯 Dataset preprocessing with YOLO normalization
    - 📊 Phase-based progress tracking (validation, processing, finalization)
    - 🔄 Backend API integration with progress callbacks
    - 🚨 Comprehensive error handling
    """
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any],
                 progress_callback: Optional[Callable] = None, 
                 log_callback: Optional[Callable] = None):
        """
        Initialize preprocessing operation.
        
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
        self.preprocessing_config = config.get('preprocessing', {})
        self.data_config = config.get('data', {})
        
        # Operation metadata
        self.operation_type = PreprocessingOperation.PREPROCESS
        self.current_phase = None
        self.processed_splits = []
        self.results = {}
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations for this handler."""
        return {
            'preprocess': self.execute
        }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute preprocessing operation.
        
        Args:
            **kwargs: Additional execution parameters
            
        Returns:
            Operation results dictionary
        """
        try:
            self.log_info("🚀 Starting dataset preprocessing operation")
            
            # Phase 1: Validation (20%) - use real backend
            await self._validation_phase_backend()
            
            # Phase 2: Processing (70%)
            await self._execute_phase(ProcessingPhase.PROCESSING)
            
            # Phase 3: Finalization (10%)
            await self._execute_phase(ProcessingPhase.FINALIZATION)
            
            # Mark operation as completed
            self.mark_completed()
            self.log_success(SUCCESS_MESSAGES['preprocessing_complete'])
            
            return {
                'success': True,
                'operation': self.operation_type.value,
                'message': SUCCESS_MESSAGES['preprocessing_complete'],
                'results': self.results,
                'processed_splits': self.processed_splits
            }
            
        except Exception as e:
            error_msg = f"Preprocessing operation failed: {str(e)}"
            self.log_error(error_msg)
            self.mark_failed(error_msg)
            
            return {
                'success': False,
                'operation': self.operation_type.value,
                'message': error_msg,
                'error': str(e)
            }
    
    async def _execute_phase(self, phase: ProcessingPhase) -> None:
        """
        Execute a specific processing phase.
        
        Args:
            phase: Processing phase to execute
        """
        self.current_phase = phase
        phase_config = PROGRESS_PHASES[phase]
        
        self.log_info(f"📋 Phase {phase.value}: {phase_config['description']}")
        
        if phase == ProcessingPhase.VALIDATION:
            await self._validation_phase()
        elif phase == ProcessingPhase.PROCESSING:
            await self._processing_phase()
        elif phase == ProcessingPhase.FINALIZATION:
            await self._finalization_phase()
        
        # Update overall progress
        self.update_progress(
            phase_config['weight'] * 100,
            f"✅ {phase_config['description']} complete"
        )
    
    async def _validation_phase(self) -> None:
        """Execute validation phase."""
        self.log_info("🔍 Validating dataset structure and configuration")
        
        # Check data directory
        data_dir = self.data_config.get('dir', 'data')
        if not data_dir:
            raise ValueError("Data directory not specified")
        
        # Check target splits
        target_splits = self.preprocessing_config.get('target_splits', ['train', 'valid'])
        if not target_splits:
            raise ValueError("No target splits specified")
        
        self.processed_splits = target_splits
        self.log_info(f"✅ Validation complete: {len(target_splits)} splits to process")
        
        # Simulate validation delay
        await asyncio.sleep(0.5)
    
    async def _processing_phase(self) -> None:
        """Execute main processing phase using real backend."""
        self.log_info("🔧 Processing dataset with YOLO normalization")
        
        try:
            # Import backend preprocessing API
            from smartcash.dataset.preprocessor import preprocess_dataset
            
            # Create progress callback for backend that maps to our operation progress
            def backend_progress_callback(level: str, current: int, total: int, message: str):
                if level == 'overall':
                    # Map to processing phase progress (20% base + 70% * progress)
                    progress = 20 + (70 * current / total) if total > 0 else 20
                    self.update_progress(progress, f"Processing: {message}")
                    self.log_info(f"📊 Overall progress: {current}/{total} - {message}")
                elif level == 'current':
                    self.log_info(f"  🔄 Current: {message}")
                elif level == 'phase':
                    self.log_info(f"📋 Phase: {message}")
            
            # Execute backend preprocessing with real backend integration
            self.log_info("🚀 Starting backend preprocessing operation")
            result = preprocess_dataset(
                config=self.config,
                progress_callback=backend_progress_callback,
                ui_components=self.ui_components
            )
            
            if result.get('success', False):
                self.results.update(result)
                stats = result.get('stats', {})
                
                # Log processing results
                self.log_success("✅ Processing phase completed successfully")
                if 'total_files' in stats:
                    self.log_info(f"📊 Processed {stats['total_files']} files")
                if 'configuration' in result:
                    config_info = result['configuration']
                    self.log_info(f"⚙️ Used preset: {config_info.get('normalization_preset', 'default')}")
                    self.log_info(f"📁 Splits: {', '.join(config_info.get('target_splits', []))}")
            else:
                error_msg = result.get('message', 'Backend processing failed')
                self.log_error(f"❌ Backend processing error: {error_msg}")
                raise RuntimeError(error_msg)
                
        except ImportError as e:
            error_msg = f"Backend preprocessing module not available: {str(e)}"
            self.log_error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            self.log_error(f"Processing failed: {str(e)}")
            raise
    
    async def _validation_phase_backend(self) -> None:
        """Execute validation phase using real backend validators."""
        self.log_info("🔍 Validating dataset structure and configuration")
        
        try:
            from smartcash.dataset.preprocessor import validate_dataset_structure, validate_filenames
            
            data_dir = self.data_config.get('dir', 'data')
            target_splits = self.preprocessing_config.get('target_splits', ['train', 'valid'])
            
            # Phase 1: Validate directory structure (10%)
            self.update_progress(5, "Validating directory structure")
            structure_result = validate_dataset_structure(
                data_dir=data_dir,
                splits=target_splits,
                auto_fix=True
            )
            
            if not structure_result.get('success', False):
                raise ValueError(f"Directory structure invalid: {structure_result.get('message', 'Unknown error')}")
            
            self.log_info(f"✅ Directory structure valid for splits: {', '.join(target_splits)}")
            
            # Phase 2: Validate filenames (10%)
            self.update_progress(15, "Validating filenames")
            filename_result = validate_filenames(
                data_dir=data_dir,
                splits=target_splits,
                auto_rename=True  # Auto-rename to research format
            )
            
            if filename_result.get('success', False):
                renamed_count = filename_result.get('total_renamed', 0)
                if renamed_count > 0:
                    self.log_info(f"✅ Renamed {renamed_count} files to research format")
                else:
                    self.log_info("✅ All filenames already in correct format")
            else:
                self.log_warning(f"⚠️ Filename validation issues: {filename_result.get('message', 'Unknown')}")
            
            self.processed_splits = target_splits
            self.log_success("✅ Validation phase completed using backend validators")
            
        except ImportError as e:
            error_msg = f"Backend validation modules not available: {str(e)}"
            self.log_error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            self.log_error(f"Validation failed: {str(e)}")
            raise
    
    async def _finalization_phase(self) -> None:
        """Execute finalization phase."""
        self.log_info("📊 Finalizing preprocessing results")
        
        # Generate final statistics
        stats = self.results.get('stats', {})
        total_files = stats.get('processed_files', 0)
        
        self.log_info(f"📈 Preprocessing Statistics:")
        self.log_info(f"  • Total files processed: {total_files}")
        self.log_info(f"  • Splits processed: {', '.join(self.processed_splits)}")
        self.log_info(f"  • Normalization preset: {self.preprocessing_config.get('normalization', {}).get('preset', 'yolov5s')}")
        
        # Simulate finalization delay
        await asyncio.sleep(0.3)
    
    def get_operation_status(self) -> Dict[str, Any]:
        """
        Get current operation status.
        
        Returns:
            Status dictionary
        """
        status = super().get_operation_status()
        status.update({
            'operation_type': self.operation_type.value,
            'current_phase': self.current_phase.value if self.current_phase else None,
            'processed_splits': self.processed_splits,
            'config_summary': {
                'preset': self.preprocessing_config.get('normalization', {}).get('preset', 'yolov5s'),
                'target_splits': self.preprocessing_config.get('target_splits', []),
                'batch_size': self.preprocessing_config.get('batch_size', 32)
            }
        })
        return status
    
    def cancel(self) -> bool:
        """
        Cancel the preprocessing operation.
        
        Returns:
            True if cancellation was successful
        """
        try:
            self.log_warning("⚠️ Cancelling preprocessing operation")
            self.mark_cancelled("Operation cancelled by user")
            return True
        except Exception as e:
            self.log_error(f"Failed to cancel operation: {e}")
            return False