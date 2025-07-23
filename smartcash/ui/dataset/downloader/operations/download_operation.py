"""
File: smartcash/ui/dataset/downloader/operations/download_operation.py
Description: Core download operation implementation with dual progress tracking.
"""

from typing import Dict, Any, Optional, Callable, TYPE_CHECKING

from .downloader_base_operation import DownloaderBaseOperation, DownloaderOperationPhase

if TYPE_CHECKING:
    from smartcash.ui.dataset.downloader.downloader_uimodule import DownloaderUIModule


class DownloadOperation(DownloaderBaseOperation):
    """
    Core download operation that handles dataset downloading with dual progress tracking.
    
    This class manages the execution of dataset downloads with:
    - Main progress bar showing overall download phases (n/total_phases) 
    - Secondary progress bar showing current backend progress (0-100%)
    - Phase-based progress mapping from backend callbacks
    - Comprehensive error handling and UI integration
    """

    # Download phases with their weights (must sum to 100)
    DOWNLOAD_PHASES = {
        'init': {'weight': 5, 'label': '⚙️ Inisialisasi', 'phase': DownloaderOperationPhase.INITIALIZING},
        'metadata': {'weight': 5, 'label': '📋 Metadata', 'phase': DownloaderOperationPhase.INITIALIZING},
        'backup': {'weight': 5, 'label': '💾 Backup', 'phase': DownloaderOperationPhase.VALIDATING},
        'download': {'weight': 30, 'label': '⬇️ Download', 'phase': DownloaderOperationPhase.DOWNLOADING},
        'extract': {'weight': 15, 'label': '📦 Ekstraksi', 'phase': DownloaderOperationPhase.EXTRACTING},
        'organize': {'weight': 10, 'label': '🗂️ Organisasi', 'phase': DownloaderOperationPhase.PROCESSING},
        'uuid_rename': {'weight': 10, 'label': '🏷️ UUID Rename', 'phase': DownloaderOperationPhase.PROCESSING},
        'validate': {'weight': 10, 'label': '✅ Validasi', 'phase': DownloaderOperationPhase.PROCESSING},
        'cleanup': {'weight': 5, 'label': '🧹 Cleanup', 'phase': DownloaderOperationPhase.FINALIZING},
        'complete': {'weight': 5, 'label': '🎉 Selesai', 'phase': DownloaderOperationPhase.COMPLETED}
    }

    def __init__(
        self,
        ui_module: 'DownloaderUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> None:
        """
        Initialize the download operation.
        
        Args:
            ui_module: Reference to the parent UI module
            config: Configuration dictionary for the download
            callbacks: Optional callbacks for operation events
        """
        # Initialize base class first
        super().__init__(ui_module, config, callbacks)
        
        # Progress tracking state
        self._reset_progress_tracking()
        
        # Register backend progress callback
        self._register_backend_callbacks()

    def _reset_progress_tracking(self) -> None:
        """Reset progress tracking state for a new operation."""
        self._completed_phases = set()
        self._current_phase_name = None
        self._current_phase_progress = 0
        self._overall_progress = 0
        self._phase_order = list(self.DOWNLOAD_PHASES.keys())
        
        # Validate phase weights sum to 100
        total_weight = sum(phase['weight'] for phase in self.DOWNLOAD_PHASES.values())
        if total_weight != 100:
            self.logger.warning(f"Phase weights sum to {total_weight}, not 100. Progress may be inaccurate.")

    def _register_backend_callbacks(self) -> None:
        """Register callbacks for backend progress updates."""
        if hasattr(self._ui_module, 'register_progress_callback'):
            self._ui_module.register_progress_callback('download', self._handle_download_progress)

    def _handle_download_progress(self, step: str, current: int, total: int = 100, message: str = "") -> None:
        """
        Handle download progress updates from the backend with dual progress tracking.
        
        Maps backend progress callbacks to dual progress tracker:
        - Main bar: Overall progress across all download phases (n/total_phases)
        - Secondary bar: Current progress within the active phase (0-100%)
        
        Args:
            step: Current download phase/step name
            current: Current progress within the phase (0-100)
            total: Total progress value (typically 100, unused but kept for compatibility)
            message: Optional progress message from backend
        """
        try:
            # Normalize step name
            step_name = step.lower().strip()
            
            # Get phase configuration
            phase_config = self.DOWNLOAD_PHASES.get(step_name)
            if not phase_config:
                # Handle unknown phases gracefully
                self.logger.warning(f"Unknown download phase: {step_name}")
                phase_config = {
                    'weight': 0,
                    'label': f"🔄 {step.title()}",
                    'phase': DownloaderOperationPhase.PROCESSING
                }
            
            # Update UI phase if changed
            if self.phase != phase_config['phase']:
                self.phase = phase_config['phase']
            
            # Track phase completion
            if step_name not in self._completed_phases and current >= 100:
                self._completed_phases.add(step_name)
                self.logger.info(f"✅ Phase completed: {phase_config['label']}")
            
            # Calculate overall progress based on completed phases + current phase progress
            completed_weight = sum(
                self.DOWNLOAD_PHASES[phase]['weight'] 
                for phase in self._completed_phases 
                if phase in self.DOWNLOAD_PHASES
            )
            
            # Current phase contribution (only if not completed)
            current_phase_weight = 0
            if step_name not in self._completed_phases:
                phase_weight = phase_config['weight']
                current_phase_weight = (current / 100.0) * phase_weight
            
            # Calculate total progress (capped at 100%)
            self._overall_progress = min(100, completed_weight + current_phase_weight)
            self._current_phase_progress = current
            self._current_phase_name = step_name
            
            # Calculate phase index for main progress display
            try:
                current_phase_index = self._phase_order.index(step_name) + 1
                total_phases = len(self._phase_order)
            except ValueError:
                # Handle unknown phases
                current_phase_index = len(self._completed_phases) + 1
                total_phases = len(self._phase_order)
            
            # Format progress messages
            main_message = f"{phase_config['label']} ({current_phase_index}/{total_phases})"
            if message:
                main_message += f" - {message}"
            
            secondary_message = f"{phase_config['label']}: {current}%"
            
            # Update dual progress tracker
            self.update_progress(
                progress=self._overall_progress,
                message=main_message,
                secondary_progress=current,
                secondary_message=secondary_message
            )
            
            # Reduce logging verbosity - only log phase transitions, not progress milestones
            # Progress milestones are handled by progress tracker, not logger
            if step_name != getattr(self, '_last_logged_step', None):
                self.logger.info(f"🔄 {main_message}")
                self._last_logged_step = step_name
                
        except Exception as e:
            self.logger.error(f"Error handling download progress: {e}", exc_info=True)

    def _should_log_progress(self, step: str, current: int) -> bool:
        """
        Determine if progress should be logged.
        
        DEPRECATED: This method is now simplified since we only log phase 
        transitions, not progress milestones. Progress milestones are 
        handled by the progress tracker to reduce logging verbosity.
        
        Args:
            step: Current step name
            current: Current progress value
            
        Returns:
            bool: True if progress should be logged (now only for phase transitions)
        """
        # Only log phase transitions, not progress milestones
        return step != getattr(self, '_last_logged_step', None)

    def execute(self) -> Dict[str, Any]:
        """
        Execute the download operation with comprehensive error handling.
        
        Returns:
            Dict[str, Any]: Operation results containing:
                - success: Boolean indicating operation success
                - message: Summary message
                - download_path: Path to downloaded dataset (if successful)
                - file_count: Number of files downloaded (if successful)
                - total_size: Total size of downloaded data (if successful)
                - error: Error details (if failed)
        """
        try:
            self.log_operation_start("Download Dataset")
            self._reset_progress_tracking()
            
            # Phase 1: Validate configuration and API key
            self.phase = DownloaderOperationPhase.INITIALIZING
            self.update_progress(0, "🔧 Memvalidasi konfigurasi...")
            
            result = self._validate_and_prepare()
            if not result['success']:
                return result
            
            api_key = result['api_key']
            backend_config = result['backend_config']
            
            # Phase 2: Initialize download service
            self.update_progress(5, "🚀 Menginisialisasi layanan download...")
            
            download_service = self.get_backend_api('download_service')
            if not download_service:
                return self._handle_error("Layanan download tidak tersedia")
            
            # Phase 3: Create downloader instance
            self.update_progress(10, "⚙️ Menyiapkan downloader...")
            
            try:
                downloader = download_service(backend_config)
                if not downloader:
                    return self._handle_error("Gagal menginisialisasi downloader")
            except Exception as e:
                return self._handle_error("Gagal membuat downloader instance", e)
            
            # Phase 4: Setup progress callback
            if hasattr(downloader, 'set_progress_callback'):
                downloader.set_progress_callback(self._handle_download_progress)
                self.logger.debug("✅ Progress callback terdaftar")
            
            # Phase 5: Execute download with backend progress tracking
            self.update_progress(15, "📥 Memulai download dataset...")
            
            try:
                # Update config with required parameters
                self.config.update({
                    'workspace': self.config['data']['roboflow']['workspace'],
                    'project': self.config['data']['roboflow']['project'],
                    'version': self.config['data']['roboflow']['version'],
                    'api_key': api_key
                })
                download_result = downloader.download_dataset()
            except Exception as e:
                return self._handle_error("Download gagal", e)
            
            # Phase 6: Process results
            if not download_result or download_result.get('status') != 'success':
                error_msg = download_result.get('message', 'Download gagal tanpa detail error') if download_result else 'Download gagal'
                return self._handle_error(error_msg)
            
            # Phase 7: Finalize and return success
            self.phase = DownloaderOperationPhase.COMPLETED
            self.update_progress(100, "✅ Download berhasil diselesaikan!")
            
            file_count = download_result.get('file_count', 0)
            total_size = download_result.get('total_size', '0B')
            download_path = download_result.get('download_path', '')
            
            self.log_operation_complete("Download Dataset")
            
            # Create result dictionary
            result = {
                'success': True,
                'message': f'Dataset berhasil didownload: {file_count} file ({total_size})',
                'download_path': download_path,
                'file_count': file_count,
                'total_size': total_size,
                'operation_id': self.operation_id
            }
            
            # Execute summary callback with formatted summary
            try:
                from ..components.operation_summary import update_download_summary
                summary_html = self._format_download_summary(result)
                self._execute_callback('on_success', summary_html)
            except Exception as e:
                self.logger.warning(f"Failed to update download summary: {e}")
            
            return result
            
        except Exception as e:
            return self._handle_error("Error tidak terduga saat download", e)

    def _format_download_summary(self, result: Dict[str, Any]) -> str:
        """Format download operation result into markdown for HTML conversion."""
        file_count = result.get('file_count', 0)
        total_size = result.get('total_size', '0B')
        download_path = result.get('download_path', 'N/A')
        
        markdown_content = f"""
## 📥 Ringkasan Download Dataset

### Status Operasi
✅ **Download Berhasil Diselesaikan**

### Statistik Download
- **File Downloaded**: 📁 {file_count:,} file
- **Total Size**: 💾 {total_size}
- **Download Path**: 📂 `{download_path}`

### Detail Operasi
| Kategori | Detail |
|:---------|:-------|
| Status | ✅ Berhasil |
| File Count | {file_count:,} file |
| Data Size | {total_size} |
| Location | {download_path} |

---

🎉 **Dataset berhasil didownload dan siap digunakan untuk preprocessing!**
"""
        
        # Convert markdown to HTML using the new formatter
        from smartcash.ui.core.utils import format_summary_to_html
        return format_summary_to_html(
            markdown_content, 
            title="📥 Download Results", 
            module_name="download"
        )

    def _validate_and_prepare(self) -> Dict[str, Any]:
        """
        Validate configuration and prepare for download.
        
        Returns:
            Dict containing validation results and prepared data
        """
        try:
            # Get and validate API key
            api_key = self._get_api_key()
            if not api_key:
                return {
                    'success': False,
                    'message': 'API key tidak ditemukan. Silakan konfigurasi API key yang valid.',
                    'error': 'missing_api_key'
                }
            
            # Validate and convert configuration
            backend_config, error = self.validate_downloader_config()
            if error:
                return {
                    'success': False,
                    'message': f'Validasi konfigurasi gagal: {error}',
                    'error': 'config_validation_failed'
                }
            
            return {
                'success': True,
                'api_key': api_key,
                'backend_config': backend_config
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error saat validasi: {str(e)}',
                'error': 'validation_error'
            }

    def _get_api_key(self) -> Optional[str]:
        """
        Get API key from config or secret manager.
        
        Returns:
            str: API key if found, None otherwise
        """
        # Check config first
        try:
            roboflow_config = self.config.get('data', {}).get('roboflow', {})
            if 'api_key' in roboflow_config and roboflow_config['api_key']:
                return roboflow_config['api_key']
        except (KeyError, TypeError, AttributeError):
            pass
        
        # Fall back to secret manager
        return self.get_roboflow_api_key()

    def _handle_error(self, message: str, exception: Optional[Exception] = None) -> Dict[str, Any]:
        """
        Handle operation errors with proper logging and progress updates.
        
        Args:
            message: Error message
            exception: Optional exception that caused the error
            
        Returns:
            Error response dictionary
        """
        error_details = str(exception) if exception else "No additional details"
        
        # Log the error
        if exception:
            self.logger.error(f"{message}: {error_details}", exc_info=True)
        else:
            self.logger.error(message)
        
        # Update UI state
        self.phase = DownloaderOperationPhase.FAILED
        self.update_progress(0, f"❌ {message}")
        
        return {
            'success': False,
            'message': message,
            'error': error_details,
            'operation_id': self.operation_id
        }