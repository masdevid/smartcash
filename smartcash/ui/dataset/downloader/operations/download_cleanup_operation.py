"""
File: smartcash/ui/dataset/downloader/operations/download_cleanup_operation.py
Description: Dataset cleanup operation implementation for the downloader module.
"""

from typing import Dict, Any, Optional, Callable, TYPE_CHECKING

from .downloader_base_operation import DownloaderBaseOperation, DownloaderOperationPhase

if TYPE_CHECKING:
    from smartcash.ui.dataset.downloader.downloader_uimodule import DownloaderUIModule


class DownloadCleanupOperation(DownloaderBaseOperation):
    """
    Dataset cleanup operation that handles dataset file removal and cleanup.
    
    This class manages the execution of dataset cleanup with:
    - Progress tracking for cleanup operations
    - Safe file deletion with confirmation
    - Space reclamation tracking
    - UI integration for confirmation and results display
    """

    # Cleanup phases with their weights (must sum to 100)
    CLEANUP_PHASES = {
        'init': {'weight': 10, 'label': '⚙️ Inisialisasi', 'phase': DownloaderOperationPhase.INITIALIZING},
        'scan_targets': {'weight': 20, 'label': '🔍 Scan Target', 'phase': DownloaderOperationPhase.PROCESSING},
        'validate_targets': {'weight': 10, 'label': '✅ Validasi Target', 'phase': DownloaderOperationPhase.PROCESSING},
        'delete_files': {'weight': 50, 'label': '🗑️ Hapus File', 'phase': DownloaderOperationPhase.PROCESSING},
        'cleanup_dirs': {'weight': 10, 'label': '📁 Cleanup Directory', 'phase': DownloaderOperationPhase.FINALIZING}
    }

    def __init__(
        self,
        ui_module: 'DownloaderUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> None:
        """
        Initialize the cleanup operation.
        
        Args:
            ui_module: Reference to the parent UI module
            config: Configuration dictionary for the cleanup
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
        self._phase_order = list(self.CLEANUP_PHASES.keys())
        
        # Validate phase weights sum to 100
        total_weight = sum(phase['weight'] for phase in self.CLEANUP_PHASES.values())
        if total_weight != 100:
            self.logger.warning(f"Cleanup phase weights sum to {total_weight}, not 100. Progress may be inaccurate.")

    def _register_backend_callbacks(self) -> None:
        """Register callbacks for backend progress updates."""
        if hasattr(self._ui_module, 'register_progress_callback'):
            self._ui_module.register_progress_callback('cleanup', self._handle_cleanup_progress)

    def _handle_cleanup_progress(self, step: str, current: int, total: int = 100, message: str = "") -> None:
        """
        Handle cleanup progress updates from the backend with dual progress tracking.
        
        Maps backend progress callbacks to dual progress tracker:
        - Main bar: Overall progress across all cleanup phases (n/total_phases)
        - Secondary bar: Current progress within the active phase (0-100%)
        
        Args:
            step: Current cleanup phase/step name
            current: Current progress within the phase (0-100)
            total: Total progress value (typically 100, unused but kept for compatibility)
            message: Optional progress message from backend
        """
        try:
            # Normalize step name
            step_name = step.lower().strip()
            
            # Get phase configuration
            phase_config = self.CLEANUP_PHASES.get(step_name)
            if not phase_config:
                # Handle unknown phases gracefully
                self.logger.warning(f"Unknown cleanup phase: {step_name}")
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
                self.CLEANUP_PHASES[phase]['weight'] 
                for phase in self._completed_phases 
                if phase in self.CLEANUP_PHASES
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
            
            # Log milestone progress
            if self._should_log_progress(step_name, current):
                self.logger.info(
                    f"📊 Cleanup Progress: {main_message} | "
                    f"Overall: {self._overall_progress:.1f}% | "
                    f"Phase: {current}%"
                )
                
        except Exception as e:
            self.logger.error(f"Error handling cleanup progress: {e}", exc_info=True)

    def _should_log_progress(self, step: str, current: int) -> bool:
        """
        Determine if progress should be logged.
        
        Args:
            step: Current step name
            current: Current progress value
            
        Returns:
            bool: True if progress should be logged
        """
        # Log at milestone percentages or phase transitions
        milestones = [0, 25, 50, 75, 100]
        return current in milestones or step != getattr(self, '_last_logged_step', None)

    def get_cleanup_targets(self) -> Dict[str, Any]:
        """
        Get cleanup targets from backend service.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - success: Boolean indicating operation success
                - message: Summary message
                - targets: Cleanup targets data (if successful)
                - total_files: Number of files to be cleaned (if successful)
                - total_size: Total size to be cleaned (if successful)
                - error: Error details (if failed)
        """
        try:
            self.logger.info("🔍 Mencari file yang dapat dibersihkan...")
            
            # Get scanner from backend API
            scanner_factory = self.get_backend_api('scanner')
            if not scanner_factory:
                return self._handle_error("Scanner service tidak tersedia")
            
            scanner = scanner_factory(self.logger)
            if not scanner:
                return self._handle_error("Gagal membuat scanner instance")
            
            # Get cleanup targets
            targets_result = scanner.get_cleanup_targets()
            
            if not targets_result or 'summary' not in targets_result:
                return self._handle_error("Gagal mendapatkan cleanup targets - respons tidak valid")
            
            summary = targets_result.get('summary', {})
            total_files = summary.get('total_files', 0)
            total_size = summary.get('total_size', '0B')
            
            if total_files == 0:
                self.logger.info("✅ Tidak ada file untuk dibersihkan")
                return {
                    'success': True,
                    'message': "Tidak ada file untuk dibersihkan",
                    'targets': None,
                    'total_files': 0,
                    'total_size': '0B'
                }
            
            self.logger.info(f"📋 Ditemukan {total_files} file untuk dibersihkan ({total_size})")
            return {
                'success': True,
                'message': f"Ditemukan {total_files} file untuk dibersihkan ({total_size})",
                'targets': targets_result,
                'total_files': total_files,
                'total_size': total_size
            }
            
        except Exception as e:
            return self._handle_error("Error saat mencari cleanup targets", e)

    def execute(self, targets_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the cleanup operation with comprehensive error handling.
        
        Args:
            targets_result: Optional pre-scanned targets result. If not provided, will scan for targets.
        
        Returns:
            Dict[str, Any]: Operation results containing:
                - success: Boolean indicating operation success
                - message: Summary message
                - deleted_files: Number of files deleted (if successful)
                - freed_space: Amount of space freed (if successful)
                - targets: Cleanup targets that were processed (if successful)
                - error: Error details (if failed)
        """
        try:
            self.log_operation_start("Cleanup Dataset")
            self._reset_progress_tracking()
            
            # Phase 1: Get cleanup targets if not provided
            if not targets_result:
                self.phase = DownloaderOperationPhase.INITIALIZING
                self.update_progress(0, "🔍 Mencari file untuk dibersihkan...")
                
                targets_response = self.get_cleanup_targets()
                if not targets_response['success']:
                    return targets_response
                
                targets_result = targets_response.get('targets')
                if not targets_result:
                    # No files to clean
                    self.phase = DownloaderOperationPhase.COMPLETED
                    self.update_progress(100, "✅ Tidak ada file untuk dibersihkan")
                    return {
                        'success': True,
                        'message': 'Tidak ada file untuk dibersihkan',
                        'deleted_files': 0,
                        'freed_space': '0B',
                        'operation_id': self.operation_id
                    }
            
            # Phase 2: Initialize cleanup service
            self.update_progress(10, "⚙️ Menginisialisasi layanan cleanup...")
            
            cleanup_service = self._create_cleanup_service()
            if not cleanup_service:
                return self._handle_error("Gagal membuat cleanup service")
            
            # Phase 3: Setup progress callback
            if hasattr(cleanup_service, 'set_progress_callback'):
                cleanup_service.set_progress_callback(self._handle_cleanup_progress)
                self.logger.debug("✅ Progress callback terdaftar untuk cleanup service")
            
            # Phase 4: Execute cleanup with backend progress tracking
            self.update_progress(20, "🗑️ Memulai pembersihan dataset...")
            
            try:
                cleanup_result = cleanup_service.cleanup_dataset_files(targets_result)
            except Exception as e:
                return self._handle_error("Cleanup gagal", e)
            
            # Phase 5: Process results
            if not cleanup_result or cleanup_result.get('status') != 'success':
                error_msg = cleanup_result.get('message', 'Cleanup gagal tanpa detail error') if cleanup_result else 'Cleanup gagal'
                return self._handle_error(error_msg)
            
            # Phase 6: Finalize and display results
            self.phase = DownloaderOperationPhase.COMPLETED
            self.update_progress(100, "✅ Pembersihan dataset selesai!")
            
            # Extract results
            deleted_files = cleanup_result.get('deleted_files', 0)
            freed_space = cleanup_result.get('freed_space', '0B')
            
            # Update UI summary
            self._update_summary_container(cleanup_result, targets_result)
            
            self.log_operation_complete("Cleanup Dataset")
            
            return {
                'success': True,
                'message': f'Pembersihan selesai: {deleted_files} file dihapus ({freed_space} dibebaskan)',
                'deleted_files': deleted_files,
                'freed_space': freed_space,
                'targets': targets_result,
                'operation_id': self.operation_id
            }
            
        except Exception as e:
            return self._handle_error("Error tidak terduga saat cleanup", e)

    def show_cleanup_confirmation(
        self, 
        targets_result: Dict[str, Any], 
        on_confirm: Callable[[], None]
    ) -> None:
        """
        Show cleanup confirmation dialog to user.
        
        Args:
            targets_result: Result from get_cleanup_targets
            on_confirm: Callback function when user confirms
        """
        try:
            summary = targets_result.get('summary', {})
            total_files = summary.get('total_files', 0)
            total_size = summary.get('total_size', '0B')
            
            message = (
                f"Akan menghapus {total_files:,} file ({total_size}).\n\n"
                "⚠️ Operasi ini tidak dapat dibatalkan!\n\n"
                "Lanjutkan dengan pembersihan?"
            )
            
            # Use operation container's confirmation dialog if available
            operation_container = getattr(self._ui_module, 'operation_container', None)
            if operation_container and hasattr(operation_container, 'show_confirmation_dialog'):
                operation_container.show_confirmation_dialog(
                    message=message,
                    callback=on_confirm,
                    title="⚠️ Konfirmasi Pembersihan Dataset",
                    confirm_text="🗑️ Bersihkan",
                    cancel_text="❌ Batal",
                    danger_mode=True
                )
            else:
                # Fallback to direct execution if no confirmation dialog
                self.logger.warning("Confirmation dialog tidak tersedia, melanjutkan cleanup langsung")
                on_confirm()
                
        except Exception as e:
            self.logger.error(f"Error showing cleanup confirmation: {e}", exc_info=True)
            # Fallback to direct execution on error
            on_confirm()

    def _create_cleanup_service(self):
        """
        Create cleanup service using backend API.
        
        Returns:
            Cleanup service instance or None if creation fails
        """
        try:
            cleanup_service_factory = self.get_backend_api('cleanup_service')
            if not cleanup_service_factory:
                self.logger.error("Cleanup service factory tidak tersedia dari backend API")
                return None
            
            return cleanup_service_factory()
            
        except Exception as e:
            self.logger.error(f"Error creating cleanup service: {e}", exc_info=True)
            return None

    def _update_summary_container(self, result: Dict[str, Any], targets_result: Dict[str, Any]) -> None:
        """
        Update summary container with cleanup results.
        
        Args:
            result: Result from cleanup operation
            targets_result: Result from get_cleanup_targets
        """
        try:
            summary_container = getattr(self._ui_module, 'summary_container', None)
            if not summary_container:
                self.logger.debug("Summary container tidak tersedia untuk update")
                return
                
            # Extract summary data
            summary = targets_result.get('summary', {})
            total_files = summary.get('total_files', 0)
            total_size = summary.get('total_size', '0B')
            targets = targets_result.get('targets', {})
            
            deleted_files = result.get('deleted_files', total_files)
            freed_space = result.get('freed_space', total_size)
            
            # Format summary content
            summary_lines = [
                "<div style='padding: 10px; background-color: #f0f8ff; border-radius: 5px; border-left: 5px solid #4682b4;'>",
                "<h3>🗑️ Ringkasan Pembersihan Dataset</h3>",
                f"<p>📁 Total file dihapus: <b>{deleted_files:,}</b></p>",
                f"<p>💾 Ruang yang dibebaskan: <b>{freed_space}</b></p>"
            ]
            
            # Add target details if available
            if targets:
                summary_lines.append("<h4>Detail pembersihan:</h4>")
                summary_lines.append("<ul>")
                for target_name, target_info in targets.items():
                    file_count = target_info.get('file_count', 0)
                    size_formatted = target_info.get('size_formatted', '0 B')
                    summary_lines.append(
                        f"<li><b>{target_name}</b>: {file_count:,} file ({size_formatted})</li>"
                    )
                summary_lines.append("</ul>")
            
            summary_lines.append("</div>")
            
            # Update summary container
            if hasattr(summary_container, 'set_content'):
                summary_container.set_content("".join(summary_lines))
            elif hasattr(summary_container, 'clear_output') and hasattr(summary_container, 'append_html'):
                summary_container.clear_output()
                summary_container.append_html("".join(summary_lines))
                
        except Exception as e:
            self.logger.error(f"Error updating summary container: {e}", exc_info=True)

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
            'deleted_files': 0,
            'freed_space': '0B',
            'error': error_details,
            'operation_id': self.operation_id
        }