"""
File: smartcash/ui/dataset/downloader/operations/download_check_operation.py
Description: Dataset check operation implementation for the downloader module.
"""

from typing import Dict, Any, Optional, Callable, TYPE_CHECKING

from .downloader_base_operation import DownloaderBaseOperation, DownloaderOperationPhase

if TYPE_CHECKING:
    from smartcash.ui.dataset.downloader.downloader_uimodule import DownloaderUIModule


class DownloadCheckOperation(DownloaderBaseOperation):
    """
    Dataset check operation that handles dataset validation and integrity checking.
    
    This class manages the execution of dataset scans with:
    - Progress tracking for scan operations
    - Dataset integrity validation
    - Comprehensive file and annotation checking
    - UI integration for results display
    """

    # Check phases with their weights (must sum to 100)
    CHECK_PHASES = {
        'init': {'weight': 10, 'label': '⚙️ Inisialisasi', 'phase': DownloaderOperationPhase.INITIALIZING},
        'scan_start': {'weight': 5, 'label': '🔍 Memulai scan', 'phase': DownloaderOperationPhase.PROCESSING},
        'scan_structure': {'weight': 15, 'label': '📁 Scan struktur', 'phase': DownloaderOperationPhase.PROCESSING},
        'scan_downloads': {'weight': 20, 'label': '📂 Scan downloads', 'phase': DownloaderOperationPhase.PROCESSING},
        'scan_splits': {'weight': 30, 'label': '📊 Scan splits', 'phase': DownloaderOperationPhase.PROCESSING},
        'scan_aggregate': {'weight': 15, 'label': '📈 Agregasi data', 'phase': DownloaderOperationPhase.PROCESSING},
        'scan_complete': {'weight': 5, 'label': '✅ Selesai', 'phase': DownloaderOperationPhase.FINALIZING}
    }

    def __init__(
        self,
        ui_module: 'DownloaderUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> None:
        """
        Initialize the check operation.
        
        Args:
            ui_module: Reference to the parent UI module
            config: Configuration dictionary for the check
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
        self._phase_order = list(self.CHECK_PHASES.keys())
        
        # Validate phase weights sum to 100
        total_weight = sum(phase['weight'] for phase in self.CHECK_PHASES.values())
        if total_weight != 100:
            self.logger.warning(f"Check phase weights sum to {total_weight}, not 100. Progress may be inaccurate.")

    def _register_backend_callbacks(self) -> None:
        """Register callbacks for backend progress updates."""
        if hasattr(self._ui_module, 'register_progress_callback'):
            self._ui_module.register_progress_callback('check', self._handle_check_progress)

    def _handle_check_progress(self, step: str, current: int, total: int = 100, message: str = "") -> None:
        """
        Handle check progress updates from the backend with dual progress tracking.
        
        Maps backend progress callbacks to dual progress tracker:
        - Main bar: Overall progress across all check phases (n/total_phases)
        - Secondary bar: Current progress within the active phase (0-100%)
        
        Args:
            step: Current check phase/step name
            current: Current progress within the phase (0-100)
            total: Total progress value (typically 100, kept for compatibility)
            message: Optional progress message from backend
        """
        try:
            # Normalize step name
            step_name = step.lower().strip()
            
            # Get phase configuration
            phase_config = self.CHECK_PHASES.get(step_name)
            if not phase_config:
                # Handle unknown phases gracefully
                self.logger.warning(f"Unknown check phase: {step_name}")
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
                self.CHECK_PHASES[phase]['weight'] 
                for phase in self._completed_phases 
                if phase in self.CHECK_PHASES
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
                    f"📊 Check Progress: {main_message} | "
                    f"Overall: {self._overall_progress:.1f}% | "
                    f"Phase: {current}%"
                )
                
        except Exception as e:
            self.logger.error(f"Error handling check progress: {e}", exc_info=True)

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

    def execute(self) -> Dict[str, Any]:
        """
        Execute the check operation with comprehensive error handling.
        
        Returns:
            Dict[str, Any]: Operation results containing:
                - success: Boolean indicating operation success
                - message: Summary message
                - exists: Boolean indicating if dataset exists
                - file_count: Number of files found (if successful)
                - total_size: Total size of dataset (if successful)
                - dataset_path: Path to dataset (if found)
                - summary: Detailed scan summary (if successful)
                - issues: List of detected issues (if any)
                - error: Error details (if failed)
        """
        try:
            self.log_operation_start("Check Dataset")
            self._reset_progress_tracking()
            
            # Phase 1: Initialize scanner
            self.phase = DownloaderOperationPhase.INITIALIZING
            self.update_progress(0, "🔧 Menginisialisasi scanner...")
            
            scanner = self._create_dataset_scanner()
            if not scanner:
                return self._handle_error("Gagal membuat dataset scanner")
            
            # Phase 2: Setup progress callback
            if hasattr(scanner, 'set_progress_callback'):
                scanner.set_progress_callback(self._handle_check_progress)
                self.logger.debug("✅ Progress callback terdaftar untuk scanner")
            
            # Phase 3: Execute dataset scan
            self.update_progress(10, "🔍 Memulai pemeriksaan dataset...")
            
            try:
                scan_result = scanner.scan_existing_dataset_parallel()
            except Exception as e:
                return self._handle_error("Scan dataset gagal", e)
            
            # Phase 4: Process results
            if not scan_result or scan_result.get('status') != 'success':
                error_msg = scan_result.get('message', 'Scan gagal tanpa detail error') if scan_result else 'Scan gagal'
                return self._handle_error(error_msg)
            
            # Phase 5: Generate summary and display results
            self.phase = DownloaderOperationPhase.COMPLETED
            self.update_progress(100, "✅ Pemeriksaan dataset selesai!")
            
            # Extract results
            summary = scan_result.get('summary', {})
            stats = scan_result.get('stats', {})
            issues = scan_result.get('issues', [])
            
            # Display results in UI
            self._display_check_results(scan_result)
            self._update_summary_container(scan_result)
            
            file_count = summary.get('total_images', 0) + summary.get('total_labels', 0)
            total_size = scan_result.get('total_size', '0B')
            dataset_path = scan_result.get('dataset_path', '')
            
            self.log_operation_complete("Check Dataset")
            
            # Create result dictionary
            result = {
                'success': True,
                'message': f'Dataset ditemukan: {file_count} file ({total_size})',
                'exists': True,
                'file_count': file_count,
                'total_size': total_size,
                'dataset_path': dataset_path,
                'summary': summary,
                'stats': stats,
                'issues': issues,
                'operation_id': self.operation_id
            }
            
            # Execute summary callback with formatted summary
            try:
                summary_html = self._format_check_summary(result)
                self._execute_callback('on_success', summary_html)
            except Exception as e:
                self.logger.warning(f"Failed to update check summary: {e}")
            
            return result
            
        except Exception as e:
            return self._handle_error("Error tidak terduga saat check", e)

    def _format_check_summary(self, result: Dict[str, Any]) -> str:
        """Format check operation result into HTML summary."""
        file_count = result.get('file_count', 0)
        total_size = result.get('total_size', '0B')
        dataset_path = result.get('dataset_path', 'N/A')
        summary = result.get('summary', {})
        issues = result.get('issues', [])
        
        # Format summary stats
        total_images = summary.get('total_images', 0)
        total_labels = summary.get('total_labels', 0)
        
        issues_text = f"⚠️ {len(issues)} issues found" if issues else "✅ No issues"
        
        return f"""
### Ringkasan Operasi Check Dataset

| Kategori | Detail |
| :--- | :--- |
| **Status** | ✅ Dataset Found |
| **Total Files** | 📁 {file_count} file |
| **Images** | 🖼️ {total_images} |
| **Labels** | 🏷️ {total_labels} |
| **Total Size** | 💾 {total_size} |
| **Dataset Path** | 📂 {dataset_path} |
| **Issues** | {issues_text} |

---

**Dataset check completed successfully!**
"""

    def _create_dataset_scanner(self):
        """
        Create dataset scanner service using backend API.
        
        Returns:
            Dataset scanner instance or None if creation fails
        """
        try:
            scanner_factory = self.get_backend_api('scanner')
            if not scanner_factory:
                self.logger.error("Scanner factory tidak tersedia dari backend API")
                return None
            
            return scanner_factory(self.logger)
            
        except Exception as e:
            self.logger.error(f"Error creating dataset scanner: {e}", exc_info=True)
            return None

    def _display_check_results(self, result: Dict[str, Any]) -> None:
        """
        Display check results in the UI summary container.
        
        Args:
            result: Dictionary containing scan results
        """
        try:
            summary_container = getattr(self._ui_module, 'summary_container', None)
            if not summary_container:
                self.logger.debug("Summary container tidak tersedia")
                return
                
            summary = result.get('summary', {})
            splits = result.get('splits', {})
            
            # Format HTML content
            html_lines = [
                "<div style='padding: 10px; background-color: #f0f8ff; border-radius: 5px; border-left: 5px solid #4682b4;'>",
                "<h3>📊 Ringkasan Dataset</h3>",
                f"<p>📂 Path: {result.get('dataset_path', 'N/A')}</p>",
                f"<p>🖼️ Total Gambar: <b>{summary.get('total_images', 0):,}</b></p>",
                f"<p>🏷️ Total Label: <b>{summary.get('total_labels', 0):,}</b></p>"
            ]
            
            # Splits detail
            if splits:
                html_lines.append("<h4>Detail per Split:</h4>")
                html_lines.append("<ul>")
                for split_name, split_data in splits.items():
                    if split_data.get('status') == 'success':
                        img_count = split_data.get('images', 0)
                        label_count = split_data.get('labels', 0)
                        size_formatted = split_data.get('size_formatted', '0 B')
                        html_lines.append(
                            f"<li><b>{split_name}</b>: {img_count:,} gambar, "
                            f"{label_count:,} label ({size_formatted})</li>"
                        )
                html_lines.append("</ul>")
            
            html_lines.append("</div>")
            
            # Update summary container
            if hasattr(summary_container, 'set_content'):
                summary_container.set_content("".join(html_lines))
            elif hasattr(summary_container, 'clear_output') and hasattr(summary_container, 'append_html'):
                summary_container.clear_output()
                summary_container.append_html("".join(html_lines))
            
        except Exception as e:
            self.logger.error(f"Error displaying check results: {e}", exc_info=True)

    def _update_summary_container(self, result: Dict[str, Any]) -> None:
        """
        Update summary container with check results.
        
        Args:
            result: Result from check operation
        """
        try:
            summary_container = getattr(self._ui_module, 'summary_container', None)
            if not summary_container:
                self.logger.debug("Summary container tidak tersedia untuk update")
                return
                
            # Extract summary data
            stats = result.get('stats', {})
            total_images = stats.get('total_images', 0)
            total_annotations = stats.get('total_annotations', 0)
            issues = result.get('issues', [])
            
            # Format summary content
            summary_lines = [
                "<div style='padding: 10px; background-color: #f0f8ff; border-radius: 5px; border-left: 5px solid #4682b4;'>",
                "<h3>📊 Ringkasan Pemeriksaan Dataset</h3>",
                f"<p>🖼️ Total gambar: <b>{total_images:,}</b></p>",
                f"<p>🏷️ Total anotasi: <b>{total_annotations:,}</b></p>"
            ]
            
            # Add class distribution if available
            class_distribution = stats.get('class_distribution', {})
            if class_distribution:
                summary_lines.append("<p>🔍 Distribusi kelas:</p>")
                summary_lines.append("<ul>")
                for class_name, count in class_distribution.items():
                    summary_lines.append(f"<li>{class_name}: {count:,}</li>")
                summary_lines.append("</ul>")
            
            # Add issues if any
            if issues:
                issue_count = len(issues)
                summary_lines.append(f"<p>⚠️ <b>{issue_count}</b> masalah terdeteksi:</p>")
                summary_lines.append("<ul>")
                for issue in issues[:5]:  # Show only top 5 issues in summary
                    issue_type = issue.get('type', 'Unknown')
                    issue_count = issue.get('count', 1)
                    summary_lines.append(f"<li>{issue_type}: {issue_count} item</li>")
                
                if len(issues) > 5:
                    summary_lines.append(f"<li>... dan {len(issues) - 5} masalah lainnya</li>")
                
                summary_lines.append("</ul>")
            else:
                summary_lines.append("<p>✅ Tidak ada masalah terdeteksi</p>")
            
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
            'exists': False,
            'file_count': 0,
            'total_size': '0B',
            'error': error_details,
            'operation_id': self.operation_id
        }