"""
File: smartcash/ui/setup/dependency/handlers/status_check_handler.py

Status check and system report generation handler.

This module provides functionality for checking system status and generating
comprehensive system reports with detailed compatibility information.
"""

# Standard library imports
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, TypedDict, Union, Literal,
    Sequence, TypeVar, Generic, Mapping, MutableMapping
)

# Absolute imports
from smartcash.common.logger import get_logger
from smartcash.ui.setup.dependency.utils.ui.state import (
    update_status_panel,
    show_progress_tracker_safe,
    complete_operation_with_message,
    with_button_context,
    update_progress_step
)
from smartcash.ui.setup.dependency.utils.system.info import get_comprehensive_system_info
from smartcash.ui.setup.dependency.utils.reporting.generators import generate_system_compatibility_report
from smartcash.ui.setup.dependency.utils.ui.utils import get_selected_packages

# Type variables for generic typing
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# Type aliases for better code clarity and type safety
UIComponents = Dict[str, Any]
PackageName = str
PackageList = List[PackageName]

class StatusResult(TypedDict, total=False):
    """Result of a single package status check."""
    success: bool
    package: str
    installed: bool
    error: Optional[str]
    version: Optional[str]

class StatusResults(TypedDict):
    """Aggregated results of package status checks."""
    installed: List[PackageName]
    not_installed: List[PackageName]
    errors: List[Dict[str, str]]

class SystemInfo(TypedDict, total=False):
    """Structure for system information."""
    python_version: str
    platform: str
    architecture: str
    memory_info: Dict[str, Union[int, float]]
    gpu_info: Dict[str, Any]
    os_info: Dict[str, str]
    python_implementation: str

class CompatibilityReport(TypedDict, total=False):
    """Structure for system compatibility report."""
    warnings: List[str]
    recommendations: List[str]
    compatibility_score: float
    system_requirements: Dict[str, Any]
    missing_requirements: List[str]

class UIButtonState(TypedDict):
    """State management for UI buttons."""
    enabled: bool
    text: str
    variant: Literal['primary', 'secondary', 'success', 'danger', 'warning', 'info']

# Constants
DEFAULT_BATCH_SIZE = 10
MAX_BATCH_SIZE = 50
MIN_PROGRESS_UPDATE_INTERVAL = 0.1  # seconds

# Type aliases for UI components
UIComponent = Any
UIButton = UIComponent
UILabel = UIComponent
UIProgressBar = UIComponent

@dataclass
class StatusCheckConfig:
    """Configuration for status check operations."""
    batch_size: int = DEFAULT_BATCH_SIZE
    show_detailed_logs: bool = True
    enable_progress_updates: bool = True

class StatusCheckHandler:
    """Handler for status check and system report generation."""
    
    def __init__(self, ui_components: UIComponents, config: Optional[StatusCheckConfig] = None):
        """Initialize the status check handler.
        
        Args:
            ui_components: Dictionary of UI components
            config: Optional configuration for status checks
        """
        self.ui = ui_components
        self.logger = ui_components.get('logger', get_logger(__name__))
        self.config = config or StatusCheckConfig()
    
    def setup_handlers(self) -> Dict[str, Callable]:
        """Setup and return the handler functions.
        
        Returns:
            Dictionary mapping handler names to functions
        """
        # Setup button handlers
        if system_report_btn := self.ui.get('system_report_button'):
            system_report_btn.on_click(lambda _: self.handle_system_report())
        
        if check_btn := self.ui.get('check_button'):
            check_btn.on_click(lambda _: self.handle_package_status_check())
        
        return {
            'handle_system_report': self.handle_system_report,
            'handle_package_status_check': self.handle_package_status_check
        }
    
    @with_button_context(self.ui, 'system_report_button')
    def handle_system_report(self) -> None:
        """Generate and display a comprehensive system report."""
        try:
            self._update_status("🔍 Collecting system information...", "info")
            show_progress_tracker_safe(self.ui, "System Analysis")
            
            self.logger.info("🔍 Generating system compatibility report...")
            
            # Collect system information
            self._update_progress(20, "Collecting system info")
            system_info = get_comprehensive_system_info()
            
            # Generate compatibility report
            self._update_progress(50, "Checking compatibility")
            compatibility_report = generate_system_compatibility_report(system_info)
            
            # Log and finalize report
            self._update_progress(80, "Generating report")
            self._log_system_report(system_info, compatibility_report)
            
            # Show completion
            self._update_progress(100, "Report completed")
            summary = self._generate_report_summary(system_info, compatibility_report)
            complete_operation_with_message(self.ui, f"✅ {summary}")
            self.logger.info(f"📊 System report completed: {summary}")
            
        except Exception as e:
            self._handle_error("System report", e)
    
    @with_button_context(self.ui, 'check_button')
    def handle_package_status_check(self) -> None:
        """Check and display the status of selected packages."""
        try:
            selected_packages = get_selected_packages(self.ui.get('package_selector', {}))
            
            if not selected_packages:
                self._update_status("⚠️ No packages selected", "warning")
                return
            
            self._update_status(f"🔍 Checking status of {len(selected_packages)} packages...", "info")
            show_progress_tracker_safe(self.ui, "Package Status Check")
            self.logger.info(f"🔍 Checking status of {len(selected_packages)} packages...")
            
            try:
                # Check packages status
                results = self._check_packages_status_batch(selected_packages)
                
                # Update status with results
                summary = self._generate_status_summary(results)
                self._update_status(summary, "success" if not results.get('errors') else "warning")
                
                # Log detailed results
                self.logger.info(f"✅ Status check completed: {len(results.get('installed', []))} installed, "
                              f"{len(results.get('not_installed', []))} not installed, "
                              f"{len(results.get('errors', []))} errors")
                
                # Show detailed results in UI if enabled
                if self.config.show_detailed_logs:
                    self.logger.debug(f"Detailed package status: {results}")
                
            except Exception as e:
                self._handle_error("Package status check", e)
        
        except Exception as e:
            self._handle_error("Status check", e)
    
    def _update_status(self, message: str, status_type: str = "info") -> None:
        """Update the status panel with a message.
        
        Args:
            message: The message to display
            status_type: Type of status (info, warning, error)
        """
        update_status_panel(self.ui, message, status_type)
    
    def _update_progress(self, percent: int, message: str) -> None:
        """Update the progress indicator.
        
        Args:
            percent: Progress percentage (0-100)
            message: Status message
        """
        if self.config.enable_progress_updates:
            update_progress_step(self.ui, "overall", percent, message)
    
    def _handle_error(self, operation: str, error: Exception) -> None:
        """Handle and log an error.
        
        Args:
            operation: Name of the operation that failed
            error: The exception that was raised
        """
        error_msg = str(error)
        self._update_status(f"❌ {operation} error: {error_msg}", "error")
        self.logger.error(f"❌ {operation} failed: {error_msg}", exc_info=True)
    
    def _check_packages_status_batch(self, packages: PackageList) -> StatusResults:
        """Check the status of packages in batches.
        
        Args:
            packages: List of package names to check
            
        Returns:
            Dictionary containing lists of installed, not_installed, and errors
        """
        from smartcash.ui.setup.dependency.utils.package.installer import batch_check_packages_status
        
        results: StatusResults = {
            'installed': [],
            'not_installed': [],
            'errors': []
        }
        
        batch_size = self.config.batch_size
        total_batches = (len(packages) + batch_size - 1) // batch_size
        
        for i in range(0, len(packages), batch_size):
            batch = packages[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            progress = int((i / len(packages)) * 100)
            
            self._update_progress(progress, f"Checking batch {batch_num}/{total_batches}")
            
            try:
                batch_results = batch_check_packages_status(batch)
                
                for result in batch_results:
                    if result.get('success', False):
                        if result.get('installed', False):
                            results['installed'].append(result['package'])
                        else:
                            results['not_installed'].append(result['package'])
                    else:
                        results['errors'].append({
                            'package': result.get('package', 'unknown'),
                            'error': result.get('error', 'Unknown error')
                        })
                        
            except Exception as e:
                self.logger.error(f"Error checking package batch {batch_num}: {str(e)}")
                for pkg in batch:
                    results['errors'].append({
                        'package': pkg,
                        'error': f"Batch processing error: {str(e)}"
                    })
        
        return results
    
    def _log_system_report(self, system_info: SystemInfo, compatibility_report: CompatibilityReport) -> None:
        """Log detailed system information and compatibility report.
        
        Args:
            system_info: Dictionary containing system information
            compatibility_report: Dictionary containing compatibility information
        """
        try:
            self._log_system_info(system_info)
            self._log_compatibility_issues(compatibility_report)
            self._log_recommendations(compatibility_report)
        except Exception as e:
            self.logger.error(f"Error logging system report: {str(e)}", exc_info=True)
    
    def _log_system_info(self, system_info: SystemInfo) -> None:
        """Log basic system information.
        
        Args:
            system_info: Dictionary containing system information
        """
        self.logger.info("💻 System Information:")
        self.logger.info(f"   • Python: {system_info.get('python_version', 'Unknown')}")
        self.logger.info(f"   • Platform: {system_info.get('platform', 'Unknown')}")
        self.logger.info(f"   • Architecture: {system_info.get('architecture', 'Unknown')}")
        
        # Log memory information if available
        if mem_info := system_info.get('memory_info', {}):
            if 'available_gb' in mem_info:
                self.logger.info(f"   • Memory: {mem_info['available_gb']:.1f} GB available")
        
        # Log GPU/CUDA information
        self._log_gpu_info(system_info.get('gpu_info', {}))
    
    def _log_gpu_info(self, gpu_info: Dict[str, Any]) -> None:
        """Log GPU and CUDA information.
        
        Args:
            gpu_info: Dictionary containing GPU information
        """
        if gpu_info.get('cuda_available', False):
            self.logger.info(
                f"   • CUDA: Available (version {gpu_info.get('cuda_version', 'Unknown')})"
            )
        else:
            self.logger.info("   • CUDA: Not available")
    
    def _log_compatibility_issues(self, report: CompatibilityReport) -> None:
        """Log compatibility warnings.
        
        Args:
            report: Dictionary containing compatibility information
        """
        if warnings := report.get('warnings', [])[:3]:  # Limit to first 3 warnings
            self.logger.warning("⚠️ Compatibility Warnings:")
            for warning in warnings:
                self.logger.warning(f"   • {warning}")
    
    def _log_recommendations(self, report: CompatibilityReport) -> None:
        """Log system recommendations.
        
        Args:
            report: Dictionary containing recommendations
        """
        if recommendations := report.get('recommendations', [])[:3]:  # Limit to first 3
            self.logger.info("💡 Recommendations:")
            for rec in recommendations:
                self.logger.info(f"   • {rec}")
    
    def _generate_report_summary(self, system_info: SystemInfo, 
                              compatibility_report: CompatibilityReport) -> str:
        """Generate a summary of the system report.
        
        Args:
            system_info: Dictionary containing system information
            compatibility_report: Dictionary containing compatibility information
            
        Returns:
            String summary of the system report
        """
        try:
            python_version = system_info.get('python_version', 'Unknown')
            warnings_count = len(compatibility_report.get('warnings', []))
            
            return (
                f"System compatible (Python {python_version})" 
                if warnings_count == 0
                else f"System report complete, {warnings_count} warnings found"
            )
        except Exception as e:
            self.logger.error(f"Error generating report summary: {str(e)}")
            return "System report generated"
    
    def _generate_status_summary(self, results: StatusResults) -> str:
        """Generate a summary of package status check results.
        
        Args:
            results: Dictionary containing package status results
            
        Returns:
            String summary of the status check
        """
        try:
            installed = len(results.get('installed', []))
            not_installed = len(results.get('not_installed', []))
            errors = len(results.get('errors', []))
            total = installed + not_installed + errors
            
            if total == 0:
                return "No packages checked"
                
            return f"{installed}/{total} installed, {errors} errors"
            
        except Exception as e:
            self.logger.error(f"Error generating status summary: {str(e)}")
            return "Status check completed"


def setup_status_check_handler(ui_components: UIComponents) -> Dict[str, Callable]:
    """Setup and return status check handlers.
    
    This function initializes the StatusCheckHandler with the provided UI components
    and returns a dictionary of handler functions that can be connected to UI events.
    
    Args:
        ui_components: Dictionary containing UI components needed by the handlers
        
    Returns:
        Dictionary mapping handler names to their corresponding functions
    """
    handler = StatusCheckHandler(ui_components)
    return handler.setup_handlers()