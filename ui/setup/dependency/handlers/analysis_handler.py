"""
File: smartcash/ui/setup/dependency/handlers/analysis_handler.py

Package analysis and compatibility checking handler.

This module provides functionality for analyzing Python packages and checking
for compatibility issues in a batch processing manner.
"""

# Standard library imports
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, TypedDict

# Absolute imports
from smartcash.common.logger import get_logger
from smartcash.ui.setup.dependency.utils.package.installer import (
    check_package_installation_status,
    get_installed_packages_dict
)
from smartcash.ui.setup.dependency.utils.ui.state import with_button_context
# Package status utilities will be imported here if needed in the future
from smartcash.ui.setup.dependency.utils.ui.components.buttons import with_button_state
from smartcash.ui.setup.dependency.utils.ui.state import (
    complete_operation_with_message,
    show_progress_tracker_safe,
    update_progress_step,
    update_status_panel
)
from smartcash.ui.setup.dependency.utils.ui.utils import get_selected_packages

# Type aliases
UIComponents = Dict[str, Any]
PackageList = List[str]
AnalysisResults = Dict[str, Any]
AnalysisReport = Dict[str, Any]

# Protocol for UI Components that have a 'value' attribute
class HasValue(Protocol):
    value: str

# Constants
DEFAULT_BATCH_SIZE = 10
DEFAULT_ANALYSIS_CONFIG = {
    'check_compatibility': True,
    'batch_size': DEFAULT_BATCH_SIZE,
    'include_dev_deps': False,
    'detailed_info': True
}

@dataclass
class AnalysisConfig:
    """Configuration for package analysis."""
    check_compatibility: bool = True
    batch_size: int = DEFAULT_BATCH_SIZE
    include_dev_deps: bool = False
    detailed_info: bool = True

class AnalysisHandler:
    """Handles package analysis operations with progress tracking."""
    
    def __init__(self, ui_components: UIComponents):
        """Initialize with UI components."""
        self.ui = ui_components
        self.logger = get_logger(__name__)
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup UI event handlers."""
        if button := self.ui.get('analyze_button'):
            button.on_click(lambda _: self.handle_analysis())
    
    @with_button_context('analyze_button')
    def handle_analysis(self) -> None:
        """Handle package analysis with progress tracking."""
        try:
            self._perform_analysis()
        except Exception as e:
            self._handle_analysis_error(e)
    
    def _perform_analysis(self) -> None:
        """Execute the analysis workflow."""
        packages = self._get_packages_for_analysis()
        if not packages:
            update_status_panel(self.ui, "⚠️ No packages to analyze", "warning")
            return
        
        config = self._get_analysis_config()
        self._start_analysis_ui(len(packages))
        
        analysis_results = self._analyze_packages_batch(packages, config)
        report = self._generate_analysis_report(analysis_results)
        
        self._update_analysis_results(report)
        self._complete_analysis(report)
    
    def _get_packages_for_analysis(self) -> PackageList:
        """Retrieve and combine selected and custom packages."""
        selected = get_selected_packages(self.ui.get('package_selector', {}))
        custom = self._get_custom_packages()
        return selected + custom
    
    def _get_custom_packages(self) -> PackageList:
        """Extract custom packages from the custom packages textarea."""
        try:
            if widget := self.ui.get('custom_packages'):
                if hasattr(widget, 'value') and widget.value.strip():
                    return [pkg.strip() for pkg in widget.value.strip().split('\n') if pkg.strip()]
        except Exception as e:
            self.logger.warning(f"Error reading custom packages: {e}")
        return []
    
    def _get_analysis_config(self) -> AnalysisConfig:
        """Extract and validate analysis configuration."""
        config_data = self._extract_analysis_config()
        return AnalysisConfig(**config_data)
    
    def _extract_analysis_config(self) -> Dict[str, Any]:
        """Extract analysis configuration from UI."""
        try:
            from ..handlers.config_extractor import extract_dependency_config
            full_config = extract_dependency_config(self.ui)
            return full_config.get('analysis', DEFAULT_ANALYSIS_CONFIG)
        except Exception as e:
            self.logger.warning(f"Error extracting analysis config: {e}")
            return DEFAULT_ANALYSIS_CONFIG
    
    def _start_analysis_ui(self, package_count: int) -> None:
        """Update UI at the start of analysis."""
        update_status_panel(self.ui, f"🔍 Analyzing {package_count} packages...", "info")
        show_progress_tracker_safe(self.ui, "Package Analysis")
        self.logger.info(f"Analyzing {package_count} packages...")
    
    def _analyze_packages_batch(self, packages: PackageList, config: AnalysisConfig) -> AnalysisResults:
        """Analyze packages in batches with progress tracking."""
        results: AnalysisResults = {
            'total': len(packages),
            'succeeded': 0,
            'failed': 0,
            'compatibility_issues': [],
            'details': []
        }
        
        # Get installed packages once for the entire batch
        installed_packages = get_installed_packages_dict()
        
        batch_size = min(config.batch_size, len(packages))
        for i in range(0, len(packages), batch_size):
            batch = packages[i:i + batch_size]
            batch_results = self._check_compatibility_batch(batch, installed_packages)
            
            # Update results
            results['succeeded'] += batch_results.get('succeeded', 0)
            results['failed'] += batch_results.get('failed', 0)
            results['compatibility_issues'].extend(batch_results.get('compatibility_issues', []))
            results['details'].extend(batch_results.get('details', []))
            
            # Update progress
            progress = min(100, int((i + len(batch)) / len(packages) * 100))
            update_progress_step(self.ui, f"Processed {i + len(batch)}/{len(packages)} packages")
            
        return results
    
    def _check_compatibility_batch(self, packages: PackageList, installed_packages: Dict[str, str]) -> Dict[str, Any]:
        """Check compatibility for a batch of packages."""
        results = {
            'succeeded': 0,
            'failed': 0,
            'compatibility_issues': [],
            'details': []
        }
        
        try:
            # Process packages in the current batch
            for pkg in packages:
                try:
                    # Check installation status
                    status = check_package_installation_status(pkg, installed_packages=installed_packages)
                    
                    # Add to results
                    if status.get('success', False):
                        results['succeeded'] += 1
                        results['details'].append({
                            'package': pkg,
                            'status': 'installed' if status.get('installed', False) else 'not_installed',
                            'version': status.get('version', 'unknown'),
                            'compatible': status.get('compatible', False)
                        })
                        
                        # Check for compatibility issues
                        if not status.get('compatible', True):
                            results['compatibility_issues'].append({
                                'package': pkg,
                                'issue': f"Version {status.get('version', 'unknown')} is not compatible with the required version"
                            })
                    else:
                        results['failed'] += 1
                        results['details'].append({
                            'package': pkg,
                            'status': 'error',
                            'error': status.get('error', 'Unknown error')
                        })
                        
                except Exception as e:
                    results['failed'] += 1
                    self.logger.warning(f"Failed to analyze package {pkg}: {str(e)}")
                    results['details'].append({
                        'package': pkg,
                        'status': 'error',
                        'error': str(e)
                    })
                        
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            results['failed'] = len(packages)
            
        return results
    
    def _generate_analysis_report(self, results: AnalysisResults) -> AnalysisReport:
        """Generate a comprehensive analysis report."""
        total = results.get('total', 0)
        succeeded = results.get('succeeded', 0)
        failed = results.get('failed', 0)
        issues = len(results.get('compatibility_issues', []))
        
        summary = f"{succeeded}/{total} packages analyzed successfully"
        if failed > 0:
            summary += f", {failed} failed"
        if issues > 0:
            summary += f", {issues} compatibility issues found"
            
        return {
            'summary': summary,
            'total_packages': total,
            'succeeded': succeeded,
            'failed': failed,
            'compatibility_issues': results.get('compatibility_issues', []),
            'details': results.get('details', [])
        }
    
    def _update_analysis_results(self, report: AnalysisReport) -> None:
        """Update UI with analysis results."""
        try:
            if results_panel := self.ui.get('results_panel'):
                # Clear previous results
                results_panel.clear_output()
                
                # Display summary
                with results_panel:
                    print("📊 Analysis Results")
                    print("=" * 40)
                    print(f"✅ {report['succeeded']}/{report['total_packages']} packages analyzed successfully")
                    
                    if report['failed'] > 0:
                        print(f"❌ {report['failed']} packages failed analysis")
                        
                    if issues := report['compatibility_issues']:
                        print("\n⚠️  Compatibility Issues:")
                        for issue in issues[:5]:  # Show first 5 issues
                            print(f"- {issue}")
                        if len(issues) > 5:
                            print(f"... and {len(issues) - 5} more")
        except Exception as e:
            self.logger.error(f"Error updating results UI: {str(e)}")
    
    def _complete_analysis(self, report: AnalysisReport) -> None:
        """Finalize analysis with success message."""
        complete_operation_with_message(self.ui, f"✅ Analysis complete: {report['summary']}")
        self.logger.info(f"Analysis completed: {report['summary']}")
    
    def _handle_analysis_error(self, error: Exception) -> None:
        """Handle analysis errors consistently."""
        error_msg = str(error)
        update_status_panel(self.ui, f"❌ Analysis error: {error_msg}", "error")
        self.logger.error(f"Analysis failed: {error_msg}", exc_info=True)

def setup_analysis_handler(ui_components: UIComponents) -> Dict[str, Callable]:
    """Initialize and return analysis handler with bound methods.
    
    Args:
        ui_components: Dictionary of UI components
        
    Returns:
        Dictionary of handler functions bound to the UI components
    """
    handler = AnalysisHandler(ui_components)
    return {
        'handle_analysis': handler.handle_analysis,
        'analyze_packages_batch': handler._analyze_packages_batch
    }