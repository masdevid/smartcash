"""
File: smartcash/ui/setup/dependency/handlers/analysis_handler_refactored.py
Deskripsi: Optimized analysis handler dengan logger reference dan code quality yang lebih baik
"""
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum, auto

from smartcash.ui.setup.dependency.utils import (
    ProgressSteps, batch_check_packages_status, create_operation_context,
    extract_package_name_from_requirement, generate_analysis_summary_report,
    get_installed_packages_dict, log_to_ui_safe, update_package_status,
    update_status_panel, get_package_categories
)
from ..utils.ui_deps import requires, get_optional

class AnalysisStatus(Enum):
    """Status analisis package"""
    INSTALLED = auto()
    MISSING = auto()
    UPGRADE_NEEDED = auto()
    ERROR = auto()
    CHECKING = auto()

@dataclass
class PackageInfo:
    """Informasi package yang dianalisis"""
    name: str
    pip_name: str
    category: str
    package_name: str
    required_version: str
    status: str
    installed_version: str
    compatible: bool

@dataclass
class AnalysisResult:
    """Hasil analisis dependensi"""
    installed: List[str] = None
    missing: List[str] = None
    upgrade_needed: List[str] = None
    package_details: Dict[str, Dict] = None
    
    def __post_init__(self):
        self.installed = self.installed or []
        self.missing = self.missing or []
        self.upgrade_needed = self.upgrade_needed or []
        self.package_details = self.package_details or {}
    
    @property
    def summary(self) -> Dict[str, int]:
        """Ringkasan hasil analisis"""
        return {
            'installed': len(self.installed),
            'missing': len(self.missing),
            'upgrade_needed': len(self.upgrade_needed),
            'total': len(self.installed) + len(self.missing) + len(self.upgrade_needed)
        }

def setup_analysis_handler(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup analysis handler dengan fixed logger reference
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi untuk handler
    """
    def execute_analysis(button=None):
        """Execute package analysis dengan operation context"""
        with create_operation_context(ui_components, 'analysis') as ctx:
            _execute_analysis(ui_components, config, ctx)
    
    if 'analyze_button' in ui_components:
        ui_components['analyze_button'].on_click(execute_analysis)
    ui_components['trigger_analysis'] = execute_analysis

def _update_progress(
    progress_tracker: Any,
    level: str,
    value: int,
    message: str,
    logger: Optional[Any] = None
) -> None:
    """Update progress dengan error handling yang aman"""
    if not progress_tracker:
        return
        
    try:
        updater = getattr(progress_tracker, f'update_{level}', None)
        if callable(updater):
            updater(value, message)
    except Exception as e:
        if logger:
            logger.debug(f"🔄 Gagal update progress {level}: {str(e)}")

@requires('progress_tracker')
def _execute_analysis(ui_components: Dict[str, Any], config: Dict[str, Any], ctx) -> None:
    """Execute analysis dengan validasi komponen otomatis"""
    progress_tracker = ui_components['progress_tracker']
    logger = ui_components.get('logger')
    
    try:
        # Inisialisasi progress tracker
        _init_progress_tracker(progress_tracker, ui_components, logger)
        
        # Dapatkan daftar package yang terinstall
        installed_packages = _get_installed_packages(progress_tracker, ui_components, logger)
        if not installed_packages:
            return
            
        # Analisis package
        package_categories = _prepare_analysis(progress_tracker, ui_components, logger)
        analysis_results = _analyze_packages(
            package_categories, installed_packages, ui_components, logger
        )
        
        # Finalisasi hasil
        _finalize_analysis(progress_tracker, ui_components, analysis_results, logger)
        
    except Exception as e:
        _handle_analysis_error(e, progress_tracker, ui_components, logger)
        raise

def _init_progress_tracker(
    progress_tracker: Any, 
    ui_components: Dict[str, Any], 
    logger: Optional[Any]
) -> None:
    """Inisialisasi progress tracker"""
    try:
        if not hasattr(progress_tracker, 'show'):
            log_to_ui_safe(ui_components, "⚠️ Progress tracker tidak didukung", "warning")
            return
            
        progress_tracker.show(
            operation="Analisis Dependensi",
            steps=["🔍 Analisis", "📊 Evaluasi"],
            level='dual'
        )
        _update_progress(progress_tracker, 'overall', 10, "🚀 Memulai analisis...", logger)
        _update_progress(progress_tracker, 'current', 25, "🔧 Mempersiapkan...", logger)
        
    except Exception as e:
        error_msg = f"⚠️ Gagal inisialisasi progress tracker: {str(e)}"
        log_to_ui_safe(ui_components, error_msg, "warning")
        if logger:
            logger.error(error_msg, exc_info=True)

def _get_installed_packages(
    progress_tracker: Any, 
    ui_components: Dict[str, Any], 
    logger: Optional[Any]
) -> Dict[str, str]:
    """Dapatkan daftar package yang terinstall"""
    _update_progress(progress_tracker, 'overall', 30, "🔍 Scanning packages...", logger)
    _update_progress(progress_tracker, 'current', 50, "📜 Mendapatkan daftar packages...", logger)
    
    installed_packages = get_installed_packages_dict()
    log_to_ui_safe(ui_components, f"📦 Ditemukan {len(installed_packages)} packages terinstall")
    return installed_packages

def _prepare_analysis(
    progress_tracker: Any, 
    ui_components: Dict[str, Any], 
    logger: Optional[Any]
) -> List[Dict]:
    """Persiapan analisis package"""
    _update_progress(progress_tracker, 'overall', 50, "📊 Evaluasi packages...", logger)
    _update_progress(progress_tracker, 'current', 75, "📁 Menganalisis categories...", logger)
    
    package_categories = get_package_categories()
    _reset_all_package_statuses(ui_components, package_categories)
    return package_categories

def _analyze_packages(
    package_categories: List[Dict], 
    installed_packages: Dict[str, str],
    ui_components: Dict[str, Any],
    logger: Optional[Any]
) -> Dict[str, Any]:
    """Analisis package dengan progress tracking"""
    progress_tracker = ui_components.get('progress_tracker')
    analysis_results = AnalysisResult()
    all_packages = [(p, c['name']) for c in package_categories for p in c['packages']]
    total_packages = len(all_packages)
    
    # Proses packages dalam batch
    package_requirements = [pkg['pip_name'] for pkg, _ in all_packages]
    batch_status = batch_check_packages_status(package_requirements)
    
    for idx, ((package, category_name), requirement) in enumerate(zip(all_packages, package_requirements), 1):
        _update_package_progress(
            progress_tracker, package, requirement, idx, total_packages, 
            installed_packages, logger
        )
        
        status_info = batch_status[requirement]
        _process_package_status(
            package, category_name, requirement, status_info, 
            analysis_results, ui_components, logger
        )
    
    return analysis_results.__dict__

def _update_package_progress(
    progress_tracker: Any,
    package: Dict,
    requirement: str,
    current: int,
    total: int,
    installed_packages: Dict[str, str],
    logger: Optional[Any]
) -> None:
    """Update progress untuk package yang sedang diproses"""
    progress = int((current / total) * 100)
    pkg_name = extract_package_name_from_requirement(requirement)
    status_emoji = (
        "✅" if requirement in installed_packages 
        else "⚠️" if pkg_name in installed_packages 
        else "❌"
    )
    
    # Update progress tracker
    _update_progress(
        progress_tracker,
        'overall',
        70 + int(progress * 0.3),  # 70-100% range untuk analisis
        f"🔄 Memeriksa {current}/{total}",
        logger
    )
    _update_progress(
        progress_tracker,
        'current',
        progress,
        f"{status_emoji} Menganalisis {package['name']}...",
        logger
    )

def _process_package_status(
    package: Dict,
    category_name: str,
    requirement: str,
    status_info: Dict,
    analysis_results: AnalysisResult,
    ui_components: Dict[str, Any],
    logger: Optional[Any]
) -> None:
    """Proses status package dan update UI"""
    package_key = package['key']
    
    # Simpan detail package
    package_info = PackageInfo(
        name=package['name'],
        pip_name=requirement,
        category=category_name,
        package_name=extract_package_name_from_requirement(requirement),
        required_version=status_info.get('required_version', ''),
        status=status_info['status'],
        installed_version=status_info.get('version'),
        compatible=status_info.get('compatible', False)
    )
    
    # Simpan hasil analisis
    analysis_results.package_details[package_key] = package_info.__dict__
    
    # Kategorikan package
    if status_info['status'] in ['installed', 'missing', 'upgrade']:
        status_list = getattr(analysis_results, {
            'installed': 'installed',
            'missing': 'missing',
            'upgrade': 'upgrade_needed'
        }[status_info['status']])
        status_list.append(package_key)
    
    # Update UI
    ui_status = {
        'installed': 'installed',
        'missing': 'missing',
        'upgrade': 'upgrade',
        'error': 'error'
    }.get(status_info['status'], 'checking')
    
    update_package_status(ui_components, package_key, ui_status)
    
    # Log progress
    status_emoji = {
        'installed': '✅', 
        'missing': '❌', 
        'upgrade': '⚠️'
    }.get(status_info['status'], '🔍')
    
    if logger:
        logger.debug(f"{status_emoji} {package['name']}: {status_info['status']}")

def _reset_all_package_statuses(
    ui_components: Dict[str, Any], 
    package_categories: List[Dict]
) -> None:
    """Reset status semua package ke 'checking'"""
    for category in package_categories:
        for package in category['packages']:
            update_package_status(ui_components, package['key'], 'checking')

def _finalize_analysis(
    progress_tracker: Any,
    ui_components: Dict[str, Any],
    analysis_results: Dict[str, Any],
    logger: Optional[Any]
) -> None:
    """Finalisasi hasil analisis"""
    # Update progress
    _update_progress(progress_tracker, 'overall', 90, "📝 Menyiapkan laporan...", logger)
    _update_progress(progress_tracker, 'current', 95, "📱 Memperbarui UI...", logger)
    
    # Tampilkan laporan
    _display_analysis_report(ui_components, analysis_results, logger)
    
    # Update status panel
    _update_status_panel(ui_components, analysis_results)
    
    # Selesaikan progress
    _complete_progress(progress_tracker, ui_components, logger)

def _display_analysis_report(
    ui_components: Dict[str, Any],
    analysis_results: Dict[str, Any],
    logger: Optional[Any]
) -> None:
    """Tampilkan laporan analisis"""
    # Generate dan tampilkan laporan HTML
    report_html = generate_analysis_summary_report(analysis_results)
    log_output = ui_components.get('log_output')
    
    if log_output and hasattr(log_output, 'clear_output'):
        with log_output:
            from IPython.display import display, HTML
            display(HTML(report_html))
    
    # Log summary
    if logger:
        installed = len(analysis_results['installed'])
        missing = len(analysis_results['missing'])
        upgrade = len(analysis_results['upgrade_needed'])
        total = installed + missing + upgrade
        
        logger.info("📊 Ringkasan Analisis:")
        logger.info(f"   ✅ Terinstall: {installed}/{total}")
        logger.info(f"   ❌ Hilang: {missing}/{total}")
        logger.info(f"   ⚠️ Perlu Upgrade: {upgrade}/{total}")

def _update_status_panel(
    ui_components: Dict[str, Any],
    analysis_results: Dict[str, Any]
) -> None:
    """Update status panel berdasarkan hasil analisis"""
    installed = len(analysis_results['installed'])
    missing = len(analysis_results['missing'])
    upgrade = len(analysis_results['upgrade_needed'])
    
    if missing == 0 and upgrade == 0:
        status_msg = f"✅ Semua {installed} packages sudah terinstall dengan benar"
        status_type = "success"
    else:
        status_msg = (
            f"📊 Hasil Analisis: {installed} terinstall, "
            f"{missing} hilang, {upgrade} perlu upgrade"
        )
        status_type = "info"
    
    update_status_panel(ui_components, status_msg, status_type)

def _complete_progress(
    progress_tracker: Any,
    ui_components: Dict[str, Any],
    logger: Optional[Any]
) -> None:
    """Selesaikan progress tracking"""
    try:
        if progress_tracker:
            _update_progress(progress_tracker, 'overall', 100, "✅ Analisis selesai", logger)
            _update_progress(progress_tracker, 'current', 100, "✅ Selesai", logger)
            
            if hasattr(progress_tracker, 'complete'):
                progress_tracker.complete("✅ Analisis dependensi selesai")
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Gagal menyelesaikan progress tracker: {str(e)}")
        
        # Fallback ke progress tracking lama
        update_progress = ui_components.get('update_progress')
        if callable(update_progress):
            update_progress('overall', 100, "✅ Analisis selesai")
            update_progress('step', 100, "✅ Selesai")

def _handle_analysis_error(
    error: Exception,
    progress_tracker: Any,
    ui_components: Dict[str, Any],
    logger: Optional[Any]
) -> None:
    """Handle error selama analisis"""
    error_msg = f"❌ Analisis gagal: {str(error)}"
    log_to_ui_safe(ui_components, error_msg, "error")
    
    # Update progress tracker untuk error
    try:
        if progress_tracker and hasattr(progress_tracker, 'error'):
            progress_tracker.error(error_msg)
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Gagal menampilkan error di progress tracker: {str(e)}")
        
        # Fallback ke error operation lama
        error_operation = ui_components.get('error_operation')
        if callable(error_operation):
            error_operation(error_msg)
    
    if logger:
        logger.error(f"💥 Error analisis: {str(error)}", exc_info=True)
    
    update_status_panel(ui_components, error_msg, "error")
