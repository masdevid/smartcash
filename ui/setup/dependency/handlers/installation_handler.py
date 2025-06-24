"""
File: smartcash/ui/setup/dependency/handlers/installation_handler_refactored.py
Deskripsi: Handler untuk instalasi package dengan logging terstandarisasi

Fitur Utama:
- Manajemen instalasi paralel
- Pelacakan progress real-time
- Penanganan error yang kuat
- Laporan hasil terperinci
"""

from dataclasses import dataclass
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Tuple

# Core imports
from smartcash.common import get_logger
from smartcash.common.threadpools import get_optimal_thread_count, process_in_parallel, safe_worker_count
from smartcash.ui.setup.dependency.utils import (
    LogLevel, with_logging, requires, log_to_ui_safe, update_status_panel,
    batch_update_package_status, create_operation_context, install_single_package,
    filter_uninstalled_packages, get_selected_packages, generate_installation_summary_report
)

# Konstanta
PROGRESS_STEPS = {
    'init': (0, "🔄 Mempersiapkan..."),
    'analysis': (10, "🔍 Menganalisis package..."),
    'installation': (50, "⚙️ Menginstal package..."),
    'complete': (100, "✅ Proses selesai")
}

# Setup logger
logger = get_logger(__name__)

@dataclass
class InstallationResult:
    """Kelas untuk menyimpan hasil instalasi"""
    success: bool
    message: str = ""
    error: Optional[Exception] = None

def _update_progress(
    progress_tracker: Any,
    step: str,
    message: Optional[str] = None,
    current: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """Update progress tracker dengan error handling"""
    if not progress_tracker:
        return
        
    progress, default_msg = PROGRESS_STEPS.get(step, (0, step))
    msg = message or default_msg
    
    try:
        if step == 'init':
            progress_tracker.show(
                operation="📦 Instalasi Package",
                steps=["🔍 Analisis", "⚙️ Instalasi"],
                level='dual'
            )
        
        progress_tracker.update_overall(progress, msg)
        if current is not None:
            progress_tracker.update_current(current, f"{msg} ({current}%)")
            
    except Exception as e:
        if logger:
            logger.warning(f"Gagal update progress: {str(e)}")

def _handle_operation_status(
    ui_components: Dict[str, Any],
    status: str,
    message: str,
    progress_tracker: Optional[Any] = None
) -> None:
    """Handle status operasi dengan konsisten"""
    log_func = getattr(ui_components.get('logger', logger), 
                      'error' if status == 'error' else 'info')
    
    try:
        log_to_ui_safe(ui_components, message, status)
        update_status_panel(ui_components, message, status)
        log_func(message)
        
        if progress_tracker:
            if status == 'error' and hasattr(progress_tracker, 'error'):
                progress_tracker.error(message)
                
    except Exception as e:
        logger.error(f"Gagal memperbarui status: {str(e)}", exc_info=True)

def _process_single_package(
    pkg: str,
    config: Dict[str, Any],
    ui_components: Dict[str, Any]
) -> Tuple[str, bool]:
    """Proses instalasi satu package"""
    try:
        result = install_single_package(pkg, config.get('timeout', 300))
        return pkg, result.get('success', False)
    except Exception as e:
        error_msg = f"Gagal menginstall {pkg}: {str(e)}"
        _handle_operation_status(ui_components, 'error', error_msg)
        return pkg, False

def _install_packages_parallel(
    packages: List[str],
    ui_components: Dict[str, Any],
    config: Dict[str, Any],
    progress_tracker: Any
) -> Dict[str, bool]:
    """Install packages secara paralel"""
    if not packages:
        return {}
        
    total = len(packages)
    results = {}
    max_workers = safe_worker_count(get_optimal_thread_count('io'))
    
    def update_progress(completed: int, current_pkg: str = '') -> None:
        progress = int((completed / total) * 100)
        _update_progress(
            progress_tracker,
            'installation',
            f"⚙️ Menginstal {current_pkg or ''}",
            progress,
            logger
        )
    
    try:
        # Proses instalasi paralel
        results_list = process_in_parallel(
            items=packages,
            process_func=lambda p: _process_single_package(p, config, ui_components),
            max_workers=max_workers,
            desc="Memproses instalasi package"
        )
        
        # Konversi ke dictionary
        results = dict(results_list)
        
        # Update progress akhir
        success_count = sum(1 for r in results.values() if r)
        update_progress(100, f"✅ {success_count}/{total} berhasil")
        
    except Exception as e:
        error_msg = f"Error saat instalasi paralel: {str(e)}"
        _handle_operation_status(ui_components, 'error', error_msg, progress_tracker)
    
    return results

@with_logging("Execute Installation Process", LogLevel.INFO, ui_components_key='ui_components')
@requires('progress_tracker')
def _execute_installation_with_utils(
    ui_components: Dict[str, Any],
    config: Dict[str, Any],
    ctx: Any
) -> None:
    """Eksekusi proses instalasi"""
    progress_tracker = ui_components['progress_tracker']
    start_time = time.time()
    
    try:
        # Inisialisasi
        _update_progress(progress_tracker, 'init', logger=logger)
        
        # Dapatkan package yang dipilih
        selected = get_selected_packages(ui_components)
        if not selected:
            _handle_operation_status(
                ui_components,
                'warning',
                "⚠️ Tidak ada package yang dipilih",
                progress_tracker
            )
            return
            
        # Analisis package
        _update_progress(progress_tracker, 'analysis', logger=logger)
        packages = filter_uninstalled_packages(
            selected,
            lambda msg: log_to_ui_safe(ui_components, msg)
        )
        
        if not packages:
            _handle_operation_status(
                ui_components,
                'success',
                "✅ Semua package sudah terinstall",
                progress_tracker
            )
            return
            
        # Proses instalasi
        results = _install_packages_parallel(packages, ui_components, config, progress_tracker)
        
        # Tampilkan hasil
        duration = time.time() - start_time
        _handle_installation_results(ui_components, results, duration, progress_tracker)
        
    except Exception as e:
        error_msg = f"❌ Error dalam proses instalasi: {str(e)}"
        _handle_operation_status(ui_components, 'error', error_msg, progress_tracker)
        raise

def _handle_installation_results(
    ui_components: Dict[str, Any],
    results: Dict[str, bool],
    duration: float,
    progress_tracker: Any
) -> None:
    """Tampilkan hasil instalasi"""
    if not results:
        _handle_operation_status(
            ui_components,
            'warning',
            "⚠️ Tidak ada package yang diproses",
            progress_tracker
        )
        return
        
    total = len(results)
    success = sum(1 for r in results.values() if r)
    failed = total - success
    
    # Update status UI
    status = 'success' if success == total else 'warning' if success > 0 else 'error'
    message = (
        f"✅ {success}/{total} package berhasil diinstall" if status == 'success' else
        f"⚠️  {success} berhasil, {failed} gagal" if status == 'warning' else
        f"❌ Gagal menginstall {failed} package"
    )
    
    _handle_operation_status(ui_components, status, message, progress_tracker)
    
    # Generate dan tampilkan laporan
    report = generate_installation_summary_report(results, duration)
    if 'log_output' in ui_components:
        try:
            from IPython.display import display, HTML
            with ui_components['log_output']:
                display(HTML(report))
        except Exception as e:
            logger.warning(f"Gagal menampilkan laporan HTML: {str(e)}")
    
    # Update status package di UI
    status_mapping = {
        pkg.split('>=')[0].split('==')[0].split('<')[0].strip(): 
        'installed' if success else 'error'
        for pkg, success in results.items()
    }
    batch_update_package_status(ui_components, status_mapping)
    
    # Update progress tracker
    _update_progress(
        progress_tracker,
        'complete',
        f"✅ Selesai ({success}/{total} berhasil)",
        100,
        logger
    )

@with_logging("Setup Installation Handler", LogLevel.INFO)
@requires('install_button')
def setup_installation_handler(
    ui_components: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Inisialisasi handler untuk tombol instalasi"""
    def execute_installation(button=None):
        with create_operation_context(ui_components, 'installation') as ctx:
            _execute_installation_with_utils(ui_components, config or {}, ctx)
    
    try:
        ui_components['install_button'].on_click(execute_installation)
        logger.debug("Installation handler berhasil dipasang")
    except Exception as e:
        logger.error(f"Gagal memasang installation handler: {str(e)}", exc_info=True)
        raise
