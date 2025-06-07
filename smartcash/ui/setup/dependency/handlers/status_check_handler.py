"""
File: smartcash/ui/setup/dependency/handlers/status_check_handler.py
Deskripsi: Fixed status check handler dengan proper logger reference
"""

from typing import Dict, Any
from IPython.display import display, HTML

from smartcash.ui.setup.dependency.utils.system_info_utils import (
    get_comprehensive_system_info, check_system_requirements
)
from smartcash.ui.setup.dependency.utils.package_utils import (
    get_installed_packages_dict, get_package_detailed_info
)
from smartcash.ui.setup.dependency.utils.ui_state_utils import (
    create_operation_context, ProgressSteps, update_package_status_by_name,
    update_status_panel, log_to_ui_safe
)
from smartcash.ui.setup.dependency.utils.report_generator_utils import (
    generate_comprehensive_status_report, generate_system_compatibility_report
)
from smartcash.ui.setup.dependency.components.package_selector import (
    get_package_categories, update_package_status
)

def setup_status_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup status check handler dengan fixed logger reference"""
    
    def execute_status_check(button=None):
        """Execute comprehensive status check dengan operation context"""
        with create_operation_context(ui_components, 'status_check') as ctx:
            _execute_status_check_with_utils(ui_components, config, ctx)
    
    ui_components['check_button'].on_click(execute_status_check)

def _execute_status_check_with_utils(ui_components: Dict[str, Any], config: Dict[str, Any], ctx):
    """Execute status check dengan fixed logger reference"""
    
    logger = ui_components.get('logger')  # Get logger from ui_components
    
    # Get progress tracker jika tersedia
    progress_tracker = ui_components.get('progress_tracker')
    
    try:
        # Step 1: Initialize check dengan emoji untuk visual feedback yang lebih detail
        if progress_tracker:
            # Gunakan show() untuk triple-level progress tracking jika tersedia
            if hasattr(progress_tracker, 'show') and hasattr(progress_tracker, 'update_step_progress'):
                progress_tracker.show("Status Check", [
                    "🔧 Persiapan", 
                    "💻 Sistem Info", 
                    "📦 Package Check", 
                    "📝 Laporan"
                ])
                # Inisialisasi level-3 progress
                progress_tracker.update_step_progress(0, "⚙️ Mempersiapkan analisis")
            
            # Update level-1 dan level-2 progress
            progress_tracker.update_overall(10, "🚀 Memulai status check...")
            progress_tracker.update_current(25, "🔍 Inisialisasi analisis dependensi...")
        else:
            # Fallback untuk progress tracking lama
            ctx.stepped_progress('STATUS_INIT', "🚀 Memulai status check...")
        
        log_to_ui_safe(ui_components, "🔍 Memeriksa status dependensi...")
        
        # Step 2: Get system information dengan emoji dan detail progress
        if progress_tracker:
            # Update level-1 progress
            progress_tracker.update_overall(30, "💻 Mengumpulkan informasi sistem...")
            
            # Update level-2 progress
            progress_tracker.update_current(50, "💻 Scanning hardware dan software...")
            
            # Update level-3 progress jika tersedia
            if hasattr(progress_tracker, 'update_step_progress'):
                progress_tracker.update_step_progress(25, "💽 Memeriksa hardware")
        else:
            # Fallback untuk progress tracking lama
            ctx.stepped_progress('STATUS_SYSTEM_INFO', "💻 Mengumpulkan informasi sistem...")
        
        # Dapatkan informasi sistem
        system_info, system_requirements = get_comprehensive_system_info(), check_system_requirements()
        
        # Update progress setelah mendapatkan informasi sistem
        if progress_tracker and hasattr(progress_tracker, 'update_step_progress'):
            progress_tracker.update_step_progress(100, "✅ Informasi sistem terkumpul")
            
        log_to_ui_safe(ui_components, "💻 System information collected")
        
        # Step 3: Get comprehensive package status dengan emoji dan detail progress
        if progress_tracker:
            # Update level-1 progress
            progress_tracker.update_overall(50, "📦 Checking package status...")
            
            # Update level-2 progress
            progress_tracker.update_current(0, "🔍 Memulai scanning packages...")
            
            # Reset level-3 progress jika tersedia
            if hasattr(progress_tracker, 'update_step_progress'):
                progress_tracker.update_step_progress(0, "📚 Mempersiapkan analisis paket")
        else:
            # Fallback untuk progress tracking lama
            ctx.stepped_progress('STATUS_PACKAGE_CHECK', "📦 Checking package status...")
            
        package_status = _get_comprehensive_package_status_with_utils(ui_components, ctx, logger)
        
        # Step 4: Generate detailed report dengan emoji dan detail progress
        if progress_tracker:
            # Update level-1 progress
            progress_tracker.update_overall(70, "📝 Generating report...")
            
            # Update level-2 progress
            progress_tracker.update_current(85, "📊 Compiling data...")
            
            # Update level-3 progress jika tersedia
            if hasattr(progress_tracker, 'update_step_progress'):
                progress_tracker.update_step_progress(50, "📈 Menyusun statistik paket")
        else:
            # Fallback untuk progress tracking lama
            ctx.stepped_progress('STATUS_REPORT', "📝 Generating report...")
            
        _display_comprehensive_report_with_utils(ui_components, system_info, system_requirements, package_status, ctx, logger)
        
        # Step 5: Update UI status dengan emoji dan detail progress
        if progress_tracker:
            # Update level-1 progress
            progress_tracker.update_overall(90, "👌 Updating UI status...")
            
            # Update level-2 progress
            progress_tracker.update_current(95, "📈 Refreshing UI components...")
            
            # Update level-3 progress jika tersedia
            if hasattr(progress_tracker, 'update_step_progress'):
                progress_tracker.update_step_progress(75, "🔄 Memperbarui komponen UI")
        else:
            # Fallback untuk progress tracking lama
            ctx.stepped_progress('STATUS_UI_UPDATE', "👌 Updating UI status...")
            
        _update_ui_status_from_check_with_utils(ui_components, package_status)
        
        # Hitung statistik paket untuk laporan
        total_packages = len(package_status)
        installed_count = sum(1 for status in package_status.values() if status.get('installed', False))
        missing_count = total_packages - installed_count
        
        # Summary dengan emoji dan progress tracker
        summary_msg = f"📊 Status Check: {installed_count}/{total_packages} packages terinstall"
        log_to_ui_safe(ui_components, f"✅ {summary_msg}")
        update_status_panel(ui_components, summary_msg, "success")
        
        # Complete operation dengan emoji, delay, dan informasi yang lebih detail
        if progress_tracker:
            # Update level-1 dan level-2 progress ke 100%
            progress_tracker.update_overall(100, "✅ Status check selesai")
            progress_tracker.update_current(100, f"✅ {installed_count}/{total_packages} paket terinstall")
            
            # Update level-3 progress jika tersedia dengan ringkasan akhir
            if hasattr(progress_tracker, 'update_step_progress'):
                progress_tracker.update_step_progress(100, f"📊 {installed_count}/{total_packages} paket terinstall")
                
            # Tandai operasi selesai dengan delay untuk UX yang lebih baik
            progress_tracker.complete("✅ Pemeriksaan status dependensi selesai", delay=1.5)
        else:
            # Fallback untuk progress tracking lama
            ctx.stepped_progress('STATUS_COMPLETE', "✅ Status check selesai", "overall")
            ctx.stepped_progress('STATUS_COMPLETE', "✅ Complete", "step")
        
    except Exception as e:
        # Buat pesan error yang informatif dengan emoji untuk visual feedback
        error_msg = f"❌ Gagal memeriksa status dependensi: {str(e)}"
        
        # Log error ke UI dan console
        log_to_ui_safe(ui_components, error_msg, "error")
        if logger:
            logger.error(f"💥 Status check error: {str(e)}")
        
        # Update status panel dengan pesan error
        update_status_panel(ui_components, error_msg, "error")
            
        # Tampilkan error dengan progress tracker jika tersedia
        if progress_tracker:
            # Update level-1 progress untuk menunjukkan error
            progress_tracker.update_overall(100, "❌ Error pada status check")
            
            # Update level-2 progress dengan detail error
            progress_tracker.update_current(100, f"❌ {str(e)[:30]}..." if len(str(e)) > 30 else f"❌ {str(e)}")
            
            # Update level-3 progress jika tersedia
            if hasattr(progress_tracker, 'update_step_progress'):
                progress_tracker.update_step_progress(100, "⚠️ Lihat log untuk detail")
            
            # Tandai operasi error dengan delay untuk UX yang lebih baik
            progress_tracker.error(error_msg, delay=1.5)
        else:
            # Fallback untuk progress tracking lama
            ui_components.get('error_operation', lambda x: None)(error_msg)
            
        # Re-raise exception untuk penanganan di level yang lebih tinggi
        raise

def _get_comprehensive_package_status_with_utils(ui_components: Dict[str, Any], ctx, logger) -> Dict[str, Dict[str, Any]]:
    """Get comprehensive status dengan fixed logger reference dan progress tracker"""
    
    # Get progress tracker jika tersedia
    progress_tracker = ui_components.get('progress_tracker')
    
    package_status = {}
    package_categories = get_package_categories()
    installed_packages = get_installed_packages_dict()
    
    if logger:
        logger.info("🔍 Analyzing package categories...")
    
    # Persiapkan progress tracker untuk triple-level tracking
    if progress_tracker:
        # Inisialisasi progress tracker dengan label yang informatif
        progress_tracker.update_overall(10, "🔍 Memulai analisis status paket...")
        
        # Tampilkan step labels jika mendukung triple-level
        if hasattr(progress_tracker, 'show') and hasattr(progress_tracker, 'update_step_progress'):
            progress_tracker.show("Status Check", [
                "🔍 Scanning", 
                "📊 Analisis", 
                "📝 Laporan"
            ])
    
    # Process all packages dengan progress tracking yang lebih detail
    total_categories = len(package_categories)
    for cat_idx, category in enumerate(package_categories):
        # Update progress level-2 (current) dengan emoji untuk visual feedback
        cat_progress = int(((cat_idx + 1) / total_categories) * 100)
        cat_message = f"📚 Checking {category['name']} category..."
        
        if progress_tracker:
            # Update level-1 (overall) dengan progress keseluruhan
            overall_progress = 10 + int((cat_idx / total_categories) * 80)  # 10-90% range
            progress_tracker.update_overall(overall_progress, f"🔍 Analyzing {cat_idx+1}/{total_categories} categories")
            
            # Update level-2 (current) dengan progress kategori
            progress_tracker.update_current(cat_progress, cat_message)
            
            # Reset level-3 (step) untuk kategori baru
            if hasattr(progress_tracker, 'update_step_progress'):
                progress_tracker.update_step_progress(0, f"📓 {category['name']}")
        
        if logger:
            logger.info(f"🔍 Checking {category['name']} category...")
        
        # Process packages dalam category dengan weight-based progress
        total_packages = len(category['packages'])
        for pkg_idx, package in enumerate(category['packages']):
            package_key = package['key']
            package_name = package['pip_name'].split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()
            
            # Tentukan status dengan emoji yang sesuai
            is_installed = package_name in installed_packages
            status_emoji = "✅" if is_installed else "❌"
            
            # Update step progress (level-3) jika tersedia
            if progress_tracker and hasattr(progress_tracker, 'update_step_progress'):
                pkg_progress = int(((pkg_idx + 1) / total_packages) * 100)
                progress_tracker.update_step_progress(pkg_progress, f"{status_emoji} {package['name']}")
            
            # Dapatkan informasi detail paket
            package_status[package_key] = {
                'name': package['name'], 
                'pip_name': package['pip_name'], 
                'category': category['name'],
                'package_name': package_name,
                **_get_detailed_package_info_with_utils(package_name, installed_packages, logger)
            }
    
    # Finalisasi progress tracking dengan informasi lengkap
    if progress_tracker:
        # Update level-1 (overall) ke 100%
        progress_tracker.update_overall(100, "✅ Analisis status paket selesai")
        
        # Update level-2 (current) dengan ringkasan
        total_packages = sum(len(cat['packages']) for cat in package_categories)
        installed_count = sum(1 for status in package_status.values() if status.get('is_installed', False))
        progress_tracker.update_current(100, f"📊 {installed_count}/{total_packages} paket terinstall")
        
        # Update level-3 (step) dengan status akhir
        if hasattr(progress_tracker, 'update_step_progress'):
            progress_tracker.update_step_progress(100, "📝 Laporan status lengkap")
            
        # Tandai operasi selesai dengan delay untuk UX yang lebih baik
        if hasattr(progress_tracker, 'complete'):
            progress_tracker.complete("✅ Status check selesai", delay=1.5)
    
    if logger:
        logger.info(f"✅ Status check selesai: {len(package_status)} paket dianalisis")
    
    return package_status

def _get_detailed_package_info_with_utils(package_name: str, installed_packages: Dict[str, str], logger) -> Dict[str, Any]:
    """Get detailed info dengan fixed logger reference"""
    
    from smartcash.ui.setup.dependency.utils.package_utils import is_package_installed, get_package_version
    
    # Check installation
    if not is_package_installed(package_name, installed_packages):
        return {'installed': False}
    
    # Get detailed info
    version = get_package_version(package_name, installed_packages)
    detailed_info = get_package_detailed_info(package_name)
    
    return {
        'installed': True, 'version': version,
        'location': detailed_info.get('location'), 'dependencies': detailed_info.get('requires', []),
        'summary': detailed_info.get('summary', ''), 'author': detailed_info.get('author', ''),
        'home_page': detailed_info.get('home-page', '')
    }

def _display_comprehensive_report_with_utils(ui_components: Dict[str, Any], system_info: Dict[str, Any], 
                                           system_requirements: Dict[str, Any], package_status: Dict[str, Dict[str, Any]], ctx, logger):
    """Display comprehensive report dengan fixed logger reference"""
    
    log_output = ui_components.get('log_output')
    if not log_output:
        return
    
    # Generate combined report
    combined_report = f"{generate_comprehensive_status_report(system_info, package_status)}{generate_system_compatibility_report(system_info)}"
    
    # Display report
    with log_output: 
        display(HTML(combined_report))
    
    # Log summary
    _log_status_summary_with_utils(ui_components, package_status, system_requirements, ctx, logger)

def _log_status_summary_with_utils(ui_components: Dict[str, Any], package_status: Dict[str, Dict[str, Any]], 
                                 system_requirements: Dict[str, Any], ctx, logger):
    """Log summary dengan fixed logger reference"""
    
    # Category summary
    category_summary = {}
    for pkg_info in package_status.values():
        category = pkg_info['category']
        if category not in category_summary:
            category_summary[category] = {'total': 0, 'installed': 0}
        category_summary[category]['total'] += 1
        if pkg_info['installed']:
            category_summary[category]['installed'] += 1
    
    # Log category summaries
    for category, stats in category_summary.items():
        percentage = (stats['installed']/stats['total']*100) if stats['total'] > 0 else 0
        if logger:
            logger.info(f"📋 {category}: {stats['installed']}/{stats['total']} ({percentage:.1f}%)")
    
    # Log system requirements warnings
    if not system_requirements.get('all_requirements_met', True) and logger:
        logger.warning("⚠️ Some system requirements not met")
        requirement_checks = [
            ('python_version_ok', 'Python version < 3.7'),
            ('memory_sufficient', 'Memory < 2GB'),
            ('platform_supported', 'Platform not officially supported')
        ]
        
        for check, msg in requirement_checks:
            if not system_requirements.get(check):
                logger.warning(f"   • {msg}")

def _update_ui_status_from_check_with_utils(ui_components: Dict[str, Any], package_status: Dict[str, Dict[str, Any]]):
    """Update UI package status dengan batch utils"""
    
    # Create status mapping dan batch update
    status_mapping = {pkg_info['package_name']: 'installed' if pkg_info['installed'] else 'missing' 
                     for pkg_info in package_status.values()}
    
    from smartcash.ui.setup.dependency.utils.ui_state_utils import batch_update_package_status
    batch_update_package_status(ui_components, status_mapping)