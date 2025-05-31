"""
File: smartcash/ui/setup/dependency_installer/handlers/package_handler.py
Deskripsi: Enhanced package handler dengan progress tracking yang lebih informatif
"""

import time
import subprocess
import sys
from typing import Dict, Any, List, Tuple
from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message

def get_all_missing_packages(ui_components: Dict[str, Any]) -> List[str]:
    """Dapatkan semua package yang perlu diinstall berdasarkan UI state"""
    from smartcash.ui.setup.dependency_installer.utils.package_utils import (
        get_package_groups, parse_custom_packages, get_installed_packages, check_missing_packages
    )
    
    # Update progress step - Analisis
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        progress_tracker.update('step', 25, "Menganalisis packages...", "#007bff")
    
    installed_packages = get_installed_packages()
    package_groups = get_package_groups()
    missing_packages = []
    
    # Add selected packages dari groups
    for pkg_key, pkg_list in package_groups.items():
        checkbox = ui_components.get(pkg_key)
        if checkbox and checkbox.value:
            package_list = pkg_list() if callable(pkg_list) else pkg_list
            missing = check_missing_packages(package_list, installed_packages)
            missing_packages.extend(missing)
    
    # Add custom packages
    custom_text = ui_components.get('custom_packages', type('', (), {'value': ''})).value.strip()
    if custom_text:
        custom_packages = parse_custom_packages(custom_text)
        missing = check_missing_packages(custom_packages, installed_packages)
        missing_packages.extend(missing)
    
    # Update progress step - Analisis selesai
    if progress_tracker:
        progress_tracker.update('step', 50, "Analisis selesai", "#28a745")
    
    return list(dict.fromkeys(missing_packages))  # Remove duplicates

def run_batch_installation(packages: List[str], ui_components: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Jalankan instalasi batch untuk package dengan progress tracking yang lebih informatif dan observer pattern"""
    if not packages:
        return True, {'total': 0, 'success': 0, 'failed': 0, 'duration': 0, 'errors': []}
    
    # Get progress tracker dan observer manager
    progress_tracker = ui_components.get('progress_tracker')
    observer_manager = ui_components.get('observer_manager')
    
    # Initialize stats
    stats = {'total': len(packages), 'success': 0, 'failed': 0, 'duration': 0, 'errors': []}
    start_time = time.time()
    
    # Notify start via observer
    if observer_manager and hasattr(observer_manager, 'notify'):
        try:
            observer_manager.notify('DEPENDENCY_INSTALL_START', None, {
                'message': f"Memulai instalasi {len(packages)} package",
                'timestamp': time.time(),
                'total_packages': len(packages)
            })
        except Exception as e:
            # Silent fail untuk observer notification
            pass
    
    # Update progress step - Persiapan
    if progress_tracker:
        progress_tracker.update('step', 25, "Mempersiapkan instalasi...", "#007bff")
    elif 'update_progress' in ui_components:
        ui_components['update_progress']('step', 25, "Mempersiapkan instalasi...", "#007bff")
    
    log_message(ui_components, f"🚀 Memulai instalasi {len(packages)} package...", "info")
    
    # Update progress step - Instalasi
    if progress_tracker:
        progress_tracker.update('step', 50, "Memulai instalasi packages...", "#007bff")
    elif 'update_progress' in ui_components:
        ui_components['update_progress']('step', 50, "Memulai instalasi packages...", "#007bff")
    
    # Install each package
    for i, package in enumerate(packages):
        progress_pct = int((i / len(packages)) * 100)
        
        # Update progress tracker dengan informasi yang lebih detail
        if progress_tracker:
            progress_tracker.update('overall', progress_pct, f"Progress: {i+1}/{len(packages)} ({progress_pct}%)", "#007bff")
            progress_tracker.update('current', 0, f"Installing {package}...", "#ffc107")
        elif 'update_progress' in ui_components:
            ui_components['update_progress']('overall', progress_pct, f"Progress: {i+1}/{len(packages)} ({progress_pct}%)", "#007bff")
            ui_components['update_progress']('current', 0, f"Installing {package}...", "#ffc107")
        
        # Notify progress via observer
        if observer_manager and hasattr(observer_manager, 'notify'):
            try:
                observer_manager.notify('DEPENDENCY_INSTALL_PROGRESS', None, {
                    'message': f"Installing {package}... ({i+1}/{len(packages)})",
                    'timestamp': time.time(),
                    'progress': progress_pct,
                    'current_package': package,
                    'current_index': i+1,
                    'total_packages': len(packages)
                })
            except Exception:
                pass  # Silent fail untuk observer notification
        
        log_message(ui_components, f"📦 Installing {package}... ({i+1}/{len(packages)})", "info")
        
        try:
            # Run pip install silently
            cmd = [sys.executable, "-m", "pip", "install", package, "--quiet"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                stats['success'] += 1
                log_message(ui_components, f"✅ Berhasil install {package}", "success")
                if progress_tracker:
                    progress_tracker.update('current', 100, f"✅ {package} berhasil diinstall", "#28a745")
                elif 'update_progress' in ui_components:
                    ui_components['update_progress']('current', 100, f"✅ {package} berhasil diinstall", "#28a745")
            else:
                stats['failed'] += 1
                error_msg = result.stderr.strip() or "Unknown error"
                stats['errors'].append((package, error_msg))
                log_message(ui_components, f"❌ Gagal install {package}: {error_msg}", "error")
                if progress_tracker:
                    progress_tracker.update('current', 100, f"❌ {package} gagal diinstall", "#dc3545")
                elif 'update_progress' in ui_components:
                    ui_components['update_progress']('current', 100, f"❌ {package} gagal diinstall", "#dc3545")
                
                # Notify error via observer
                if observer_manager and hasattr(observer_manager, 'notify'):
                    try:
                        observer_manager.notify('DEPENDENCY_INSTALL_ERROR', None, {
                            'message': f"Gagal install {package}: {error_msg}",
                            'timestamp': time.time(),
                            'package': package,
                            'error': error_msg
                        })
                    except Exception:
                        pass  # Silent fail untuk observer notification
        
        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append((package, str(e)))
            log_message(ui_components, f"💥 Error install {package}: {str(e)}", "error")
            if progress_tracker:
                progress_tracker.update('current', 100, f"💥 {package} error: {str(e)}", "#dc3545")
            elif 'update_progress' in ui_components:
                ui_components['update_progress']('current', 100, f"💥 {package} error: {str(e)}", "#dc3545")
            
            # Notify error via observer
            if observer_manager and hasattr(observer_manager, 'notify'):
                try:
                    observer_manager.notify('DEPENDENCY_INSTALL_ERROR', None, {
                        'message': f"Error install {package}: {str(e)}",
                        'timestamp': time.time(),
                        'package': package,
                        'error': str(e)
                    })
                except Exception:
                    pass  # Silent fail untuk observer notification
    
    # Final progress update
    if progress_tracker:
        progress_tracker.update('overall', 100, "Instalasi selesai", "#28a745")
        progress_tracker.update('step', 100, "Instalasi selesai", "#28a745")
    elif 'update_progress' in ui_components:
        ui_components['update_progress']('overall', 100, "Instalasi selesai", "#28a745")
        ui_components['update_progress']('step', 100, "Instalasi selesai", "#28a745")
    
    stats['duration'] = time.time() - start_time
    
    # Format pesan dengan highlight parameter numerik
    success_message = f"📊 Instalasi selesai: {stats['success']}/{stats['total']} berhasil, {stats['failed']} gagal ({stats['duration']:.1f}s)"
    log_message(ui_components, success_message, "success")
    
    # Notify complete via observer
    if observer_manager and hasattr(observer_manager, 'notify'):
        try:
            observer_manager.notify('DEPENDENCY_INSTALL_COMPLETE', None, {
                'message': success_message,
                'timestamp': time.time(),
                'duration': stats['duration'],
                'success': stats['success'],
                'failed': stats['failed'],
                'total': stats['total'],
                'stats': stats
            })
        except Exception:
            pass  # Silent fail untuk observer notification
    
    return stats['failed'] == 0, stats