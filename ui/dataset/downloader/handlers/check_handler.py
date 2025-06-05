"""
File: smartcash/ui/dataset/downloader/handlers/check_handler.py
Deskripsi: Fixed check handler dengan proper report formatting dan separate report lines
"""

from typing import Dict, Any, Callable, List
from smartcash.ui.utils.fallback_utils import show_status_safe
from smartcash.dataset.downloader.roboflow_client import create_roboflow_client
from smartcash.dataset.utils.path_validator import get_path_validator

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Callable:
    """Setup check handler dengan proper validation dan report formatting"""
    
    def handle_check(button):
        """Handle check dataset operation dengan proper flow dan formatting"""
        try:
            # Get current config dari config handler
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                show_status_safe("❌ Config handler tidak ditemukan", "error", ui_components)
                return
            
            # Extract config dengan proper widget access
            current_config = config_handler.extract_config(ui_components)
            
            # Validate config
            validation = config_handler.validate_config(current_config)
            if not validation['valid']:
                error_msg = f"❌ Config tidak valid: {'; '.join(validation['errors'])}"
                show_status_safe(error_msg, "error", ui_components)
                return
            
            # Execute check dengan proper progress tracking
            _execute_check_sync(ui_components, current_config, logger)
            
        except Exception as e:
            logger.error(f"❌ Error check handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
    
    return handle_check

def _execute_check_sync(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Execute check operation dengan proper dual-level progress tracking"""
    try:
        # Show progress container dengan dual-level tracking
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            # Reset progress tracker terlebih dahulu
            progress_tracker.reset()
            # Tampilkan container progress
            if 'container' in ui_components:
                ui_components['container'].layout.display = 'block'
            # Update progress awal
            progress_tracker.update('level1', 0, "🔍 Memulai pengecekan dataset...")
            progress_tracker.update('level2', 0, "Menyiapkan validasi...")
        
        workspace, project, version, api_key = config['workspace'], config['project'], config['version'], config['api_key']
        dataset_id = f"{workspace}/{project}:v{version}"
        
        # Step 1: Validate parameters
        if progress_tracker:
            progress_tracker.update('level1', 10, "🔍 Validasi parameter")
            progress_tracker.update('level2', 50, "🔍 Validating parameters...")
        
        # Step 2: Check Roboflow connection
        if progress_tracker:
            progress_tracker.update('level1', 20, "🌐 Koneksi Roboflow")
            progress_tracker.update('level2', 0, "🌐 Connecting to Roboflow...")
        roboflow_client = create_roboflow_client(api_key, logger)
        if progress_tracker:
            progress_tracker.update('level2', 100, "🌐 Connected to Roboflow")
        
        # Step 3: Validate credentials
        if progress_tracker:
            progress_tracker.update('level1', 30, "🔑 Validasi kredensial")
            progress_tracker.update('level2', 0, "🔑 Validating credentials...")
        cred_result = roboflow_client.validate_credentials(workspace, project)
        
        if not cred_result['valid']:
            error_msg = f"❌ Kredensial tidak valid: {cred_result['message']}"
            if progress_tracker:
                progress_tracker.error('level1', error_msg)
                progress_tracker.error('level2', "Validasi gagal")
            show_status_safe(error_msg, "error", ui_components)
            return
        
        if progress_tracker:
            progress_tracker.update('level2', 100, "🔑 Credentials validated")
        
        # Step 4: Get dataset metadata
        if progress_tracker:
            progress_tracker.update('level1', 50, "📊 Mengambil metadata")
            progress_tracker.update('level2', 0, "📊 Fetching dataset metadata...")
        metadata_result = roboflow_client.get_dataset_metadata(workspace, project, version)
        
        if not metadata_result['success']:
            error_msg = f"❌ Gagal mendapatkan metadata: {metadata_result['message']}"
            if progress_tracker:
                progress_tracker.error('level1', error_msg)
                progress_tracker.error('level2', "Metadata gagal")
            show_status_safe(error_msg, "error", ui_components)
            return
        
        if progress_tracker:
            progress_tracker.update('level2', 100, "📊 Metadata retrieved")
        
        # Step 5: Check local dataset
        if progress_tracker:
            progress_tracker.update('level1', 70, "📁 Memeriksa dataset lokal")
            progress_tracker.update('level2', 0, "📁 Checking local dataset...")
        local_check = _check_local_dataset_sync(config)
        if progress_tracker:
            progress_tracker.update('level2', 100, "📁 Local check completed")
        
        # Step 6: Generate report
        if progress_tracker:
            progress_tracker.update('level1', 90, "📋 Membuat laporan")
            progress_tracker.update('level2', 0, "📋 Generating report...")
        report = _generate_detailed_check_report(metadata_result['data'], local_check, dataset_id)
        # Show report
        ui_components['log_output'].clear_output(wait=True)
        with ui_components['log_output']:
            from IPython.display import Markdown, display
            display(Markdown(report))
        
        if progress_tracker:
            progress_tracker.update('level1', 100, "✅ Pengecekan selesai")
            progress_tracker.update('level2', 100, "Selesai")
            # Sembunyikan progress setelah delay
            import time
            time.sleep(0.5)
            if 'container' in ui_components:
                ui_components['container'].layout.display = 'none'
        show_status_safe("✅ Pengecekan dataset selesai", "success", ui_components)
        logger.info(f"📊 Check completed: {dataset_id}")
        
    except Exception as e:
        error_msg = f"❌ Error saat check: {str(e)}"
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            progress_tracker.error('level1', error_msg)
            progress_tracker.error('level2', "Error saat check")
            # Sembunyikan progress setelah delay
            import time
            time.sleep(1.0)
            if 'container' in ui_components:
                ui_components['container'].layout.display = 'none'
        show_status_safe(error_msg, "error", ui_components)
        logger.error(error_msg)

def _check_local_dataset_sync(config: Dict[str, Any]) -> Dict[str, Any]:
    """Check existing local dataset dengan proper path validation"""
    try:
        path_validator = get_path_validator()
        dataset_paths = path_validator.get_dataset_paths()
        validation_result = path_validator.validate_dataset_structure(dataset_paths['data_root'])
        
        return {
            'exists': validation_result['valid'],
            'total_images': validation_result['total_images'],
            'total_labels': validation_result['total_labels'],
            'splits': validation_result['splits'],
            'issues': validation_result.get('issues', [])
        }
        
    except Exception as e:
        return {
            'exists': False,
            'error': str(e),
            'total_images': 0,
            'total_labels': 0,
            'splits': {},
            'issues': [f"Error checking local: {str(e)}"]
        }

def _generate_detailed_check_report(remote_metadata: Dict[str, Any], local_check: Dict[str, Any], dataset_id: str) -> str:
    """Generate comprehensive check report dengan proper line separation dan formatting"""
    
    # Remote dataset info dengan one-liner extraction
    remote_classes = len(remote_metadata.get('project', {}).get('classes', []))
    remote_images = remote_metadata.get('version', {}).get('images', 0)
    remote_size_mb = remote_metadata.get('export', {}).get('size', 0)
    
    # Local dataset info
    local_exists = local_check['exists']
    local_images = local_check['total_images']
    local_labels = local_check['total_labels']
    
    # Build report sections separately untuk better formatting
    report_sections = []
    
    # Header section
    report_sections.append(f"📊 **Dataset Check Report: {dataset_id}**")
    report_sections.append("")
    
    # Remote dataset section
    remote_section = [
        "🌐 **Remote Dataset (Roboflow):**",
        f"   • Kelas: {remote_classes}",
        f"   • Gambar: {remote_images:,}",
        f"   • Ukuran: {remote_size_mb:.1f} MB"
    ]
    report_sections.extend(remote_section)
    report_sections.append("")
    
    # Local dataset section
    local_section = ["💻 **Local Dataset:**"]
    
    if local_exists:
        splits_found = ', '.join(s for s in local_check['splits'] if local_check['splits'][s]['exists'])
        local_details = [
            f"   • Status: ✅ Ditemukan",
            f"   • Gambar: {local_images:,}",
            f"   • Label: {local_labels:,}",
            f"   • Splits: {splits_found}"
        ]
        local_section.extend(local_details)
        
        # Comparison section jika ada data lokal
        if local_images != remote_images:
            diff = abs(local_images - remote_images)
            status = "🔄" if local_images < remote_images else "📈"
            comparison_section = [
                "",
                "🔍 **Perbandingan:**",
                f"   • Selisih gambar: {status} {diff:,}"
            ]
            report_sections.extend(local_section)
            report_sections.extend(comparison_section)
        else:
            report_sections.extend(local_section)
    else:
        local_not_found = [
            f"   • Status: ❌ Tidak ditemukan",
            f"   • Rekomendasi: Download dataset terlebih dahulu"
        ]
        local_section.extend(local_not_found)
        report_sections.extend(local_section)
    
    # Issues section jika ada
    if local_check.get('issues'):
        report_sections.append("")
        issues_section = ["⚠️ **Issues:**"]
        issue_details = [f"   • {issue}" for issue in local_check['issues'][:5]]
        issues_section.extend(issue_details)
        report_sections.extend(issues_section)
    
    # Status summary section
    status_emoji = "✅" if local_exists and not local_check.get('issues') else "⚠️" if local_exists else "❌"
    status_text = 'Ready' if local_exists and not local_check.get('issues') else 'Needs attention' if local_exists else 'Download required'
    
    summary_section = [
        "",
        f"{status_emoji} **Status: {status_text}**"
    ]
    report_sections.extend(summary_section)
    
    # Join all sections dengan proper line separation
    return '\n'.join(report_sections)