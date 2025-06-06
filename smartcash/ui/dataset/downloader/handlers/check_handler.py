"""
File: smartcash/ui/dataset/downloader/handlers/check_handler.py
Deskripsi: Optimized check handler dengan progress tracker dual-level dan one-liner style
"""

from typing import Dict, Any, Callable
from smartcash.ui.utils.fallback_utils import show_status_safe
from smartcash.dataset.downloader.roboflow_client import create_roboflow_client
from smartcash.dataset.utils.path_validator import get_path_validator

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Callable:
    """Setup optimized check handler dengan dual-level progress"""
    
    def handle_check(button):
        """Handle check operation dengan streamlined flow"""
        try:
            # Get current config
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                show_status_safe("❌ Config handler tidak ditemukan", "error", ui_components)
                return
            
            current_config = config_handler.extract_config(ui_components)
            
            # Validate config
            validation = config_handler.validate_config(current_config)
            if not validation['valid']:
                error_msg = f"❌ Config tidak valid: {'; '.join(validation['errors'])}"
                show_status_safe(error_msg, "error", ui_components)
                return
            
            # Execute check dengan optimized progress
            _execute_check_sync(ui_components, current_config, logger)
            
        except Exception as e:
            logger.error(f"❌ Error check handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
    
    return handle_check

def _execute_check_sync(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Execute check dengan optimized dual-level progress tracking"""
    try:
        # Get progress tracker
        progress_tracker = ui_components.get('progress_tracker')
        if not progress_tracker:
            logger.error("❌ Progress tracker tidak ditemukan")
            show_status_safe("❌ Progress tracker tidak tersedia", "error", ui_components)
            return
        
        # Show progress
        progress_tracker.show("Check Dataset")
        
        workspace, project, version, api_key = config['workspace'], config['project'], config['version'], config['api_key']
        dataset_id = f"{workspace}/{project}:v{version}"
        
        # Step 1: Validate parameters (0-20%)
        progress_tracker.update_overall(10, "🔍 Validasi parameter")
        progress_tracker.update_current(50, "Validating parameters...")
        
        # Step 2: Connect to Roboflow (20-40%)
        progress_tracker.update_overall(25, "🌐 Koneksi Roboflow")
        progress_tracker.update_current(0, "Connecting to Roboflow...")
        roboflow_client = create_roboflow_client(api_key, logger)
        progress_tracker.update_current(100, "Connected to Roboflow")
        
        # Step 3: Validate credentials (40-60%)
        progress_tracker.update_overall(45, "🔑 Validasi kredensial")
        progress_tracker.update_current(0, "Validating credentials...")
        cred_result = roboflow_client.validate_credentials(workspace, project)
        
        if not cred_result['valid']:
            error_msg = f"❌ Kredensial tidak valid: {cred_result['message']}"
            progress_tracker.error(error_msg)
            show_status_safe(error_msg, "error", ui_components)
            return
        
        progress_tracker.update_current(100, "Credentials validated")
        
        # Step 4: Get dataset metadata (60-80%)
        progress_tracker.update_overall(65, "📊 Mengambil metadata")
        progress_tracker.update_current(0, "Fetching dataset metadata...")
        metadata_result = roboflow_client.get_dataset_metadata(workspace, project, version)
        
        if metadata_result['status'] != 'success':
            error_msg = f"❌ Gagal mendapatkan metadata: {metadata_result['message']}"
            progress_tracker.error(error_msg)
            show_status_safe(error_msg, "error", ui_components)
            return
        
        progress_tracker.update_current(100, "Metadata retrieved")
        
        # Step 5: Check local dataset (80-95%)
        progress_tracker.update_overall(85, "📁 Memeriksa dataset lokal")
        progress_tracker.update_current(0, "Checking local dataset...")
        local_check = _check_local_dataset_sync(config)
        progress_tracker.update_current(100, "Local check completed")
        
        # Step 6: Generate report (95-100%)
        progress_tracker.update_overall(95, "📋 Membuat laporan")
        progress_tracker.update_current(0, "Generating report...")
        report = _generate_streamlined_check_report(metadata_result['data'], local_check, dataset_id)
        
        # Show report
        ui_components['log_output'].clear_output(wait=True)
        with ui_components['log_output']:
            from IPython.display import Markdown, display
            display(Markdown(report))
        
        # Complete progress
        progress_tracker.complete("✅ Pengecekan selesai")
        show_status_safe("✅ Pengecekan dataset selesai", "success", ui_components)
        logger.info(f"📊 Check completed: {dataset_id}")
        
    except Exception as e:
        error_msg = f"❌ Error saat check: {str(e)}"
        progress_tracker = ui_components.get('progress_tracker')
        progress_tracker and progress_tracker.error(error_msg)
        show_status_safe(error_msg, "error", ui_components)
        logger.error(error_msg)

def _check_local_dataset_sync(config: Dict[str, Any]) -> Dict[str, Any]:
    """Check existing local dataset dengan optimized validation"""
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
            'exists': False, 'error': str(e), 'total_images': 0,
            'total_labels': 0, 'splits': {}, 'issues': [f"Error checking local: {str(e)}"]
        }

def _generate_streamlined_check_report(remote_metadata: Dict[str, Any], local_check: Dict[str, Any], dataset_id: str) -> str:
    """Generate streamlined check report dengan optimized formatting"""
    
    # Remote dataset info dengan one-liner extraction
    remote_classes = len(remote_metadata.get('project', {}).get('classes', []))
    remote_images = remote_metadata.get('version', {}).get('images', 0)
    remote_size_mb = remote_metadata.get('export', {}).get('size', 0)
    
    # Local dataset info
    local_exists, local_images, local_labels = local_check['exists'], local_check['total_images'], local_check['total_labels']
    
    # Build report dengan optimized sections
    report_sections = [
        f"📊 **Dataset Check Report: {dataset_id}**",
        "",
        "🌐 **Remote Dataset (Roboflow):**",
        f"   • Kelas: {remote_classes}",
        f"   • Gambar: {remote_images:,}",
        f"   • Ukuran: {remote_size_mb:.1f} MB",
        f"   • Format: YOLOv5 PyTorch (hardcoded)",
        "",
        "💻 **Local Dataset:**"
    ]
    
    if local_exists:
        splits_found = ', '.join(s for s in local_check['splits'] if local_check['splits'][s]['exists'])
        local_details = [
            f"   • Status: ✅ Ditemukan",
            f"   • Gambar: {local_images:,}",
            f"   • Label: {local_labels:,}",
            f"   • Splits: {splits_found}"
        ]
        report_sections.extend(local_details)
        
        # Comparison section jika ada data lokal
        if local_images != remote_images:
            diff = abs(local_images - remote_images)
            status = "🔄" if local_images < remote_images else "📈"
            comparison_section = [
                "",
                "🔍 **Perbandingan:**",
                f"   • Selisih gambar: {status} {diff:,}"
            ]
            report_sections.extend(comparison_section)
    else:
        local_not_found = [
            f"   • Status: ❌ Tidak ditemukan",
            f"   • Rekomendasi: Download dataset terlebih dahulu"
        ]
        report_sections.extend(local_not_found)
    
    # Issues section jika ada
    if local_check.get('issues'):
        report_sections.extend([
            "",
            "⚠️ **Issues:**"
        ])
        issue_details = [f"   • {issue}" for issue in local_check['issues'][:5]]
        report_sections.extend(issue_details)
    
    # Status summary dengan one-liner conditional
    status_emoji = "✅" if local_exists and not local_check.get('issues') else "⚠️" if local_exists else "❌"
    status_text = 'Ready' if local_exists and not local_check.get('issues') else 'Needs attention' if local_exists else 'Download required'
    
    report_sections.extend([
        "",
        f"{status_emoji} **Status: {status_text}**"
    ])
    
    return '\n'.join(report_sections)

# Optimized utilities dengan one-liner style
get_check_status = lambda ui: {'ready': 'progress_tracker' in ui and 'config_handler' in ui, 'tracker_available': 'progress_tracker' in ui}
validate_check_requirements = lambda config: all(config.get(field, '').strip() for field in ['workspace', 'project', 'version', 'api_key'])
create_dataset_identifier = lambda w, p, v: f"{w}/{p}:v{v}"
format_check_summary = lambda remote, local: f"Remote: {remote.get('images', 0)} imgs | Local: {'Found' if local.get('exists') else 'Not found'}"