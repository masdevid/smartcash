"""
File: smartcash/ui/dataset/downloader/handlers/check_handler.py
Deskripsi: Fixed check handler dengan proper method calls dan one-liner style
"""

from typing import Dict, Any, Callable
from smartcash.ui.utils.fallback_utils import show_status_safe
from smartcash.dataset.downloader.roboflow_client import create_roboflow_client
from smartcash.dataset.utils.path_validator import get_path_validator

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Callable:
    """Setup optimized check handler dengan fixed method calls"""
    
    def handle_check(button):
        """Handle check operation dengan fixed validation flow"""
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
            
            # Execute check dengan fixed progress API
            _execute_check_sync_fixed(ui_components, current_config, logger)
            
        except Exception as e:
            logger.error(f"❌ Error check handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
    
    return handle_check

def _execute_check_sync_fixed(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Execute check dengan fixed local dataset checking"""
    try:
        # Get progress tracker
        progress_tracker = ui_components.get('progress_tracker')
        if not progress_tracker:
            logger.error("❌ Progress tracker tidak ditemukan")
            show_status_safe("❌ Progress tracker tidak tersedia", "error", ui_components)
            return
        
        # Show progress dengan API yang benar
        check_steps = ["validate", "connect", "credentials", "metadata", "local_check", "report"]
        step_weights = {"validate": 10, "connect": 15, "credentials": 20, "metadata": 25, "local_check": 20, "report": 10}
        progress_tracker.show("Check Dataset", check_steps, step_weights)
        
        workspace, project, version, api_key = config['workspace'], config['project'], config['version'], config['api_key']
        dataset_id = f"{workspace}/{project}:v{version}"
        
        # Step 1: Validate parameters
        progress_tracker.update('overall', 10, "🔍 Validasi parameter")
        progress_tracker.update('current', 50, "Validating parameters...")
        
        # Step 2: Connect to Roboflow
        progress_tracker.update('overall', 20, "🌐 Koneksi Roboflow")
        progress_tracker.update('current', 0, "Connecting to Roboflow...")
        roboflow_client = create_roboflow_client(api_key, logger)
        progress_tracker.update('current', 100, "Connected to Roboflow")
        
        # Step 3: Validate credentials
        progress_tracker.update('overall', 35, "🔑 Validasi kredensial")
        progress_tracker.update('current', 0, "Validating credentials...")
        cred_result = roboflow_client.validate_credentials(workspace, project)
        
        if not cred_result['valid']:
            error_msg = f"❌ Kredensial tidak valid: {cred_result['message']}"
            progress_tracker.error(error_msg)
            show_status_safe(error_msg, "error", ui_components)
            return
        
        progress_tracker.update('current', 100, "Credentials validated")
        
        # Step 4: Get dataset metadata
        progress_tracker.update('overall', 55, "📊 Mengambil metadata")
        progress_tracker.update('current', 0, "Fetching dataset metadata...")
        metadata_result = roboflow_client.get_dataset_metadata(workspace, project, version)
        
        if metadata_result['status'] != 'success':
            error_msg = f"❌ Gagal mendapatkan metadata: {metadata_result['message']}"
            progress_tracker.error(error_msg)
            show_status_safe(error_msg, "error", ui_components)
            return
        
        progress_tracker.update('current', 100, "Metadata retrieved")
        
        # Step 5: Check local dataset - FIXED
        progress_tracker.update('overall', 80, "📁 Memeriksa dataset lokal")
        progress_tracker.update('current', 0, "Checking local dataset...")
        local_check = _check_local_dataset_fixed(config)  # Fixed function call
        progress_tracker.update('current', 100, "Local check completed")
        
        # Step 6: Generate report
        progress_tracker.update('overall', 95, "📋 Membuat laporan")
        progress_tracker.update('current', 0, "Generating report...")
        report = _generate_check_report_fixed(metadata_result['data'], local_check, dataset_id)
        
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

def _check_local_dataset_fixed(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fixed local dataset checking dengan proper path validation"""
    try:
        path_validator = get_path_validator()
        dataset_paths = path_validator.get_dataset_paths()
        
        # Fixed validation call - method not property
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

def _generate_check_report_fixed(remote_metadata: Dict[str, Any], local_check: Dict[str, Any], dataset_id: str) -> str:
    """Generate fixed check report dengan proper data extraction"""
    
    # Fixed remote data extraction
    remote_classes = len(remote_metadata.get('project', {}).get('classes', []))
    remote_images = remote_metadata.get('version', {}).get('images', 0)
    remote_size_mb = remote_metadata.get('export', {}).get('size', 0)
    
    # Local dataset info
    local_exists, local_images, local_labels = local_check['exists'], local_check['total_images'], local_check['total_labels']
    
    # Build report sections
    base_sections = [
        f"📊 **Dataset Check Report: {dataset_id}**", "",
        "🌐 **Remote Dataset (Roboflow):**",
        f"   • Kelas: {remote_classes}",
        f"   • Gambar: {remote_images:,}",
        f"   • Ukuran: {remote_size_mb:.1f} MB",
        f"   • Format: YOLOv5 PyTorch (hardcoded)", "",
        "💻 **Local Dataset:**"
    ]
    
    # Fixed local status sections
    if local_exists:
        local_sections = [
            f"   • Status: ✅ Ditemukan",
            f"   • Gambar: {local_images:,}",
            f"   • Label: {local_labels:,}",
            f"   • Splits: {', '.join(s for s in local_check['splits'] if local_check['splits'][s].get('exists', False))}"
        ]
        
        # Fixed comparison section
        if local_images != remote_images:
            comparison_sections = [
                "", "🔍 **Perbandingan:**",
                f"   • Selisih gambar: {'🔄' if local_images < remote_images else '📈'} {abs(local_images - remote_images):,}"
            ]
        else:
            comparison_sections = []
    else:
        local_sections = [
            f"   • Status: ❌ Tidak ditemukan",
            f"   • Rekomendasi: Download dataset terlebih dahulu"
        ]
        comparison_sections = []
    
    # Fixed issues section
    if local_check.get('issues'):
        issue_sections = [
            "", "⚠️ **Issues:**"
        ] + [f"   • {issue}" for issue in local_check['issues'][:5]]
    else:
        issue_sections = []
    
    # Fixed status summary
    if local_exists and not local_check.get('issues'):
        status_emoji, status_text = "✅", 'Ready'
    elif local_exists:
        status_emoji, status_text = "⚠️", 'Needs attention'
    else:
        status_emoji, status_text = "❌", 'Download required'
    
    # Combine all sections
    all_sections = base_sections + local_sections + comparison_sections + issue_sections + ["", f"{status_emoji} **Status: {status_text}**"]
    
    return '\n'.join(all_sections)

# Fixed utilities
get_check_status_fixed = lambda ui: {'ready': 'progress_tracker' in ui and 'config_handler' in ui, 'tracker_available': 'progress_tracker' in ui}
validate_check_requirements_fixed = lambda config: all(config.get(field, '').strip() for field in ['workspace', 'project', 'version', 'api_key'])
create_dataset_identifier_fixed = lambda w, p, v: f"{w}/{p}:v{v}"
format_check_summary_fixed = lambda remote, local: f"Remote: {remote.get('images', 0)} imgs | Local: {'Found' if local.get('exists') else 'Not found'}"