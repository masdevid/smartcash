"""
File: smartcash/ui/dataset/downloader/handlers/check_handler.py
Deskripsi: Fixed check handler tanpa threading dan proper widget access
"""

from typing import Dict, Any, Callable
from smartcash.ui.utils.fallback_utils import show_status_safe
from smartcash.dataset.downloader.roboflow_client import create_roboflow_client
from smartcash.dataset.utils.path_validator import get_path_validator

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Callable:
    """Setup check handler tanpa threading"""
    
    def handle_check(button):
        """Handle check dataset operation tanpa threading"""
        button.disabled = True
        
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
            
            # Execute check langsung tanpa threading
            _execute_check_sync(ui_components, current_config, logger)
            
        except Exception as e:
            logger.error(f"❌ Error check handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
        finally:
            button.disabled = False
    
    return handle_check

def _execute_check_sync(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Execute check operation secara synchronous"""
    try:
        # Show progress
        progress_tracker = ui_components.get('tracker')
        progress_tracker and progress_tracker.show('check')
        progress_tracker and progress_tracker.update('overall', 0, "🔍 Memulai pengecekan dataset...")
        
        workspace, project, version, api_key = config['workspace'], config['project'], config['version'], config['api_key']
        dataset_id = f"{workspace}/{project}:v{version}"
        
        # Step 1: Check Roboflow connection
        progress_tracker and progress_tracker.update('overall', 10, "🌐 Mengecek koneksi Roboflow...")
        roboflow_client = create_roboflow_client(api_key, logger)
        
        # Step 2: Validate credentials
        progress_tracker and progress_tracker.update('overall', 30, "🔑 Validasi kredensial...")
        cred_result = roboflow_client.validate_credentials(workspace, project)
        
        if not cred_result['valid']:
            error_msg = f"❌ Kredensial tidak valid: {cred_result['message']}"
            progress_tracker and progress_tracker.error(error_msg)
            show_status_safe(error_msg, "error", ui_components)
            return
        
        # Step 3: Get dataset metadata
        progress_tracker and progress_tracker.update('overall', 50, "📊 Mengambil metadata dataset...")
        metadata_result = roboflow_client.get_dataset_metadata(workspace, project, version)
        
        if metadata_result['status'] != 'success':
            error_msg = f"❌ Gagal ambil metadata: {metadata_result['message']}"
            progress_tracker and progress_tracker.error(error_msg)
            show_status_safe(error_msg, "error", ui_components)
            return
        
        # Step 4: Check local dataset
        progress_tracker and progress_tracker.update('overall', 70, "📁 Mengecek dataset lokal...")
        local_check = _check_local_dataset_sync(config)
        
        # Step 5: Generate report
        progress_tracker and progress_tracker.update('overall', 90, "📋 Menyusun laporan...")
        report = _generate_check_report(metadata_result['data'], local_check, dataset_id)
        
        # Show results
        progress_tracker and progress_tracker.complete("✅ Pengecekan selesai")
        show_status_safe(report, "info", ui_components)
        logger.info(f"📊 Check completed: {dataset_id}")
        
    except Exception as e:
        error_msg = f"❌ Error saat check: {str(e)}"
        progress_tracker = ui_components.get('tracker')
        progress_tracker and progress_tracker.error(error_msg)
        show_status_safe(error_msg, "error", ui_components)
        logger.error(error_msg)

def _check_local_dataset_sync(config: Dict[str, Any]) -> Dict[str, Any]:
    """Check existing local dataset dengan path validator"""
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

def _generate_check_report(remote_metadata: Dict[str, Any], local_check: Dict[str, Any], dataset_id: str) -> str:
    """Generate comprehensive check report dengan emoji dan formatting"""
    
    # Remote dataset info
    remote_classes = len(remote_metadata.get('project', {}).get('classes', []))
    remote_images = remote_metadata.get('version', {}).get('images', 0)
    remote_size_mb = remote_metadata.get('export', {}).get('size', 0)
    
    # Local dataset info
    local_exists = local_check['exists']
    local_images = local_check['total_images']
    local_labels = local_check['total_labels']
    
    # Build report dengan one-liner formatting
    report_lines = [
        f"📊 **Dataset Check Report: {dataset_id}**",
        "",
        "🌐 **Remote Dataset (Roboflow):**",
        f"   • Kelas: {remote_classes}",
        f"   • Gambar: {remote_images:,}",
        f"   • Ukuran: {remote_size_mb:.1f} MB",
        "",
        "💻 **Local Dataset:**"
    ]
    
    if local_exists:
        splits_found = ', '.join(s for s in local_check['splits'] if local_check['splits'][s]['exists'])
        report_lines.extend([
            f"   • Status: ✅ Ditemukan",
            f"   • Gambar: {local_images:,}",
            f"   • Label: {local_labels:,}",
            f"   • Splits: {splits_found}"
        ])
        
        # Comparison dengan one-liner
        if local_images != remote_images:
            diff = abs(local_images - remote_images)
            status = "🔄" if local_images < remote_images else "📈"
            report_lines.extend([
                "",
                "🔍 **Perbandingan:**",
                f"   • Selisih gambar: {status} {diff:,}"
            ])
    else:
        report_lines.extend([
            f"   • Status: ❌ Tidak ditemukan",
            f"   • Rekomendasi: Download dataset terlebih dahulu"
        ])
    
    # Issues dengan one-liner processing
    local_check.get('issues') and report_lines.extend([
        "",
        "⚠️ **Issues:**"
    ] + [f"   • {issue}" for issue in local_check['issues'][:5]])
    
    # Status summary dengan one-liner logic
    status_emoji = "✅" if local_exists and not local_check.get('issues') else "⚠️" if local_exists else "❌"
    status_text = 'Ready' if local_exists and not local_check.get('issues') else 'Needs attention' if local_exists else 'Download required'
    report_lines.extend([
        "",
        f"{status_emoji} **Status: {status_text}**"
    ])
    
    return "\n".join(report_lines)