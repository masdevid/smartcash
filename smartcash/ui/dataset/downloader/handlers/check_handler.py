"""
File: smartcash/ui/dataset/downloader/handlers/check_handler.py
Deskripsi: Fixed check handler tanpa threading dan dengan synchronous execution
"""

from typing import Dict, Any, Callable
from smartcash.ui.utils.fallback_utils import show_status_safe
from smartcash.dataset.downloader.roboflow_client import create_roboflow_client
from smartcash.dataset.utils.path_validator import get_path_validator

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Callable:
    """Setup check handler tanpa konfirmasi dan tanpa threading"""
    
    def handle_check(button):
        """Handle check dataset operation secara synchronous"""
        button.disabled = True
        
        try:
            # Get current config
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                show_status_safe(ui_components, "❌ Config handler tidak ditemukan", "error")
                return
            
            current_config = config_handler.extract_config(ui_components)
            
            # Basic validation
            validation = config_handler.validate_config(current_config)
            if not validation['valid']:
                show_status_safe(ui_components, f"❌ Config tidak valid: {'; '.join(validation['errors'])}", "error")
                return
            
            # Execute check secara synchronous
            _execute_check_sync(ui_components, current_config, logger)
            
        except Exception as e:
            logger.error(f"❌ Error check handler: {str(e)}")
            show_status_safe(ui_components, f"❌ Error: {str(e)}", "error")
        finally:
            button.disabled = False
    
    return handle_check

def _execute_check_sync(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Execute check operation secara synchronous"""
    
    try:
        # Show progress
        progress_tracker = ui_components.get('tracker')
        if progress_tracker:
            progress_tracker.show('check')
            progress_tracker.update('overall', 0, "🔍 Memulai pengecekan dataset...")
        
        workspace = config['workspace']
        project = config['project']
        version = config['version']
        api_key = config['api_key']
        
        dataset_id = f"{workspace}/{project}:v{version}"
        
        # Step 1: Check Roboflow connection
        if progress_tracker:
            progress_tracker.update('overall', 10, "🌐 Mengecek koneksi Roboflow...")
        
        roboflow_client = create_roboflow_client(api_key, logger)
        
        # Step 2: Validate credentials
        if progress_tracker:
            progress_tracker.update('overall', 30, "🔑 Validasi kredensial...")
        
        cred_result = roboflow_client.validate_credentials(workspace, project)
        if not cred_result['valid']:
            error_msg = f"❌ Kredensial tidak valid: {cred_result['message']}"
            if progress_tracker:
                progress_tracker.error(error_msg)
            show_status_safe(ui_components, error_msg, "error")
            return
        
        # Step 3: Get dataset metadata
        if progress_tracker:
            progress_tracker.update('overall', 50, "📊 Mengambil metadata dataset...")
        
        metadata_result = roboflow_client.get_dataset_metadata(workspace, project, version)
        if metadata_result['status'] != 'success':
            error_msg = f"❌ Gagal ambil metadata: {metadata_result['message']}"
            if progress_tracker:
                progress_tracker.error(error_msg)
            show_status_safe(ui_components, error_msg, "error")
            return
        
        # Step 4: Check local dataset
        if progress_tracker:
            progress_tracker.update('overall', 70, "📁 Mengecek dataset lokal...")
        
        local_check = _check_local_dataset(config)
        
        # Step 5: Generate report
        if progress_tracker:
            progress_tracker.update('overall', 90, "📋 Menyusun laporan...")
        
        report = _generate_check_report(metadata_result['data'], local_check, dataset_id)
        
        # Show results
        if progress_tracker:
            progress_tracker.complete("✅ Pengecekan selesai")
        
        show_status_safe(ui_components, report, "info")
        logger.info(f"📊 Check completed: {dataset_id}")
        
    except Exception as e:
        error_msg = f"❌ Error saat check: {str(e)}"
        if progress_tracker:
            progress_tracker.error(error_msg)
        show_status_safe(ui_components, error_msg, "error")
        logger.error(error_msg)

def _check_local_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Check existing local dataset dengan path validator"""
    try:
        path_validator = get_path_validator()
        dataset_paths = path_validator.get_dataset_paths()
        
        # Check dataset structure
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
    
    # Build report
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
        report_lines.extend([
            f"   • Status: ✅ Ditemukan",
            f"   • Gambar: {local_images:,}",
            f"   • Label: {local_labels:,}",
            f"   • Splits: {', '.join(s for s in local_check['splits'] if local_check['splits'][s]['exists'])}"
        ])
        
        # Comparison
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
    
    # Issues
    if local_check.get('issues'):
        report_lines.extend([
            "",
            "⚠️ **Issues:**"
        ])
        report_lines.extend([f"   • {issue}" for issue in local_check['issues'][:5]])
    
    # Status summary
    status_emoji = "✅" if local_exists and not local_check.get('issues') else "⚠️" if local_exists else "❌"
    report_lines.extend([
        "",
        f"{status_emoji} **Status: {'Ready' if local_exists and not local_check.get('issues') else 'Needs attention' if local_exists else 'Download required'}**"
    ])
    
    return "\n".join(report_lines)