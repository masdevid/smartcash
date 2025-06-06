"""
File: smartcash/ui/dataset/downloader/handlers/check_handler.py
Deskripsi: Fixed check handler dengan safe method calls dan proper error handling
"""

from typing import Dict, Any, Callable
from smartcash.ui.utils.fallback_utils import show_status_safe
from smartcash.dataset.downloader.roboflow_client import create_roboflow_client
from smartcash.dataset.utils.path_validator import get_path_validator
from smartcash.common.utils.one_liner_fixes import safe_operation_or_none

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Callable:
    """Setup optimized check handler dengan fixed method calls"""
    
    def handle_check(button):
        """Handle check operation dengan safe validation flow"""
        try:
            # Get current config safely
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                show_status_safe("❌ Config handler tidak ditemukan", "error", ui_components)
                return
            
            current_config = safe_operation_or_none(lambda: config_handler.extract_config(ui_components)) or {}
            
            # Validate config safely
            validation = safe_operation_or_none(lambda: config_handler.validate_config(current_config)) or {'valid': False, 'errors': ['Validation failed']}
            if not validation.get('valid', False):
                error_msg = f"❌ Config tidak valid: {'; '.join(validation.get('errors', []))}"
                show_status_safe(error_msg, "error", ui_components)
                return
            
            # Execute check dengan safe method calls
            _execute_check_sync_fixed(ui_components, current_config, logger)
            
        except Exception as e:
            logger.error(f"❌ Error check handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
    
    return handle_check

def _execute_check_sync_fixed(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Execute check dengan safe local dataset checking"""
    try:
        # Get progress tracker safely
        progress_tracker = ui_components.get('progress_tracker')
        if not progress_tracker:
            logger.error("❌ Progress tracker tidak ditemukan")
            show_status_safe("❌ Progress tracker tidak tersedia", "error", ui_components)
            return
        
        # Show progress dengan safe API calls
        check_steps = ["validate", "connect", "credentials", "metadata", "local_check", "report"]
        step_weights = {"validate": 10, "connect": 15, "credentials": 20, "metadata": 25, "local_check": 20, "report": 10}
        
        # Safe progress tracker initialization
        safe_operation_or_none(lambda: progress_tracker.show("Check Dataset", check_steps, step_weights))
        
        workspace = config.get('workspace', '')
        project = config.get('project', '')
        version = config.get('version', '')
        api_key = config.get('api_key', '')
        dataset_id = f"{workspace}/{project}:v{version}"
        
        # Step 1: Validate parameters
        safe_operation_or_none(lambda: progress_tracker.update('overall', 10, "🔍 Validasi parameter"))
        safe_operation_or_none(lambda: progress_tracker.update('current', 50, "Validating parameters..."))
        
        # Step 2: Connect to Roboflow
        safe_operation_or_none(lambda: progress_tracker.update('overall', 20, "🌐 Koneksi Roboflow"))
        safe_operation_or_none(lambda: progress_tracker.update('current', 0, "Connecting to Roboflow..."))
        
        roboflow_client = safe_operation_or_none(lambda: create_roboflow_client(api_key, logger))
        if not roboflow_client:
            error_msg = "❌ Gagal membuat Roboflow client"
            safe_operation_or_none(lambda: progress_tracker.error(error_msg))
            show_status_safe(error_msg, "error", ui_components)
            return
        
        safe_operation_or_none(lambda: progress_tracker.update('current', 100, "Connected to Roboflow"))
        
        # Step 3: Validate credentials
        safe_operation_or_none(lambda: progress_tracker.update('overall', 35, "🔑 Validasi kredensial"))
        safe_operation_or_none(lambda: progress_tracker.update('current', 0, "Validating credentials..."))
        
        cred_result = safe_operation_or_none(lambda: roboflow_client.validate_credentials(workspace, project)) or {'valid': False, 'message': 'Connection failed'}
        
        if not cred_result.get('valid', False):
            error_msg = f"❌ Kredensial tidak valid: {cred_result.get('message', 'Unknown error')}"
            safe_operation_or_none(lambda: progress_tracker.error(error_msg))
            show_status_safe(error_msg, "error", ui_components)
            return
        
        safe_operation_or_none(lambda: progress_tracker.update('current', 100, "Credentials validated"))
        
        # Step 4: Get dataset metadata
        safe_operation_or_none(lambda: progress_tracker.update('overall', 55, "📊 Mengambil metadata"))
        safe_operation_or_none(lambda: progress_tracker.update('current', 0, "Fetching dataset metadata..."))
        
        metadata_result = safe_operation_or_none(lambda: roboflow_client.get_dataset_metadata(workspace, project, version)) or {'status': 'error', 'message': 'Metadata fetch failed'}
        
        if metadata_result.get('status') != 'success':
            error_msg = f"❌ Gagal mendapatkan metadata: {metadata_result.get('message', 'Unknown error')}"
            safe_operation_or_none(lambda: progress_tracker.error(error_msg))
            show_status_safe(error_msg, "error", ui_components)
            return
        
        safe_operation_or_none(lambda: progress_tracker.update('current', 100, "Metadata retrieved"))
        
        # Step 5: Check local dataset - FIXED
        safe_operation_or_none(lambda: progress_tracker.update('overall', 80, "📁 Memeriksa dataset lokal"))
        safe_operation_or_none(lambda: progress_tracker.update('current', 0, "Checking local dataset..."))
        
        local_check = _check_local_dataset_fixed(config)
        safe_operation_or_none(lambda: progress_tracker.update('current', 100, "Local check completed"))
        
        # Step 6: Generate report
        safe_operation_or_none(lambda: progress_tracker.update('overall', 95, "📋 Membuat laporan"))
        safe_operation_or_none(lambda: progress_tracker.update('current', 0, "Generating report..."))
        
        report = _generate_check_report_fixed(
            metadata_result.get('data', {}), 
            local_check, 
            dataset_id
        )
        
        # Show report safely
        _show_report_safely(ui_components, report)
        
        # Complete progress
        safe_operation_or_none(lambda: progress_tracker.complete("✅ Pengecekan selesai"))
        show_status_safe("✅ Pengecekan dataset selesai", "success", ui_components)
        logger.info(f"📊 Check completed: {dataset_id}")
        
    except Exception as e:
        error_msg = f"❌ Error saat check: {str(e)}"
        progress_tracker = ui_components.get('progress_tracker')
        safe_operation_or_none(lambda: progress_tracker.error(error_msg) if progress_tracker else None)
        show_status_safe(error_msg, "error", ui_components)
        logger.error(error_msg)

def _check_local_dataset_fixed(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fixed local dataset checking dengan safe path validation"""
    def check_operation():
        path_validator = get_path_validator()
        dataset_paths = path_validator.get_dataset_paths()
        
        # Safe validation call
        validation_result = path_validator.validate_dataset_structure(dataset_paths['data_root'])
        
        return {
            'exists': validation_result.get('valid', False),
            'total_images': validation_result.get('total_images', 0),
            'total_labels': validation_result.get('total_labels', 0),
            'splits': validation_result.get('splits', {}),
            'issues': validation_result.get('issues', [])
        }
    
    return safe_operation_or_none(check_operation) or {
        'exists': False, 'total_images': 0, 'total_labels': 0, 
        'splits': {}, 'issues': ['Error checking local dataset']
    }

def _generate_check_report_fixed(remote_metadata: Dict[str, Any], local_check: Dict[str, Any], dataset_id: str) -> str:
    """Generate fixed check report dengan safe data extraction"""
    def report_operation():
        # Safe remote data extraction
        project_info = remote_metadata.get('project', {})
        version_info = remote_metadata.get('version', {})
        export_info = remote_metadata.get('export', {})
        
        remote_classes = len(project_info.get('classes', []))
        remote_images = version_info.get('images', 0)
        remote_size_mb = export_info.get('size', 0)
        
        # Local dataset info
        local_exists = local_check.get('exists', False)
        local_images = local_check.get('total_images', 0)
        local_labels = local_check.get('total_labels', 0)
        
        # Build report sections
        base_sections = [
            f"📊 **Dataset Check Report: {dataset_id}**", "",
            "🌐 **Remote Dataset (Roboflow):**",
            f"   • Kelas: {remote_classes}",
            f"   • Gambar: {remote_images:,}",
            f"   • Ukuran: {remote_size_mb:.1f} MB",
            f"   • Format: YOLOv5 PyTorch", "",
            "💻 **Local Dataset:**"
        ]
        
        # Safe local status sections
        if local_exists:
            splits_info = local_check.get('splits', {})
            available_splits = [s for s, info in splits_info.items() if info.get('exists', False)]
            
            local_sections = [
                f"   • Status: ✅ Ditemukan",
                f"   • Gambar: {local_images:,}",
                f"   • Label: {local_labels:,}",
                f"   • Splits: {', '.join(available_splits) if available_splits else 'None'}"
            ]
            
            # Safe comparison section
            comparison_sections = []
            if local_images != remote_images:
                diff_emoji = '🔄' if local_images < remote_images else '📈'
                comparison_sections = [
                    "", "🔍 **Perbandingan:**",
                    f"   • Selisih gambar: {diff_emoji} {abs(local_images - remote_images):,}"
                ]
        else:
            local_sections = [
                f"   • Status: ❌ Tidak ditemukan",
                f"   • Rekomendasi: Download dataset terlebih dahulu"
            ]
            comparison_sections = []
        
        # Safe issues section
        issues = local_check.get('issues', [])
        issue_sections = []
        if issues:
            issue_sections = [
                "", "⚠️ **Issues:**"
            ] + [f"   • {issue}" for issue in issues[:5]]
        
        # Safe status summary
        if local_exists and not issues:
            status_emoji, status_text = "✅", 'Ready'
        elif local_exists:
            status_emoji, status_text = "⚠️", 'Needs attention'
        else:
            status_emoji, status_text = "❌", 'Download required'
        
        # Combine all sections
        all_sections = (base_sections + local_sections + comparison_sections + 
                       issue_sections + ["", f"{status_emoji} **Status: {status_text}**"])
        
        return '\n'.join(all_sections)
    
    return safe_operation_or_none(report_operation) or f"❌ Error generating report for {dataset_id}"

def _show_report_safely(ui_components: Dict[str, Any], report: str) -> None:
    """Show report safely dalam log output"""
    def show_operation():
        log_output = ui_components.get('log_output')
        if log_output and hasattr(log_output, 'clear_output'):
            with log_output:
                log_output.clear_output(wait=True)
                from IPython.display import Markdown, display
                display(Markdown(report))
    
    safe_operation_or_none(show_operation)

# Fixed utilities dengan safe operations
def get_check_status_fixed(ui: Dict[str, Any]) -> Dict[str, Any]:
    """Get check status safely"""
    def status_operation():
        return {
            'ready': bool(ui.get('progress_tracker')) and bool(ui.get('config_handler')),
            'tracker_available': bool(ui.get('progress_tracker')),
            'config_available': bool(ui.get('config_handler'))
        }
    
    return safe_operation_or_none(status_operation) or {'ready': False, 'tracker_available': False, 'config_available': False}

def validate_check_requirements_fixed(config: Dict[str, Any]) -> bool:
    """Validate check requirements safely"""
    def validate_operation():
        required_fields = ['workspace', 'project', 'version', 'api_key']
        return all(config.get(field, '').strip() for field in required_fields)
    
    return bool(safe_operation_or_none(validate_operation))

def create_dataset_identifier_fixed(workspace: str, project: str, version: str) -> str:
    """Create dataset identifier safely"""
    def create_operation():
        w_clean = workspace.strip() if workspace else ''
        p_clean = project.strip() if project else ''
        v_clean = version.strip() if version else ''
        
        if not all([w_clean, p_clean, v_clean]):
            return 'invalid/invalid:v0'
        
        return f"{w_clean}/{p_clean}:v{v_clean}"
    
    return safe_operation_or_none(create_operation) or 'unknown/unknown:v0'

def format_check_summary_fixed(remote: Dict[str, Any], local: Dict[str, Any]) -> str:
    """Format check summary safely"""
    def format_operation():
        remote_images = remote.get('images', 0) if remote else 0
        local_status = 'Found' if local.get('exists', False) else 'Not found'
        return f"Remote: {remote_images} imgs | Local: {local_status}"
    
    return safe_operation_or_none(format_operation) or "Remote: 0 imgs | Local: Unknown"