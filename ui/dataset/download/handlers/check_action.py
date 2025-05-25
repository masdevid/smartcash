"""
File: smartcash/ui/dataset/download/handlers/check_action.py
Deskripsi: Fixed check action dengan integrasi latest progress_tracking dan button_state_manager
"""

from typing import Dict, Any, List
from IPython.display import display, HTML
from smartcash.ui.utils.button_state_manager import get_button_state_manager
from smartcash.ui.dataset.download.utils.dataset_checker import check_complete_dataset_status, get_dataset_readiness_score

def execute_check_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Check dataset dengan latest progress tracking integration."""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    with button_manager.operation_context('check'):
        try:
            logger and logger.info("🔍 Memulai analisis komprehensif dataset")
            
            _clear_ui_outputs(ui_components)
            
            # Step 1: Initialize progress tracking untuk check operation
            _initialize_check_progress(ui_components)
            
            # Step 2: Analisis struktur dataset
            _update_check_progress(ui_components, 20, "📊 Menganalisis struktur dataset...")
            dataset_status = check_complete_dataset_status()
            
            # Step 3: Calculate readiness score
            _update_check_progress(ui_components, 60, "📈 Menghitung skor kesiapan training...")
            readiness_score = get_dataset_readiness_score(dataset_status)
            
            # Step 4: Generate recommendations
            _update_check_progress(ui_components, 80, "💡 Menggenerate rekomendasi...")
            recommendations = _generate_actionable_recommendations(dataset_status, readiness_score)
            
            # Step 5: Display comprehensive results
            _update_check_progress(ui_components, 95, "📋 Menyiapkan laporan...")
            _display_enhanced_results(ui_components, dataset_status, readiness_score, recommendations)
            
            # Complete progress dengan latest integration
            _complete_check_progress(ui_components, "Analisis dataset selesai")
            
            logger and logger.success("🎉 Analisis dataset selesai!")
            
        except Exception as e:
            logger and logger.error(f"❌ Check error: {str(e)}")
            _error_check_progress(ui_components, f"Check error: {str(e)}")
            raise

def _initialize_check_progress(ui_components: Dict[str, Any]) -> None:
    """Initialize progress tracking dengan latest ProgressTracker integration."""
    # Use latest progress tracking methods
    if 'show_for_operation' in ui_components:
        ui_components['show_for_operation']('check')
    elif 'tracker' in ui_components:
        ui_components['tracker'].show('check')
    
    # Initialize with starting message
    _update_check_progress(ui_components, 0, "🔍 Memulai analisis dataset...")

def _update_check_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update check progress dengan latest ProgressTracker integration."""
    # Use latest progress tracking methods
    if 'update_progress' in ui_components:
        ui_components['update_progress']('overall', progress, message)
    elif 'tracker' in ui_components:
        ui_components['tracker'].update('overall', progress, message)

def _complete_check_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Complete check progress dengan latest integration."""
    if 'complete_operation' in ui_components:
        ui_components['complete_operation'](f"✅ {message}")
    elif 'tracker' in ui_components:
        ui_components['tracker'].complete(message)

def _error_check_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Set error state dengan latest integration."""
    if 'error_operation' in ui_components:
        ui_components['error_operation'](message)
    elif 'tracker' in ui_components:
        ui_components['tracker'].error(message)

def _generate_actionable_recommendations(dataset_status: Dict[str, Any], 
                                       readiness_score: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate actionable recommendations berdasarkan analysis results."""
    recommendations = []
    
    final_dataset = dataset_status['final_dataset']
    downloads_folder = dataset_status['downloads_folder']
    summary = dataset_status['summary']
    
    # Dataset availability recommendations
    if summary['status'] == 'empty':
        recommendations.append({
            'type': 'critical', 'icon': '📥',
            'title': 'Download Dataset',
            'description': 'Dataset belum tersedia. Download dataset terlebih dahulu.',
            'priority': 'high'
        })
    elif summary['status'] == 'downloaded':
        recommendations.append({
            'type': 'warning', 'icon': '📁',
            'title': 'Organisir Dataset',
            'description': f'Dataset sudah terdownload ({downloads_folder["total_files"]} files) tapi belum diorganisir.',
            'priority': 'high'
        })
    
    # Training readiness recommendations
    if readiness_score['overall_score'] < 60:
        recommendations.append({
            'type': 'warning', 'icon': '⚠️',
            'title': 'Dataset Belum Siap Training',
            'description': f'Skor kesiapan: {readiness_score["overall_score"]}/100 - {readiness_score["readiness_level"]}',
            'priority': 'medium'
        })
    elif readiness_score['overall_score'] >= 80:
        recommendations.append({
            'type': 'success', 'icon': '🚀',
            'title': 'Siap untuk Training',
            'description': f'Dataset siap digunakan (Skor: {readiness_score["overall_score"]}/100)',
            'priority': 'info'
        })
    
    # Split-specific recommendations
    train_split = final_dataset['splits'].get('train', {})
    valid_split = final_dataset['splits'].get('valid', {})
    
    if not train_split.get('exists', False) or train_split.get('images', 0) < 100:
        recommendations.append({
            'type': 'error', 'icon': '🏋️',
            'title': 'Split Training Kurang',
            'description': f'Split training hanya {train_split.get("images", 0)} gambar (minimum 100)',
            'priority': 'high'
        })
    
    if not valid_split.get('exists', False) or valid_split.get('images', 0) < 20:
        recommendations.append({
            'type': 'warning', 'icon': '✅',
            'title': 'Split Validation Kurang',
            'description': f'Split validation hanya {valid_split.get("images", 0)} gambar (minimum 20)',
            'priority': 'medium'
        })
    
    return recommendations

def _display_enhanced_results(ui_components: Dict[str, Any],
                            dataset_status: Dict[str, Any],
                            readiness_score: Dict[str, Any], 
                            recommendations: List[Dict[str, Any]]) -> None:
    """Display comprehensive results dengan enhanced UI."""
    logger = ui_components.get('logger')
    
    # 1. Overall Status
    _display_overall_status(logger, dataset_status, readiness_score)
    
    # 2. Dataset Structure Details
    _display_dataset_structure(logger, dataset_status)
    
    # 3. Training Readiness Score
    _display_readiness_score(logger, readiness_score)
    
    # 4. Actionable Recommendations
    _display_recommendations(logger, recommendations)
    
    # 5. Storage Information
    _display_storage_info(logger, dataset_status)

def _display_overall_status(logger, dataset_status: Dict[str, Any], 
                          readiness_score: Dict[str, Any]) -> None:
    """Display overall status summary."""
    if not logger:
        return
    
    summary = dataset_status['summary']
    storage_type = dataset_status['storage_info']['type']
    
    # Status dengan emoji dan color coding
    status_config = {
        'ready': {'emoji': '✅', 'color': 'success'},
        'downloaded': {'emoji': '📥', 'color': 'warning'}, 
        'empty': {'emoji': '❌', 'color': 'error'}
    }
    
    config = status_config.get(summary['status'], {'emoji': '❓', 'color': 'info'})
    
    logger.info("=" * 60)
    logger.info(f"📊 DATASET STATUS REPORT")
    logger.info("=" * 60)
    logger.info(f"{config['emoji']} Status: {summary['message']}")
    logger.info(f"💾 Storage: {storage_type}")
    logger.info(f"🎯 Training Score: {readiness_score['overall_score']}/100 ({readiness_score['readiness_level']})")

def _display_dataset_structure(logger, dataset_status: Dict[str, Any]) -> None:
    """Display detailed dataset structure."""
    if not logger:
        return
    
    final_dataset = dataset_status['final_dataset']
    downloads_folder = dataset_status['downloads_folder']
    
    logger.info("")
    logger.info("📁 STRUKTUR DATASET")
    logger.info("-" * 40)
    
    # Final dataset structure
    if final_dataset['exists']:
        logger.success(f"✅ Dataset Final: {final_dataset['total_images']} gambar, {final_dataset['total_labels']} label")
        
        for split in ['train', 'valid', 'test']:
            split_info = final_dataset['splits'][split]
            if split_info['exists'] and split_info['images'] > 0:
                logger.info(f"   📂 {split.title()}: {split_info['images']} gambar ({split_info['labels']} label)")
                logger.info(f"      Path: {split_info['path']}")
    else:
        logger.warning("⚠️ Dataset final belum tersedia")
    
    # Downloads folder
    if downloads_folder['exists']:
        logger.info(f"📥 Downloads: {downloads_folder['total_files']} files ({downloads_folder['total_size_mb']} MB)")

def _display_readiness_score(logger, readiness_score: Dict[str, Any]) -> None:
    """Display training readiness score breakdown."""
    if not logger:
        return
    
    logger.info("")
    logger.info("🎯 SKOR KESIAPAN TRAINING")
    logger.info("-" * 40)
    
    breakdown = readiness_score['score_breakdown']
    overall_score = readiness_score['overall_score']
    readiness_level = readiness_score['readiness_level']
    
    # Overall score dengan color coding
    if overall_score >= 80:
        logger.success(f"🏆 Score Keseluruhan: {overall_score}/100 - {readiness_level}")
    elif overall_score >= 60:
        logger.warning(f"⚠️ Score Keseluruhan: {overall_score}/100 - {readiness_level}")
    else:
        logger.error(f"❌ Score Keseluruhan: {overall_score}/100 - {readiness_level}")

def _display_recommendations(logger, recommendations: List[Dict[str, Any]]) -> None:
    """Display actionable recommendations."""
    if not logger or not recommendations:
        return
    
    logger.info("")
    logger.info("💡 REKOMENDASI TINDAKAN")
    logger.info("-" * 40)
    
    # Group by priority
    high_priority = [r for r in recommendations if r.get('priority') == 'high']
    medium_priority = [r for r in recommendations if r.get('priority') == 'medium']
    
    # Display high priority first
    for group_name, items in [("🔴 PRIORITAS TINGGI", high_priority), ("🟡 PRIORITAS SEDANG", medium_priority)]:
        if items:
            logger.info(f"{group_name}:")
            for rec in items:
                logger.info(f"   {rec['icon']} {rec['title']}")
                logger.info(f"      {rec['description']}")
                logger.info("")

def _display_storage_info(logger, dataset_status: Dict[str, Any]) -> None:
    """Display storage information."""
    if not logger:
        return
    
    storage_info = dataset_status['storage_info']
    
    logger.info("💾 INFORMASI STORAGE")
    logger.info("-" * 40)
    logger.info(f"📍 Type: {storage_info['type']}")
    logger.info(f"📁 Base Path: {storage_info['base_path']}")
    logger.info(f"🔒 Persistent: {'Ya' if storage_info['persistent'] else 'Tidak'}")

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs untuk memulai fresh."""
    try:
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
    except Exception:
        pass