"""
File: smartcash/ui/dataset/download/actions/check_executor.py
Deskripsi: Check action executor dengan comprehensive dataset analysis
"""

from typing import Dict, Any
from smartcash.ui.dataset.download.utils.dataset_checker import check_complete_dataset_status, get_dataset_readiness_score

def execute_check_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Execute check action dengan comprehensive dataset analysis."""
    logger = ui_components.get('logger')
    
    try:
        logger and logger.info("🔍 Memulai analisis komprehensif dataset")
        _clear_ui_outputs(ui_components)
        _initialize_check_progress(ui_components)
        
        # Step 1: Analisis struktur dataset (30%)
        _update_check_progress(ui_components, 30, "📊 Menganalisis struktur dataset...")
        dataset_status = check_complete_dataset_status()
        
        # Step 2: Calculate readiness score (60%)
        _update_check_progress(ui_components, 60, "📈 Menghitung skor kesiapan training...")
        readiness_score = get_dataset_readiness_score(dataset_status)
        
        # Step 3: Generate recommendations (80%)
        _update_check_progress(ui_components, 80, "💡 Menggenerate rekomendasi...")
        recommendations = _generate_recommendations(dataset_status, readiness_score)
        
        # Step 4: Display results (100%)
        _update_check_progress(ui_components, 95, "📋 Menyiapkan laporan...")
        _display_comprehensive_results(logger, dataset_status, readiness_score, recommendations)
        
        _complete_check_progress(ui_components, "Analisis dataset selesai")
        logger and logger.success("🎉 Analisis dataset selesai!")
        
    except Exception as e:
        logger and logger.error(f"❌ Check error: {str(e)}")
        _error_check_progress(ui_components, f"Check error: {str(e)}")

def _initialize_check_progress(ui_components: Dict[str, Any]) -> None:
    """Initialize progress tracking untuk check operation."""
    if 'show_for_operation' in ui_components:
        ui_components['show_for_operation']('check')
    _update_check_progress(ui_components, 0, "🔍 Memulai analisis dataset...")

def _update_check_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update check progress."""
    if 'update_progress' in ui_components:
        ui_components['update_progress']('overall', progress, message)

def _complete_check_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Complete check progress."""
    if 'complete_operation' in ui_components:
        ui_components['complete_operation'](f"✅ {message}")

def _error_check_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Set error state."""
    if 'error_operation' in ui_components:
        ui_components['error_operation'](message)

def _generate_recommendations(dataset_status: Dict[str, Any], readiness_score: Dict[str, Any]) -> list:
    """Generate actionable recommendations."""
    recommendations = []
    final_dataset = dataset_status['final_dataset']
    summary = dataset_status['summary']
    
    # Status-based recommendations
    if summary['status'] == 'empty':
        recommendations.append("📥 Download dataset terlebih dahulu")
    elif summary['status'] == 'downloaded':
        recommendations.append("📁 Organisir dataset ke struktur final")
    
    # Training readiness
    if readiness_score['overall_score'] < 60:
        recommendations.append(f"⚠️ Dataset belum siap training (Score: {readiness_score['overall_score']}/100)")
    elif readiness_score['overall_score'] >= 80:
        recommendations.append(f"🚀 Dataset siap untuk training (Score: {readiness_score['overall_score']}/100)")
    
    # Split-specific
    train_split = final_dataset['splits'].get('train', {})
    if not train_split.get('exists', False) or train_split.get('images', 0) < 100:
        recommendations.append("🏋️ Split training kurang dari minimum (100 gambar)")
    
    return recommendations

def _display_comprehensive_results(logger, dataset_status: Dict[str, Any], readiness_score: Dict[str, Any], recommendations: list) -> None:
    """Display comprehensive results."""
    if not logger: return
    
    # Overall status
    summary = dataset_status['summary']
    storage_type = dataset_status['storage_info']['type']
    
    logger.info("=" * 60)
    logger.info("📊 DATASET STATUS REPORT")
    logger.info("=" * 60)
    logger.info(f"📋 Status: {summary['message']}")
    logger.info(f"💾 Storage: {storage_type}")
    logger.info(f"🎯 Training Score: {readiness_score['overall_score']}/100 ({readiness_score['readiness_level']})")
    
    # Dataset structure
    final_dataset = dataset_status['final_dataset']
    if final_dataset['exists']:
        logger.info("")
        logger.info("📁 STRUKTUR DATASET")
        logger.info("-" * 40)
        logger.success(f"✅ Dataset Final: {final_dataset['total_images']} gambar, {final_dataset['total_labels']} label")
        
        for split in ['train', 'valid', 'test']:
            split_info = final_dataset['splits'][split]
            if split_info['exists'] and split_info['images'] > 0:
                logger.info(f"   📂 {split.title()}: {split_info['images']} gambar ({split_info['labels']} label)")
    
    # Recommendations
    if recommendations:
        logger.info("")
        logger.info("💡 REKOMENDASI")
        logger.info("-" * 40)
        for rec in recommendations:
            logger.info(f"   {rec}")

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs."""
    try:
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
    except: pass