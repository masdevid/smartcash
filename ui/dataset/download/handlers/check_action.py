"""
File: smartcash/ui/dataset/download/handlers/check_action.py
Deskripsi: Check action dengan tqdm progress tracking dan comprehensive analysis
"""

from typing import Dict, Any
from smartcash.ui.dataset.download.utils.button_state_manager import get_button_state_manager
from smartcash.ui.dataset.download.utils.dataset_checker import check_complete_dataset_status, get_dataset_readiness_score

def execute_check_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Check dataset dengan comprehensive analysis dan tqdm progress."""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    with button_manager.operation_context('check'):
        try:
            logger and logger.info("🔍 Memulai analisis komprehensif dataset")
            
            _clear_ui_outputs(ui_components)
            
            # Step 1: Analisis struktur dataset
            _update_check_progress(ui_components, 20, "📊 Menganalisis struktur dataset...")
            dataset_status = check_complete_dataset_status()
            
            # Step 2: Calculate readiness score
            _update_check_progress(ui_components, 60, "📈 Menghitung skor kesiapan training...")
            readiness_score = get_dataset_readiness_score(dataset_status)
            
            # Step 3: Generate recommendations
            _update_check_progress(ui_components, 80, "💡 Menggenerate rekomendasi...")
            recommendations = _generate_actionable_recommendations(dataset_status, readiness_score)
            
            # Step 4: Display comprehensive results
            _update_check_progress(ui_components, 95, "📋 Menyiapkan laporan...")
            _display_enhanced_results(ui_components, dataset_status, readiness_score, recommendations)
            
            _update_check_progress(ui_components, 100, "✅ Analisis selesai")
            
        except Exception as e:
            logger and logger.error(f"❌ Error check: {str(e)}")
            raise