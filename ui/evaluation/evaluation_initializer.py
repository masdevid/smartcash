"""
File: smartcash/ui/evaluation/evaluation_initializer.py
Deskripsi: Initializer untuk model evaluation dengan checkpoint selection dan testing pada data raw
"""

from typing import Dict, Any, List
import ipywidgets as widgets
from IPython.display import display

from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.ui_logger_namespace import EVALUATION_LOGGER_NAMESPACE, KNOWN_NAMESPACES
from smartcash.common.environment import get_environment_manager
from smartcash.ui.evaluation.components.evaluation_form import create_evaluation_form
from smartcash.ui.evaluation.components.evaluation_layout import create_evaluation_layout
from smartcash.ui.evaluation.handlers.checkpoint_handler import setup_checkpoint_handlers
from smartcash.ui.evaluation.handlers.evaluation_handler import setup_evaluation_handlers
from smartcash.ui.evaluation.handlers.metrics_handler import setup_metrics_handlers

# Gunakan logger dari utils
MODULE_LOGGER_NAME = KNOWN_NAMESPACES.get(EVALUATION_LOGGER_NAMESPACE, 'evaluation')
logger = None  # Akan diinisialisasi nanti jika diperlukan

class EvaluationInitializer(CommonInitializer):
    """Initializer untuk model evaluation dengan Common Initializer pattern"""
    
    def __init__(self):
        super().__init__(module_name=MODULE_LOGGER_NAME, logger_namespace=EVALUATION_LOGGER_NAMESPACE)
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat komponen UI untuk evaluation dengan one-liner style"""
        form_components = create_evaluation_form(config)
        layout_components = create_evaluation_layout(form_components, config)
        
        # Merge semua components dengan one-liner
        return {**form_components, **layout_components, 
                'config': config, 'env': env}
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers untuk checkpoint selection, evaluation, dan metrics"""
        ui_components = setup_checkpoint_handlers(ui_components, config, env)
        ui_components = self._setup_handlers(ui_components, config, env)
        ui_components = setup_metrics_handlers(ui_components, config, env)
        return ui_components
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
        """Setup handlers untuk UI components"""
        from smartcash.ui.evaluation.handlers.evaluation_handler import setup_evaluation_handlers
        from smartcash.ui.evaluation.handlers.scenario_handler import setup_scenario_handlers
        
        # Setup scenario handlers terlebih dahulu
        setup_scenario_handlers(ui_components, config, env)
        
        # Setup evaluation handlers
        ui_components = setup_evaluation_handlers(ui_components, config, env)
        
        return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default config untuk evaluation"""
        return {
            'checkpoint': {
                'auto_select_best': True,
                'custom_checkpoint_path': '',
                'validation_metrics': ['mAP@0.5', 'mAP@0.5:0.95', 'precision', 'recall']
            },
            'test_data': {
                'test_folder': 'data/test',
                'apply_augmentation': True,
                'batch_size': 16,
                'image_size': 640,
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45
            },
            'evaluation': {
                'save_predictions': True,
                'save_metrics': True,
                'generate_confusion_matrix': True,
                'class_names': ['100', '500', '1000', '2000', '5000', '10000', '20000', '50000', '75000', '100000']
            },
            'scenario': {
                'selected_scenario': 'scenario_1',
                'save_to_drive': True,
                'drive_path': '/content/drive/MyDrive/SmartCash/evaluation_results',
                'test_folder': '/content/drive/MyDrive/SmartCash/dataset/test',
                'scenarios': {
                    'scenario_1': {
                        'name': 'Skenario 1: YOLOv5 Default (CSPDarknet) backbone dengan positional variation',
                        'description': 'Skenario ini mengevaluasi model YOLOv5 dengan backbone default (CSPDarknet) pada variasi posisi mata uang.',
                        'folder_name': 'scenario_1_cspdarknet_position',
                        'backbone': 'cspdarknet_s',
                        'augmentation_type': 'position'
                    },
                    'scenario_2': {
                        'name': 'Skenario 2: YOLOv5 Default (CSPDarknet) backbone dengan lighting variation',
                        'description': 'Skenario ini mengevaluasi model YOLOv5 dengan backbone default (CSPDarknet) pada variasi pencahayaan mata uang.',
                        'folder_name': 'scenario_2_cspdarknet_lighting',
                        'backbone': 'cspdarknet_s',
                        'augmentation_type': 'lighting'
                    },
                    'scenario_3': {
                        'name': 'Skenario 3: YOLOv5 dengan EfficientNet-B4 backbone dengan positional variation',
                        'description': 'Skenario ini mengevaluasi model YOLOv5 dengan backbone EfficientNet-B4 pada variasi posisi mata uang.',
                        'folder_name': 'scenario_3_efficientnet_position',
                        'backbone': 'efficientnet_b4',
                        'augmentation_type': 'position'
                    },
                    'scenario_4': {
                        'name': 'Skenario 4: YOLOv5 dengan EfficientNet-B4 backbone dengan lighting variation',
                        'description': 'Skenario ini mengevaluasi model YOLOv5 dengan backbone EfficientNet-B4 pada variasi pencahayaan mata uang.',
                        'folder_name': 'scenario_4_efficientnet_lighting',
                        'backbone': 'efficientnet_b4',
                        'augmentation_type': 'lighting'
                    }
                }
            },
            'output': {
                'results_folder': 'output/evaluation',
                'export_format': ['csv', 'json'],
                'visualize_results': True
            }
        }
    
    def _get_critical_components(self) -> List[str]:
        """Komponen critical untuk evaluation"""
        return ['main_container', 'checkpoint_selector', 'evaluate_button', 
                'metrics_table', 'log_output', 'progress_container']
                
    def _get_return_value(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Custom return value untuk modul evaluation - one-liner style"""
        return ui_components

def initialize_evaluation_ui(env=None, config=None, **kwargs) -> Dict[str, Any]:
    """Factory function untuk initialize evaluation UI dengan auto-display"""
    initializer = EvaluationInitializer()
    ui_components = initializer.initialize(env or get_environment_manager(), config, **kwargs)
    
    # Auto-display main container jika ada
    if 'main_container' in ui_components:
        display(ui_components['main_container'])
    
    return ui_components