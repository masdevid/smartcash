"""
File: smartcash/ui/evaluation/evaluation_initializer.py
Deskripsi: Initializer untuk model evaluation dengan checkpoint selection dan testing pada data raw
"""

from typing import Dict, Any, List, Optional, Type
import ipywidgets as widgets
from IPython.display import display

from smartcash.ui.initializers.common_initializer import CommonInitializer, create_common_initializer
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.ui_logger_namespace import EVALUATION_LOGGER_NAMESPACE, KNOWN_NAMESPACES
from smartcash.common.environment import get_environment_manager
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.evaluation.components.evaluation_form import create_evaluation_form
from smartcash.ui.evaluation.components.evaluation_layout import create_evaluation_layout
from smartcash.ui.evaluation.handlers.checkpoint_handler import setup_checkpoint_handlers
from smartcash.ui.evaluation.handlers.evaluation_handler import setup_evaluation_handlers
from smartcash.ui.evaluation.handlers.metrics_handler import setup_metrics_handlers

# Gunakan logger dari utils
MODULE_LOGGER_NAME = KNOWN_NAMESPACES.get(EVALUATION_LOGGER_NAMESPACE, 'evaluation')
logger = None  # Akan diinisialisasi nanti jika diperlukan

class EvaluationConfigHandler(ConfigHandler):
    """Config handler untuk evaluation dengan fixed implementation"""
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components"""
        config = {}
        
        # Ekstrak checkpoint config
        checkpoint_config = {}
        if 'auto_select_best' in ui_components:
            checkpoint_config['auto_select_best'] = ui_components['auto_select_best'].value
        if 'custom_checkpoint_path' in ui_components:
            checkpoint_config['custom_checkpoint_path'] = ui_components['custom_checkpoint_path'].value
        if 'validation_metrics' in ui_components:
            checkpoint_config['validation_metrics'] = ui_components['validation_metrics'].value
        
        # Ekstrak test data config
        test_data_config = {}
        if 'test_folder' in ui_components:
            test_data_config['test_folder'] = ui_components['test_folder'].value
        if 'apply_augmentation' in ui_components:
            test_data_config['apply_augmentation'] = ui_components['apply_augmentation'].value
        if 'batch_size' in ui_components:
            test_data_config['batch_size'] = ui_components['batch_size'].value
        if 'image_size' in ui_components:
            test_data_config['image_size'] = ui_components['image_size'].value
        if 'confidence_threshold' in ui_components:
            test_data_config['confidence_threshold'] = ui_components['confidence_threshold'].value
        if 'iou_threshold' in ui_components:
            test_data_config['iou_threshold'] = ui_components['iou_threshold'].value
        
        # Combine configs
        config['checkpoint'] = checkpoint_config
        config['test_data'] = test_data_config
        
        return config
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        # Update checkpoint config
        checkpoint_config = config.get('checkpoint', {})
        if 'auto_select_best' in ui_components and 'auto_select_best' in checkpoint_config:
            ui_components['auto_select_best'].value = checkpoint_config['auto_select_best']
        if 'custom_checkpoint_path' in ui_components and 'custom_checkpoint_path' in checkpoint_config:
            ui_components['custom_checkpoint_path'].value = checkpoint_config['custom_checkpoint_path']
        if 'validation_metrics' in ui_components and 'validation_metrics' in checkpoint_config:
            ui_components['validation_metrics'].value = checkpoint_config['validation_metrics']
        
        # Update test data config
        test_data_config = config.get('test_data', {})
        if 'test_folder' in ui_components and 'test_folder' in test_data_config:
            ui_components['test_folder'].value = test_data_config['test_folder']
        if 'apply_augmentation' in ui_components and 'apply_augmentation' in test_data_config:
            ui_components['apply_augmentation'].value = test_data_config['apply_augmentation']
        if 'batch_size' in ui_components and 'batch_size' in test_data_config:
            ui_components['batch_size'].value = test_data_config['batch_size']
        if 'image_size' in ui_components and 'image_size' in test_data_config:
            ui_components['image_size'].value = test_data_config['image_size']
        if 'confidence_threshold' in ui_components and 'confidence_threshold' in test_data_config:
            ui_components['confidence_threshold'].value = test_data_config['confidence_threshold']
        if 'iou_threshold' in ui_components and 'iou_threshold' in test_data_config:
            ui_components['iou_threshold'].value = test_data_config['iou_threshold']
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config untuk evaluation"""
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

class EvaluationInitializer(CommonInitializer):
    """Initializer untuk model evaluation dengan Common Initializer pattern"""
    
    def __init__(self, module_name: str = MODULE_LOGGER_NAME, config_handler_class: Optional[Type[ConfigHandler]] = None, 
                 parent_module: Optional[str] = None):
        super().__init__(module_name, config_handler_class or EvaluationConfigHandler, parent_module)
    
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
        # Menggunakan config handler untuk mendapatkan default config
        config_handler = self._create_config_handler()
        return config_handler.get_default_config()
    
    def _get_critical_components(self) -> List[str]:
        """Komponen critical untuk evaluation"""
        return ['main_container', 'checkpoint_selector', 'evaluate_button', 
                'metrics_table', 'log_output', 'progress_container']
                
    def _get_return_value(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Custom return value untuk modul evaluation - one-liner style"""
        return ui_components

def initialize_evaluation_ui(env=None, config=None, parent_callbacks=None, **kwargs) -> Dict[str, Any]:
    """Factory function untuk initialize evaluation UI dengan auto-display dan parent module support"""
    initializer = EvaluationInitializer()
    ui_components = initializer.initialize(env or get_environment_manager(), config, parent_callbacks=parent_callbacks, **kwargs)
    return ui_components