from typing import Dict, Any

def get_default_evaluation_config() -> Dict[str, Any]:
    """Get default config untuk evaluation"""
    return {
        'checkpoint': {
            'auto_select_best': True,
            'custom_checkpoint_path': '',
            'validation_metrics': ['mAP@0.5', 'mAP@0.5:0.95', 'precision', 'recall', 'inference_time']
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