# SmartCash Project Documentation - File Structure

## Models (`models/`)
- `baseline.py`: Base model implementation
- `yolov5_model.py`: YOLOv5 model implementation
- `detection_head.py`: Detection head architecture
- `losses.py`: Loss functions implementation

### Backbone Networks (`models/backbones/`)
- CSPDarknet implementation
- EfficientNet-B4 implementation
- Custom backbone adapters

### Feature Processing (`models/necks/`)
- FPN implementation
- PAN implementation

## Handlers (`handlers/`)
- `model_handler.py` (42KB): Model operations management
- `data_manager.py` (24KB): Dataset operations handling
- `checkpoint_handler.py` (24KB): Checkpoint management
- `evaluation_handler.py`: Model evaluation functionality
- `unified-preprocessing.py`: Preprocessing pipeline
- `multilayer_dataset_handler.py`: Multi-layer dataset management
- `research_scenario_handler.py`: Research experiment management
- `roboflow_handler.py`: Roboflow dataset integration
- `detection_handler.py`: Object detection operations
- `backbone_handler.py`: Backbone network management
- `base_evaluation_handler.py`: Base evaluation handler
- `evaluator.py`: Core evaluation functionality

### UI Handlers (`smartcash/ui_handlers/`)
- `config_handlers.py` (17KB): Configuration UI handling
- `model_handlers.py` (36KB): Model UI operations
- `repository_handlers.py` (10KB): Repository management
- `common_utils.py` (8.8KB): Common UI utilities
- `data_handlers.py` (18KB): Data UI operations
- `research_handlers.py` (18KB): Research UI management
- `directory_handlers.py` (13KB): Directory UI operations
- `training_execution_handlers.py` (34KB): Training execution
- `evaluation_handlers.py` (22KB): Evaluation UI handling
- `model_playground_handlers.py` (11KB): Model testing UI
- `augmentation_handlers.py` (11KB): Augmentation UI
- `training_pipeline_handlers.py` (13KB): Training pipeline UI
- `training_config_handlers.py` (13KB): Training configuration
- `dataset_handlers.py` (13KB): Dataset UI operations

## Utils (`utils/`)
- `config_manager.py`: Configuration management
- `environment_manager.py`: Environment setup
- `logger.py`: Logging system
- `memory_optimizer.py`: Memory optimization
- `debug_helper.py`: Debugging tools
- `preprocessing.py`: Data preprocessing
- `optimized-augmentation.py`: Data augmentation
- `enhanced-dataset-validator.py`: Dataset validation
- `enhanced-cache.py`: Caching system
- `coordinate_normalizer.py`: Coordinate normalization
- `model_visualizer.py` (61KB): Model visualization
- `visualization.py`: General visualization
- `experiment_tracker.py`: Experiment tracking
- `ui_utils.py`: UI utilities
- `training_pipeline.py`: Training pipeline
- `early_stopping.py`: Early stopping mechanism
- `metrics.py`: Performance metrics
- `polygon_metrics.py`: Polygon-based metrics
- `model_exporter.py`: Model export utilities
- `layer-config-manager.py`: Layer configuration

## UI Components (`ui_components/`)
- `model_components.py` (13KB): Model configuration UI
- `training_components.py` (18KB): Training interface
- `config_components.py` (13KB): Configuration interface
- `data_components.py`: Dataset management UI
- `dataset_components.py`: Dataset visualization
- `augmentation_components.py`: Augmentation controls
- `directory_components.py`: File system management
- `evaluation_components.py`: Evaluation interface
- `research_components.py`: Research experiment UI
- `model_playground_components.py`: Model testing UI
- `repository_components.py`: Version control UI

## Exceptions (`exceptions/`)
- `base.py`: Base exception classes
- `handler.py`: Exception handling system
