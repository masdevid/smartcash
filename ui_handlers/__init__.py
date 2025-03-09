"""
File: smartcash/smartcash/ui_handlers/__init__.py
Author: Alfrida Sabar
Deskripsi: Package initialization untuk UI handlers dengan organisasi imports yang lebih baik
           dan menyelesaikan masalah circular dependency.
"""

# Common utilities
from smartcash.ui_handlers.common_utils import (
    memory_manager,
    is_colab,
    save_config,
    load_config,
    display_gpu_info,
    create_timestamp_filename,
    plot_metrics,
    validate_ui_components
)


# Data handling handlers
from smartcash.ui_handlers.data_handlers import (
    on_refresh_info_clicked,
    get_dataset_info,
    on_split_button_clicked,
    update_total_ratio,
    check_data_availability,
    visualize_batch,
    setup_dataset_info_handlers,
    setup_split_dataset_handlers
)

# Dataset handlers
from smartcash.ui_handlers.dataset_handlers import (
    on_download_button_clicked,
    on_cleanup_button_clicked,
    on_check_status_button_clicked,
    on_storage_option_change,
    setup_dataset_dirs,
    check_dataset_status,
    setup_dataset_handlers
)

# Directory management handlers
from smartcash.ui_handlers.directory_handlers import (
    setup_google_drive,
    create_directory_structure,
    create_symlinks,
    get_directory_tree,
    on_setup_button_clicked,
    on_drive_checkbox_changed,
    setup_directory_handlers
)

# Augmentation handlers
from smartcash.ui_handlers.augmentation_handlers import (
    on_augment_button_clicked,
    on_clean_button_clicked,
    refresh_dataset_stats,
    setup_augmentation_handlers
)

# Configuration handlers
from smartcash.ui_handlers.config_handlers import (
    update_config_from_ui,
    save_config_to_file,
    update_layer_info,
    on_save_config_button_clicked,
    on_layer_selection_change,
    setup_global_config_handlers,
    setup_pipeline_config_handlers,
    on_pipeline_save_button_clicked,
    on_pipeline_reload_button_clicked,
    on_data_source_change
)

# Model handlers
from smartcash.ui_handlers.model_handlers import (
    setup_gpu,
    on_initialize_model_clicked,
    on_visualize_model_clicked,
    on_list_checkpoints_clicked,
    on_cleanup_checkpoints_clicked,
    on_compare_checkpoints_clicked,
    on_mount_drive_clicked,
    on_check_memory_clicked,
    on_clear_memory_clicked,
    on_optimize_batch_size_clicked,
    on_clear_cache_clicked,
    on_verify_cache_clicked,
    on_export_format_change,
    on_export_button_clicked,
    setup_model_initialization_handlers,
    setup_model_visualizer_handlers,
    setup_checkpoint_manager_handlers,
    setup_model_optimization_handlers,
    setup_model_exporter_handlers
)

# Model playground handlers
from smartcash.ui_handlers.model_playground_handlers import (
    on_test_model_button_clicked,
    setup_model_playground_handlers
)

# Evaluation handlers
from smartcash.ui_handlers.evaluation_handlers import (
    init_ui,
    evaluate_model,
    evaluate_multiple_runs,
    visualize_evaluation_results,
    on_run_evaluation_button_clicked,
    setup_evaluation_handlers
)

# Research handlers
from smartcash.ui_handlers.research_handlers import (
    run_research_evaluation,
    visualize_research_results,
    on_run_button_clicked,
    load_existing_results,
    setup_research_handlers
)

# Training execution handlers
from smartcash.ui_handlers.training_execution_handlers import (
    TrainingMetricsTracker,
    metrics_callback,
    update_plot,
    update_metrics_table,
    update_status,
    run_training,
    backup_to_drive,
    on_start_button_clicked,
    run_training_thread,
    on_stop_button_clicked,
    setup_training_handlers
)

# Training pipeline handlers
from smartcash.ui_handlers.training_pipeline_handlers import (
    on_check_status_button_clicked,
    init_components,
    setup_training_pipeline_handlers
)

# Training config handlers
from smartcash.ui_handlers.training_config_handlers import (
    on_generate_name_button_clicked,
    on_save_config_button_clicked,
    on_show_lr_schedule_button_clicked,
    on_backbone_change,
    setup_training_config_handlers,
    simulate_lr_schedule
)

# Repository handlers
from smartcash.ui_handlers.repository_handlers import (
    setup_repository_handlers,
    on_clone_button_clicked,
    on_custom_repo_checkbox_changed
)

__all__ = [
    # Common utilities
    'memory_manager', 'is_colab', 'save_config', 'load_config',
    'display_gpu_info', 'create_timestamp_filename', 'plot_metrics', 'validate_ui_components',
    
    # Data handling
    'on_refresh_info_clicked', 'get_dataset_info', 'on_split_button_clicked',
    'update_total_ratio', 'check_data_availability', 'visualize_batch',
    'setup_dataset_info_handlers', 'setup_split_dataset_handlers',
    
    # Dataset
    'on_download_button_clicked', 'on_cleanup_button_clicked', 'on_check_status_button_clicked',
    'on_storage_option_change', 'setup_dataset_dirs', 'check_dataset_status',
    'setup_dataset_handlers',
    
    # Directory
    'setup_google_drive', 'create_directory_structure', 'create_symlinks',
    'get_directory_tree', 'on_setup_button_clicked', 'on_drive_checkbox_changed',
    'setup_directory_handlers',
    
    # Augmentation
    'on_augment_button_clicked', 'on_clean_button_clicked', 'refresh_dataset_stats',
    'setup_augmentation_handlers',
    
    # Config
    'update_config_from_ui', 'save_config_to_file', 'update_layer_info',
    'on_save_config_button_clicked', 'on_layer_selection_change',
    'setup_global_config_handlers', 'setup_pipeline_config_handlers',
    'on_pipeline_save_button_clicked', 'on_pipeline_reload_button_clicked',
    'on_data_source_change',
    
    # Model
    'setup_gpu', 'on_initialize_model_clicked', 'on_visualize_model_clicked',
    'on_list_checkpoints_clicked', 'on_cleanup_checkpoints_clicked', 'on_compare_checkpoints_clicked',
    'on_mount_drive_clicked', 'on_check_memory_clicked', 'on_clear_memory_clicked',
    'on_optimize_batch_size_clicked', 'on_clear_cache_clicked', 'on_verify_cache_clicked',
    'on_export_format_change', 'on_export_button_clicked', 'setup_model_initialization_handlers',
    'setup_model_visualizer_handlers', 'setup_checkpoint_manager_handlers',
    'setup_model_optimization_handlers', 'setup_model_exporter_handlers',
    
    # Model playground
    'on_test_model_button_clicked', 'setup_model_playground_handlers',
    
    # Evaluation
    'init_ui', 'evaluate_model', 'evaluate_multiple_runs', 'visualize_evaluation_results',
    'on_run_evaluation_button_clicked', 'setup_evaluation_handlers',
    
    # Research
    'run_research_evaluation', 'visualize_research_results', 'on_run_button_clicked',
    'load_existing_results', 'setup_research_handlers',
    
    # Training execution
    'TrainingMetricsTracker', 'metrics_callback', 'update_plot', 'update_metrics_table',
    'update_status', 'run_training', 'backup_to_drive', 'on_start_button_clicked',
    'run_training_thread', 'on_stop_button_clicked', 'setup_training_handlers',
    
    # Training pipeline
    'on_check_status_button_clicked', 'init_components', 'setup_training_pipeline_handlers',
    
    # Training config
    'on_generate_name_button_clicked', 'on_save_config_button_clicked',
    'on_show_lr_schedule_button_clicked', 'on_backbone_change', 'setup_training_config_handlers',
    'simulate_lr_schedule',
    
    # Repository
    'setup_repository_handlers', 'on_clone_button_clicked', 'on_custom_repo_checkbox_changed'
]