"""
File: smartcash/ui/evaluation/components/evaluation_layout.py
Deskripsi: Layout arrangement untuk evaluation UI dengan sections dan responsive design
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.header_utils import create_header, create_section_title
from smartcash.ui.utils.layout_utils import create_responsive_two_column, create_divider
from smartcash.ui.evaluation.components.evaluation_form import create_metrics_display
from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
from smartcash.ui.components.log_accordion import create_log_accordion

def create_evaluation_layout(form_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Buat layout untuk evaluation UI dengan flex layout dan mencegah horizontal scrollbar"""
    
    # Progress tracking container dengan dual level untuk overall dan step progress
    progress_tracker_obj = create_dual_progress_tracker(operation="Evaluation")
    
    # Membuat dictionary untuk backward compatibility
    progress_components = {
        'container': progress_tracker_obj.container,
        'progress_container': progress_tracker_obj.container,
        'status_widget': progress_tracker_obj.status_widget,
        'step_info_widget': progress_tracker_obj.step_info_widget,
        'tqdm_container': progress_tracker_obj.tqdm_container,
        'tracker': progress_tracker_obj,
        'show_container': progress_tracker_obj.show,
        'hide_container': progress_tracker_obj.hide,
        'show_for_operation': progress_tracker_obj.show,
        'update_overall': progress_tracker_obj.update_overall,
        'update_step': progress_tracker_obj.update_step,
        'update_current': progress_tracker_obj.update_current,
        'update_progress': progress_tracker_obj.update,
        'complete_operation': progress_tracker_obj.complete,
        'error_operation': progress_tracker_obj.error,
        'reset_all': progress_tracker_obj.reset
    }
    
    # Log accordion untuk evaluation logs
    log_components = create_log_accordion('evaluation', height='250px')
    
    # Metrics display components
    metrics_components = create_metrics_display()
    
    # Section 0: Scenario Selection dengan flex layout
    scenario_section = widgets.VBox([
        create_section_title("🧪 Skenario Pengujian"),
        form_components['scenario_dropdown'],
        form_components['scenario_description']
    ], layout=widgets.Layout(
        margin='10px 0', 
        padding='10px', 
        border='1px solid #e0e0e0', 
        border_radius='5px',
        overflow_x='hidden'
    ))
    
    # Section 1: Checkpoint Selection dengan flex layout
    # Gunakan komponen checkpoint_selector yang sudah dibuat
    checkpoint_section = widgets.VBox([
        create_section_title("🏆 Checkpoint Selection"),
        form_components['checkpoint_selector']
    ], layout=widgets.Layout(
        margin='10px 0',
        padding='10px',
        border='1px solid #e0e0e0',
        border_radius='5px',
        overflow_x='hidden'
    ))
    
    # Section 2: Test Configuration dengan flex layout yang lebih baik
    test_config_left = widgets.VBox([
        form_components['test_folder_text'],
        form_components['batch_size_slider']
    ], layout=widgets.Layout(flex='1 1 300px', min_width='250px', overflow_x='hidden', padding='5px'))
    
    test_config_right = widgets.VBox([
        form_components['image_size_dropdown'],
        form_components['confidence_slider'],
        form_components['iou_slider']
    ], layout=widgets.Layout(flex='1 1 300px', min_width='250px', overflow_x='hidden', padding='5px'))
    
    # Checkbox dikelompokkan terpisah agar rata kiri dan teks lengkap terlihat
    test_config_checkboxes = widgets.VBox([
        form_components['apply_augmentation_checkbox']
    ], layout=widgets.Layout(width='100%', margin='10px 0', padding='5px'))
    
    # Gunakan HBox dengan flex layout yang lebih baik untuk mencegah horizontal scrollbar
    test_config_container = widgets.HBox(
        [test_config_left, test_config_right],
        layout=widgets.Layout(
            display='flex',
            flex_flow='row wrap',
            justify_content='space-between',
            align_items='flex-start',
            width='100%',
            overflow_x='hidden',
            gap='10px'
        )
    )
    
    test_section = widgets.VBox([
        create_section_title("⚙️ Konfigurasi Testing"),
        test_config_container,
        test_config_checkboxes,
        widgets.VBox([
            form_components['save_predictions_checkbox'],
            form_components['save_metrics_checkbox']
        ], layout=widgets.Layout(margin='10px 0', width='100%'))
    ], layout=widgets.Layout(
        margin='10px 0',
        padding='10px',
        border='1px solid #e0e0e0',
        border_radius='5px',
        overflow_x='hidden'
    ))
    
    # Section 3: Evaluation Options dengan flex layout
    # Semua checkbox dikelompokkan dalam satu VBox agar rata kiri
    eval_options_checkboxes = widgets.VBox([
        form_components['confusion_matrix_checkbox'],
        form_components['visualize_results_checkbox'],
        form_components['inference_time_checkbox'],
        form_components['save_to_drive_checkbox']
    ], layout=widgets.Layout(width='100%', margin='5px 0'))
    
    # Drive path dipisahkan agar tidak terpengaruh oleh pengelompokan checkbox
    eval_options_drive = widgets.VBox([
        form_components['drive_path_text']
    ], layout=widgets.Layout(width='100%', margin='5px 0'))
    
    # Gunakan VBox untuk layout vertikal yang lebih konsisten
    eval_options_container = widgets.VBox(
        [eval_options_checkboxes, eval_options_drive],
        layout=widgets.Layout(
            width='100%',
            overflow_x='hidden'
        )
    )
    
    eval_options = widgets.VBox([
        create_section_title("📊 Opsi Evaluasi"),
        eval_options_container
    ], layout=widgets.Layout(
        margin='10px 0',
        padding='10px',
        border='1px solid #e0e0e0',
        border_radius='5px',
        overflow_x='hidden'
    ))
    
    # Section 4: Actions - Hapus save_reset buttons, hanya gunakan action buttons
    actions_section = widgets.VBox([
        create_section_title("🚀 Actions"),
        form_components['container']  # Action buttons container
    ], layout=widgets.Layout(
        margin='10px 0',
        padding='10px',
        border='1px solid #e0e0e0',
        border_radius='5px',
        overflow_x='hidden'
    ))
    
    # Section 5: Results Display dengan tabs
    results_tabs = widgets.Tab([
        metrics_components['metrics_table'],
        metrics_components['confusion_matrix_output'],
        metrics_components['predictions_output']
    ])
    
    results_tabs.set_title(0, "📈 Metrik Evaluasi")
    results_tabs.set_title(1, "🔥 Confusion Matrix")
    results_tabs.set_title(2, "🎯 Sample Predictions")
    
    results_section = widgets.VBox([
        create_section_title("📋 Hasil Evaluasi"),
        results_tabs
    ], layout=widgets.Layout(
        margin='15px 0',
        padding='10px',
        border='1px solid #e0e0e0',
        border_radius='5px',
        overflow_x='hidden'
    ))
    
    # Main layout dengan flex container
    main_content = widgets.VBox([
        scenario_section,
        checkpoint_section,
        test_section,
        eval_options,
        actions_section,
        progress_components['container'],
        results_section,
        log_components['log_accordion']
    ], layout=widgets.Layout(
        padding='15px',
        width='100%',
        max_width='1200px',  # Mencegah terlalu lebar pada layar besar
        margin='0 auto',     # Auto margin untuk centering
        overflow_x='hidden'  # Mencegah horizontal scrollbar
    ))
    
    # Main container dengan header
    main_container = widgets.VBox([
        create_header(
            "🧪 Model Evaluation", 
            "Evaluasi performa model dengan checkpoint terbaik dan testing pada data raw"
        ),
        main_content
    ], layout=widgets.Layout(
        width='100%',
        max_width='100%',
        overflow_x='hidden'  # Mencegah horizontal scrollbar
    ))
    
    # Return all layout components
    return {
        'main_container': main_container,
        'main_content': main_content,
        'scenario_section': scenario_section,
        'checkpoint_section': checkpoint_section,
        'test_section': test_section,
        'eval_options': eval_options,
        'actions_section': actions_section,
        'results_section': results_section,
        'results_tabs': results_tabs,
        **progress_components,
        **log_components,
        **metrics_components
    }