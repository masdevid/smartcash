# Opsi tambahan
with ui.GridBox(n_columns=2) as checkbox_grid:
    ui_components['validate_results_checkbox'] = ui.Checkbox(
        value=True,
        description='Validasi Hasil',
        style=validation_style
    )
    
    ui_components['process_bboxes_checkbox'] = ui.Checkbox(
        value=True,
        description='Proses Bounding Box',
        style=bbox_style
    )
    
    ui_components['move_to_preprocessed_checkbox'] = ui.Checkbox(
        value=True,
        description='Buat Symlink ke Preprocessed',
        style=move_style
    )
    
    ui_components['balance_classes_checkbox'] = ui.Checkbox(
        value=True,
        description='Balancing Class',
        style=balance_style
    ) 