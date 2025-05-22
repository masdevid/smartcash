
# File: smartcash/ui/dataset/augmentation/components/augmentation_types_widget.py
def create_augmentation_types_widget() -> Dict[str, Any]:
    """
    Buat widget UI murni untuk jenis augmentasi dan target split (tanpa logika bisnis).
    
    Returns:
        Dictionary berisi container dan mapping widget individual
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Jenis augmentasi
    augmentation_types = widgets.SelectMultiple(
        options=[
            ('Combined: Kombinasi posisi dan pencahayaan (direkomendasikan)', 'combined'),
            ('Position: Variasi posisi seperti rotasi, flipping, dan scaling', 'position'),
            ('Lighting: Variasi pencahayaan seperti brightness, contrast dan HSV', 'lighting')
        ],
        value=['combined'],
        description='Jenis:',
        disabled=False,
        layout=widgets.Layout(width='95%', height='100px')
    )
    
    # Target split
    target_split = widgets.Dropdown(
        options=['train', 'valid', 'test'],
        value='train',
        description='Target Split:',
        disabled=False,
        layout=widgets.Layout(width='95%')
    )
    
    # Info split
    split_info = widgets.HTML(
        f"""
        <div style="padding: 5px; color: {COLORS['dark']};">
            <p><b>{ICONS['info']} Informasi Split:</b></p>
            <ul>
                <li><b>train</b>: Augmentasi pada data training (rekomendasi)</li>
                <li><b>valid</b>: Augmentasi pada data validasi (jarang diperlukan)</li>
                <li><b>test</b>: Augmentasi pada data testing (tidak direkomendasikan)</li>
            </ul>
        </div>
        """
    )
    
    container = widgets.VBox([
        widgets.HTML(f"<h5 style='color: {COLORS['dark']}; margin: 5px 0;'>{ICONS['augmentation']} Jenis Augmentasi & Target Split</h5>"),
        widgets.HBox([
            widgets.VBox([
                widgets.HTML(f"<h6 style='color: {COLORS['dark']}; margin: 5px 0;'>Pilih Jenis Augmentasi:</h6>"),
                augmentation_types
            ], layout=widgets.Layout(width='60%')),
            
            widgets.VBox([
                widgets.HTML(f"<h6 style='color: {COLORS['dark']}; margin: 5px 0;'>{ICONS['split']} Target Split:</h6>"),
                target_split,
                split_info
            ], layout=widgets.Layout(width='40%'))
        ], layout=widgets.Layout(width='100%'))
    ], layout=widgets.Layout(padding='10px', width='100%'))
    
    return {
        'container': container,
        'widgets': {
            'augmentation_types': augmentation_types,
            'target_split': target_split
        }
    }