
# File: smartcash/ui/dataset/augmentation/components/advanced_options_widget.py
def create_advanced_options_widget() -> Dict[str, Any]:
    """
    Buat widget UI murni untuk opsi lanjutan (tanpa logika bisnis).
    
    Returns:
        Dictionary berisi container dan mapping widget individual
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Parameter posisi
    fliplr = widgets.FloatSlider(
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.05,
        description='Flip Horizontal:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='95%')
    )
    
    degrees = widgets.IntSlider(
        value=15,
        min=0,
        max=45,
        step=5,
        description='Rotasi (°):',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        layout=widgets.Layout(width='95%')
    )
    
    translate = widgets.FloatSlider(
        value=0.15,
        min=0.0,
        max=0.5,
        step=0.05,
        description='Translasi:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='95%')
    )
    
    scale = widgets.FloatSlider(
        value=0.15,
        min=0.0,
        max=0.5,
        step=0.05,
        description='Skala:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='95%')
    )
    
    # Parameter pencahayaan
    hsv_h = widgets.FloatSlider(
        value=0.025,
        min=0.0,
        max=0.1,
        step=0.005,
        description='HSV Hue:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.3f',
        layout=widgets.Layout(width='95%')
    )
    
    hsv_s = widgets.FloatSlider(
        value=0.7,
        min=0.0,
        max=1.0,
        step=0.1,
        description='HSV Saturation:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=widgets.Layout(width='95%')
    )
    
    brightness = widgets.FloatSlider(
        value=0.3,
        min=0.0,
        max=1.0,
        step=0.1,
        description='Brightness:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=widgets.Layout(width='95%')
    )
    
    contrast = widgets.FloatSlider(
        value=0.3,
        min=0.0,
        max=1.0,
        step=0.1,
        description='Contrast:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=widgets.Layout(width='95%')
    )
    
    # Tab layout
    position_tab = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS['dark']}; margin: 5px 0;'>Parameter Posisi</h6>"),
        fliplr, degrees, translate, scale
    ])
    
    lighting_tab = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS['dark']}; margin: 5px 0;'>Parameter Pencahayaan</h6>"),
        hsv_h, hsv_s, brightness, contrast
    ])
    
    tabs = widgets.Tab(children=[position_tab, lighting_tab])
    tabs.set_title(0, "Posisi")
    tabs.set_title(1, "Pencahayaan")
    
    container = widgets.VBox([tabs], layout=widgets.Layout(margin='10px 0'))
    
    return {
        'container': container,
        'widgets': {
            'fliplr': fliplr,
            'degrees': degrees,
            'translate': translate,
            'scale': scale,
            'hsv_h': hsv_h,
            'hsv_s': hsv_s,
            'brightness': brightness,
            'contrast': contrast
        }
    }

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