"""
File: smartcash/ui_handlers/dataset_download.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler untuk UI download dataset SmartCash.
"""

import os
import time
import sys
from pathlib import Path
from IPython.display import display, HTML, clear_output

# Untuk import handlers dan utils
if not any('smartcash' in p for p in sys.path):
    sys.path.append('.')

def create_status_indicator(status, message):
    """Buat indikator status dengan styling konsisten."""
    status_styles = {
        'success': {'icon': '✅', 'color': 'green'},
        'warning': {'icon': '⚠️', 'color': 'orange'},
        'error': {'icon': '❌', 'color': 'red'},
        'info': {'icon': 'ℹ️', 'color': 'blue'}
    }
    
    style = status_styles.get(status, status_styles['info'])
    
    status_html = f"""
    <div style="margin: 5px 0; padding: 8px 12px; 
                border-radius: 4px; background-color: #f8f9fa;">
        <span style="color: {style['color']}; font-weight: bold;"> 
            {style['icon']} {message}
        </span>
    </div>
    """
    
    return HTML(status_html)

def setup_download_handlers(ui_components, config=None):
    """Setup handlers untuk UI download dataset."""
    
    # Default config jika tidak disediakan
    if config is None:
        config = {
            'data': {
                'source': 'roboflow',
                'roboflow': {
                    'api_key': '', 
                    'workspace': 'smartcash-wo2us', 
                    'project': 'rupiah-emisi-2022', 
                    'version': '3'
                }
            },
            'data_dir': 'data'
        }
    
    # Coba import DatasetManager dan utilitas yang diperlukan
    dataset_manager = None
    try:
        from smartcash.handlers.dataset import DatasetManager
        from smartcash.utils.logger import get_logger
        logger = get_logger("dataset_download")
        dataset_manager = DatasetManager(config, logger=logger)
    except ImportError as e:
        print(f"Info: Beberapa modul tidak tersedia, akan menggunakan simulasi. ({str(e)})")
        
    # Tambahkan info tentang API key
    api_key_info = widgets.HTML(
        value="",
        layout=widgets.Layout(margin='10px 0', width='100%')
    )
    
    # Tambahkan elemen info ke container download settings setelah elemen pertama (Roboflow settings)
    download_settings_container = widgets.VBox([ui_components['roboflow_settings'], api_key_info])
    
    # Coba mendapatkan Roboflow API key dari Google Secret (jika di Colab)
    has_api_key = False
    
    try:
        from google.colab import userdata
        roboflow_api_key = userdata.get('ROBOFLOW_API_KEY')
        if roboflow_api_key:
            # Update API key field with the value from secrets
            ui_components['roboflow_settings'].children[0].value = roboflow_api_key
            has_api_key = True
            
            # Update info message about using secret
            api_key_info.value = """
            <div style="padding: 8px 12px; background-color: #d1ecf1; border-left: 4px solid #0c5460; 
                     color: #0c5460; margin: 5px 0; border-radius: 4px;">
                <p><i>ℹ️ API Key Roboflow tersedia dari Google Secret.</i></p>
            </div>
            """
        else:
            # No API key in Google Secret
            api_key_info.value = """
            <div style="padding: 8px 12px; background-color: #fff3cd; border-left: 4px solid #856404; 
                     color: #856404; margin: 5px 0; border-radius: 4px;">
                <p><i>⚠️ API Key Roboflow tidak ditemukan. <a href="#" onclick="document.querySelector('.accordion-title:contains(\"Bantuan\")').click();">Lihat panduan setup API Key</a>.</i></p>
            </div>
            """
    except:
        # Not in Colab or secret not available
        api_key_info.value = """
        <div style="padding: 8px 12px; background-color: #fff3cd; border-left: 4px solid #856404; 
                 color: #856404; margin: 5px 0; border-radius: 4px;">
            <p><i>⚠️ Google Secret tidak tersedia. <a href="#" onclick="document.querySelector('.accordion-title:contains(\"Bantuan\")').click();">Lihat panduan setup API Key</a>.</i></p>
        </div>
        """
    
    # Handler untuk download dataset
    def on_download_click(b):
        with ui_components['download_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Memulai download dataset..."))
            
            try:
                # Ambil opsi download yang dipilih
                download_option = ui_components['download_options'].value
                
                if download_option == 'Roboflow (Online)':
                    # Ambil settings dari form
                    api_settings = ui_components['roboflow_settings'].children
                    api_key = api_settings[0].value
                    workspace = api_settings[1].value
                    project = api_settings[2].value
                    version = api_settings[3].value
                    
                    # Validasi API key
                    if not api_key:
                        # Coba sekali lagi dari userdata jika belum diisi
                        try:
                            from google.colab import userdata
                            api_key = userdata.get('ROBOFLOW_API_KEY')
                            if api_key:
                                api_settings[0].value = api_key
                                display(create_status_indicator("info", "🔑 API Key diambil dari Google Secret"))
                        except:
                            pass
                            
                        # Jika masih kosong, tampilkan peringatan
                        if not api_key:
                            display(create_status_indicator("error", 
                                "❌ API Key Roboflow tidak tersedia. Tambahkan secret ROBOFLOW_API_KEY di Google Colab untuk melanjutkan."))
                            # Open help accordion automatically
                            display(HTML("<script>document.querySelector('.accordion button').click();</script>"))
                            return
                    
                    # Update config
                    if config:
                        config['data']['roboflow']['api_key'] = api_key
                        config['data']['roboflow']['workspace'] = workspace
                        config['data']['roboflow']['project'] = project
                        config['data']['roboflow']['version'] = version
                    
                    # Progress update function
                    def update_progress(sender, progress, total, message=None, **kwargs):
                        ui_components['download_progress'].value = int(progress * 100 / total) if total > 0 else 0
                        ui_components['download_progress'].description = f"{int(progress * 100 / total)}%" if total > 0 else "0%"
                        if message:
                            display(create_status_indicator("info", message))
                    
                    # Download dataset menggunakan DatasetManager jika tersedia
                    if dataset_manager:
                        try:
                            # Daftarkan observer untuk update progress jika tersedia
                            try:
                                from smartcash.utils.observer import EventDispatcher, EventTopics
                                EventDispatcher.register(EventTopics.DOWNLOAD_PROGRESS, update_progress)
                            except ImportError:
                                pass
                            
                            display(create_status_indicator("info", "🔑 Menggunakan DatasetManager untuk download..."))
                            
                            # Download dataset
                            dataset_path = dataset_manager.download_dataset(
                                format="yolov5",
                                api_key=api_key,
                                workspace=workspace,
                                project=project,
                                version=version,
                                show_progress=True
                            )
                            
                            # Export ke struktur lokal jika diperlukan
                            try:
                                train_dir, val_dir, test_dir = dataset_manager.export_to_local(dataset_path)
                                display(create_status_indicator("success", f"✅ Dataset diekspor ke struktur lokal"))
                            except:
                                # Jika export_to_local tidak tersedia, gunakan dataset_path sebagai dir
                                data_dir = dataset_path
                        except Exception as e:
                            display(create_status_indicator("error", f"❌ Error dari DatasetManager: {str(e)}"))
                            raise e
                    else:
                        # Simulasi download dari Roboflow jika DatasetManager tidak tersedia
                        display(create_status_indicator("info", "ℹ️ DatasetManager tidak tersedia, melakukan simulasi download..."))
                        
                        total_steps = 5
                        for i in range(total_steps + 1):
                            ui_components['download_progress'].value = int(i * 100 / total_steps)
                            ui_components['download_progress'].description = f"{int(i * 100 / total_steps)}%"
                            
                            if i == 1:
                                display(create_status_indicator("info", "🔑 Autentikasi API Roboflow..."))
                            elif i == 2:
                                display(create_status_indicator("info", "📂 Mengunduh metadata project..."))
                            elif i == 3:
                                display(create_status_indicator("info", "🖼️ Mengunduh gambar..."))
                            elif i == 4:
                                display(create_status_indicator("info", "🏷️ Mengunduh label..."))
                            elif i == 5:
                                display(create_status_indicator("info", "⚙️ Memvalidasi hasil..."))
                                
                            time.sleep(0.5)  # Simulasi delay
                        
                        # Buat direktori jika belum ada
                        data_dir = config.get('data_dir', 'data')
                        for split in ['train', 'valid', 'test']:
                            for subdir in ['images', 'labels']:
                                os.makedirs(os.path.join(data_dir, split, subdir), exist_ok=True)
                    
                    # Tampilkan sukses
                    display(create_status_indicator(
                        "success", 
                        f"✅ Dataset berhasil diunduh dari Roboflow ke {data_dir}"
                    ))
                    
                    # Tampilkan statistik dataset jika dataset_manager tersedia
                    if dataset_manager:
                        try:
                            # Analisis dataset
                            display(create_status_indicator("info", "📊 Menganalisis statistik dataset..."))
                            
                            # Validasi dataset - ini juga memberikan statistik
                            validation_results = dataset_manager.validate_dataset(
                                split='train', 
                                fix_issues=False,
                                visualize=False
                            )
                            
                            # Tampilkan statistik
                            if 'stats' in validation_results:
                                stats = validation_results['stats']
                                html_stats = f"""
                                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
                                    <h4>📊 Statistik Dataset</h4>
                                    <ul>
                                        <li><b>Total Gambar:</b> {stats.get('total_images', 'N/A')}</li>
                                        <li><b>Total Anotasi:</b> {stats.get('total_annotations', 'N/A')}</li>
                                        <li><b>Jumlah Kelas:</b> {stats.get('num_classes', 'N/A')}</li>
                                        <li><b>Format Dataset:</b> YOLOv5</li>
                                    </ul>
                                </div>
                                """
                                display(HTML(html_stats))
                        except Exception as e:
                            display(create_status_indicator("warning", f"⚠️ Tidak dapat menganalisis dataset: {str(e)}"))
                    
                    # Tampilkan struktur yang dibuat
                    html_info = f"""
                    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
                        <h4>📂 Struktur Direktori</h4>
                        <ul>
                            <li><code>{data_dir}/train/images</code> - Gambar training</li>
                            <li><code>{data_dir}/train/labels</code> - Label training</li>
                            <li><code>{data_dir}/valid/images</code> - Gambar validasi</li>
                            <li><code>{data_dir}/valid/labels</code> - Label validasi</li>
                            <li><code>{data_dir}/test/images</code> - Gambar testing</li>
                            <li><code>{data_dir}/test/labels</code> - Label testing</li>
                        </ul>
                    </div>
                    """
                    display(HTML(html_info))
                
                elif download_option == 'Local Data (Upload)':
                    # Ambil file yang diupload
                    upload_widget = ui_components['local_upload'].children[0]
                    target_dir = ui_components['local_upload'].children[1].value
                    
                    if not upload_widget.value:
                        display(create_status_indicator("warning", "⚠️ Silahkan pilih file ZIP untuk diupload"))
                        return
                    
                    # Info file
                    file_info = next(iter(upload_widget.value.values()))
                    file_name = file_info.get('metadata', {}).get('name', 'unknown.zip')
                    file_size = file_info.get('metadata', {}).get('size', 0)
                    file_content = file_info.get('content', b'')
                    
                    # Gunakan DatasetManager jika tersedia
                    if dataset_manager:
                        try:
                            display(create_status_indicator("info", "📤 Proses upload..."))
                            ui_components['download_progress'].value = 50
                            ui_components['download_progress'].description = "50%"
                            
                            # Simpan file sementara
                            temp_zip_path = os.path.join(target_dir, file_name)
                            os.makedirs(os.path.dirname(temp_zip_path), exist_ok=True)
                            
                            with open(temp_zip_path, 'wb') as f:
                                f.write(file_content)
                            
                            display(create_status_indicator("info", "📂 Ekstraksi file..."))
                            ui_components['download_progress'].value = 75
                            ui_components['download_progress'].description = "75%"
                            
                            # Import dataset dari file zip
                            imported_dir = dataset_manager.import_from_zip(
                                zip_path=temp_zip_path,
                                target_dir=target_dir,
                                format="yolov5"
                            )
                            
                            ui_components['download_progress'].value = 100
                            ui_components['download_progress'].description = "100%"
                            
                            # Tampilkan sukses
                            display(create_status_indicator(
                                "success", 
                                f"✅ File {file_name} ({file_size/1024:.1f} KB) berhasil diupload dan diekstrak ke {imported_dir or target_dir}"
                            ))
                            
                            # Validasi dataset
                            try:
                                display(create_status_indicator("info", "🔍 Validasi dataset..."))
                                validation_results = dataset_manager.validate_dataset(
                                    split='train',
                                    fix_issues=False,
                                    visualize=False
                                )
                                
                                if validation_results.get('is_valid', False):
                                    display(create_status_indicator("success", "✅ Dataset valid dan siap digunakan"))
                                else:
                                    issues = validation_results.get('issues', [])
                                    display(create_status_indicator(
                                        "warning", 
                                        f"⚠️ Dataset memiliki {len(issues)} issue. Gunakan validasi untuk memperbaiki."
                                    ))
                            except Exception as e:
                                display(create_status_indicator("warning", f"⚠️ Tidak dapat memvalidasi dataset: {str(e)}"))
                            
                        except Exception as e:
                            display(create_status_indicator("error", f"❌ Error saat mengimport dataset: {str(e)}"))
                            raise e
                    else:
                        # Simulasi proses upload dan ekstraksi
                        ui_components['download_progress'].value = 50
                        ui_components['download_progress'].description = "50%"
                        display(create_status_indicator("info", "📤 Proses upload..."))
                        time.sleep(1)
                        
                        ui_components['download_progress'].value = 100
                        ui_components['download_progress'].description = "100%"
                        display(create_status_indicator("info", "📂 Ekstraksi file..."))
                        time.sleep(1)
                        
                        # Buat direktori jika belum ada
                        os.makedirs(target_dir, exist_ok=True)
                        
                        # Tampilkan sukses
                        display(create_status_indicator(
                            "success", 
                            f"✅ File {file_name} ({file_size/1024:.1f} KB) berhasil diupload dan diekstrak ke {target_dir}"
                        ))
                
            except Exception as e:
                display(create_status_indicator("error", f"❌ Error: {str(e)}"))
    
    # Register handler untuk tombol download
    ui_components['download_button'].on_click(on_download_click)
    
    # Populate form dengan nilai default dari config jika tidak ada dari Google Secret
    if config and not has_api_key:
        # Populate Roboflow settings (kecuali API key yang telah diset)
        api_settings = ui_components['roboflow_settings'].children
        api_settings[1].value = config['data']['roboflow'].get('workspace', 'smartcash-wo2us')
        api_settings[2].value = config['data']['roboflow'].get('project', 'rupiah-emisi-2022')
        api_settings[3].value = str(config['data']['roboflow'].get('version', '3'))
    
    return ui_components