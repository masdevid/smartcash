"""
Cell 1.0 - Project Setup
Sel ini menangani clone repository YOLOv5 dan SmartCash.
"""

import os
import sys
import subprocess
from pathlib import Path
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

# Create minimal UI components
def create_status_alert(message, alert_type='info', icon='ℹ️'):
    """Create simple status alert."""
    colors = {
        'info': ('#0c5460', '#d1ecf1'),
        'success': ('#155724', '#d4edda'),
        'warning': ('#856404', '#fff3cd'),
        'error': ('#721c24', '#f8d7da')
    }
    color, bg = colors.get(alert_type, colors['info'])
    
    return HTML(f"""
    <div style="padding: 10px; margin: 10px 0; background-color: {bg}; color: {color}; border-radius: 4px;">
        <div style="display: flex; align-items: center;">
            <div style="margin-right: 10px; font-size: 1.2em;">{icon}</div>
            <div>{message}</div>
        </div>
    </div>
    """)

# Clone function with simple output
def clone_repos(b):
    """Clone YOLOv5 and SmartCash repositories."""
    with output:
        clear_output()
        
        # Clone YOLOv5
        yolo_repo = "yolov5"
        yolo_success = False
        smartcash_success = False
        
        if Path(yolo_repo).exists():
            display(create_status_alert(f"⚠️ Repository {yolo_repo} sudah ada", "warning"))
            yolo_success = True
        else:
            display(create_status_alert("🔄 Cloning YOLOv5...", "info"))
            try:
                result = subprocess.run(
                    f"git clone https://github.com/ultralytics/yolov5.git",
                    shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                display(create_status_alert(f"✅ YOLOv5 berhasil di-clone", "success"))
                yolo_success = True
            except subprocess.CalledProcessError as e:
                display(create_status_alert(f"❌ Gagal clone YOLOv5: {e.stderr}", "error"))
        
        # Clone SmartCash if URL provided
        if smartcash_url.value:
            smartcash_repo = smartcash_url.value.split("/")[-1].replace(".git", "")
            if Path(smartcash_repo).exists():
                display(create_status_alert(f"⚠️ Repository {smartcash_repo} sudah ada", "warning"))
                smartcash_success = True
            else:
                display(create_status_alert("🔄 Cloning SmartCash...", "info"))
                try:
                    result = subprocess.run(
                        f"git clone {smartcash_url.value}",
                        shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    display(create_status_alert(f"✅ SmartCash berhasil di-clone", "success"))
                    smartcash_success = True
                except subprocess.CalledProcessError as e:
                    display(create_status_alert(f"❌ Gagal clone SmartCash: {e.stderr}", "error"))
        
        # Success message with restart instruction
        if yolo_success and smartcash_success:
            display(HTML("""
            <div style="padding: 15px; margin: 15px 0; background-color: #d4edda; color: #155724; border-radius: 4px; border-left: 4px solid #28a745;">
                <h3 style="margin-top: 0; color: #155724;">✅ Repository berhasil di-clone!</h3>
                <p><b>Penting:</b> Silakan restart runtime untuk menerapkan perubahan:</p>
                <ol>
                    <li>Klik menu <b>Runtime</b> di bagian atas notebook</li>
                    <li>Pilih <b>Restart runtime</b></li>
                    <li>Klik <b>Yes</b> pada konfirmasi</li>
                    <li>Setelah restart, jalankan cell berikutnya</li>
                </ol>
            </div>
            """))
            
            # Try to restart programmatically
            try:
                display(HTML("<p>Mencoba restart otomatis dalam 5 detik...</p>"))
                
                # For Google Colab environment
                import time
                time.sleep(5)  # Give user time to read message
                try:
                    import google.colab
                    import IPython
                    IPython.display.Javascript("google.colab.kernel.restart()")
                    display(HTML("<p style='color: #155724;'><b>✅ Restarting...</b></p>"))
                except:
                    pass  # Not in Colab or restart failed
            except:
                pass  # If anything fails, user will restart manually

# Create simple UI
title = HTML("<h1>🚀 SmartCash Project Setup</h1><p>Clone repository yang diperlukan untuk project</p>")

smartcash_url = widgets.Text(
    value='https://github.com/masdevid/smartcash.git',
    description='SmartCash URL:',
    style={'description_width': 'initial'},
    layout={'width': '60%', 'margin': '10px 0'}
)

clone_button = widgets.Button(
    description='Clone Repositories',
    button_style='primary',
    icon='download',
    layout={'margin': '10px 0'}
)

output = widgets.Output(layout={'width': '100%', 'border': '1px solid #ddd', 'min_height': '100px', 'margin': '10px 0', 'padding': '8px 4px'})

# Attach handler
clone_button.on_click(clone_repos)

# Display UI
display(title)
display(smartcash_url)
display(clone_button)
display(output)