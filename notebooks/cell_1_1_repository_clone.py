"""
Cell 1.1 - Repository Clone
Sel ini menangani clone/update repository YOLOv5 dan SmartCash.
"""

import os
import sys
import subprocess
from pathlib import Path
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import time

def create_status_alert(message, alert_type='info', icon='ℹ️'):
    """Create simple status alert dengan styling konsisten."""
    colors = {
        'info': ('#0c5460', '#d1ecf1', 'ℹ️'),
        'success': ('#155724', '#d4edda', '✅'),
        'warning': ('#856404', '#fff3cd', '⚠️'),
        'error': ('#721c24', '#f8d7da', '❌')
    }
    color, bg, default_icon = colors.get(alert_type, colors['info'])
    icon = icon or default_icon
    
    return HTML(f"""
    <div style="padding: 10px; margin: 10px; background-color: {bg}; color: {color}; 
                border-radius: 4px; display: flex; align-items: center;">
        <div style="margin-right: 10px; font-size: 1.2em;">{icon}</div>
        <div>{message}</div>
    </div>
    """)

def get_git_status(repo_path):
    """Periksa status repository git."""
    if not Path(repo_path).exists():
        return "not_exists"
        
    try:
        # Fetch latest updates
        subprocess.run(
            f"cd {repo_path} && git fetch",
            shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        
        # Cek status
        result = subprocess.run(
            f"cd {repo_path} && git status -uno",
            shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        
        if "Your branch is behind" in result.stdout:
            return "behind"
        elif "Your branch is up to date" in result.stdout:
            return "up_to_date"
        else:
            return "unknown"
    except subprocess.CalledProcessError:
        return "status_error"

def update_repo(repo_path):
    """Update repository git."""
    try:
        result = subprocess.run(
            f"cd {repo_path} && git pull",
            shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, str(e)

def clone_or_update_repos(b):
    """Clone atau update YOLOv5 dan SmartCash repositories."""
    with output:
        clear_output()
        
        # Validasi URL
        if not smartcash_url.value.strip():
            display(create_status_alert("⚠️ URL repository tidak boleh kosong", "warning"))
            return
        
        # Header
        display(HTML("""<h2 style="margin-left: 10px;">🚀 Repository Clone/Update</h2>"""))
        
        # Proses YOLOv5
        display(create_status_alert("🔄 Memproses Repository YOLOv5", "info"))
        yolo_repo = "yolov5"
        yolo_success = clone_or_update_single_repo(
            repo_url="https://github.com/ultralytics/yolov5.git",
            repo_path=yolo_repo,
            repo_name="YOLOv5"
        )
        
        # Proses SmartCash
        display(create_status_alert("🔄 Memproses Repository SmartCash", "info"))
        smartcash_repo = smartcash_url.value.split("/")[-1].replace(".git", "")
        smartcash_success = clone_or_update_single_repo(
            repo_url=smartcash_url.value,
            repo_path=smartcash_repo,
            repo_name="SmartCash"
        )
        
        # Kesimpulan
        if yolo_success and smartcash_success:
            display(create_status_alert(
                "✅ Semua repository berhasil diproses. Silakan restart runtime.", 
                "success"
            ))
            
            # Tampilkan instruksi restart
            display(HTML("""
            <div style="padding: 15px; margin: 15px 10px; 
                        background-color: #f0f0f0;
                        color: #000;
                        border-left: 5px solid #2ecc71; 
                        border-radius: 4px;">
                <h3 style="color: #000;">🔄 Langkah Selanjutnya</h3>
                <ol>
                    <li>Klik <b>Runtime</b> di menu atas</li>
                    <li>Pilih <b>Restart runtime</b></li>
                    <li>Klik <b>Yes</b> pada konfirmasi</li>
                </ol>
            </div>
            """))

def clone_or_update_single_repo(repo_url, repo_path, repo_name):
    """Proses clone atau update untuk satu repository."""
    repo_status = get_git_status(repo_path)
    
    try:
        if repo_status == "not_exists":
            # Clone repository
            result = subprocess.run(
                f"git clone {repo_url}",
                shell=True, check=True, 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            display(create_status_alert(f"✅ {repo_name} berhasil di-clone", "success"))
            return True
        
        elif repo_status == "behind":
            # Update repository
            success, message = update_repo(repo_path)
            if success:
                display(create_status_alert(f"✅ {repo_name} berhasil diupdate", "success"))
                return True
            else:
                display(create_status_alert(f"❌ Gagal update {repo_name}: {message}", "error"))
                return False
        
        elif repo_status == "up_to_date":
            display(create_status_alert(f"✅ {repo_name} sudah dalam versi terbaru", "success"))
            return True
        
        else:
            display(create_status_alert(
                f"⚠️ Status {repo_name} tidak dapat ditentukan", "warning"
            ))
            return True
    
    except subprocess.CalledProcessError as e:
        display(create_status_alert(
            f"❌ Gagal memproses {repo_name}: {e.stderr}", "error"
        ))
        return False

# UI Components dengan layout yang lebih konsisten
title = widgets.HTML("""
<div style="background-color: #f8f9fa; 
            padding: 15px; 
            border-radius: 5px; 
            border-left: 5px solid #3498db; 
            margin-bottom: 15px;">
    <h1 style="margin: 0; color: #2c3e50; padding: 10px;">🚀 SmartCash Project Setup</h1>
    <p style="margin: 5px 0; color: #7f8c8d;">
        Clone atau update repository yang diperlukan untuk project
    </p>
</div>
""")

smartcash_url = widgets.Text(
    value='https://github.com/masdevid/smartcash.git',
    description='🔗 Repository URL:',
    placeholder='Masukkan URL repository SmartCash',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='100%', margin='10px 0')
)

clone_button = widgets.Button(
    description='Clone/Update Repository',
    button_style='primary',
    icon='sync',
    layout=widgets.Layout(width='100%', margin='10px 0')
)

output = widgets.Output(
    layout=widgets.Layout(
        width='100%', 
        border='1px solid #ddd', 
        min_height='200px', 
        margin='10px 0', 
    )
)

# Attach handler
clone_button.on_click(clone_or_update_repos)

# Setup layout
container = widgets.VBox([
    title, 
    smartcash_url, 
    clone_button, 
    output
], layout=widgets.Layout(width='100%', padding='15px'))

# Display UI
display(container)