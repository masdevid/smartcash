"""
File: smartcash/ui/cells/setup/cell_1_1_repository_clone.py
Deskripsi: Clone/update repository YOLOv5 dan SmartCash dengan pilihan branch
"""
# !pip install -q ipywidgets tqdm pyyaml  # silent install packages

import subprocess, os
from pathlib import Path
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

# Mount Drive & helper functions
if not os.path.exists('/content/drive/MyDrive'): from google.colab import drive; drive.mount('/content/drive')
def create_alert(msg, type='info'): return HTML(f"""<div style="padding:10px;margin:10px;background-color:{'#d4edda' if type=='success' else '#fff3cd' if type=='warning' else '#f8d7da' if type=='error' else '#d1ecf1'};color:{'#155724' if type=='success' else '#856404' if type=='warning' else '#721c24' if type=='error' else '#0c5460'};border-radius:4px">{"✅ " if type=='success' else "⚠️ " if type=='warning' else "❌ " if type=='error' else "📘 "}{msg}</div>""")

def repo_action(repo_url, repo_path, branch=None, is_sc=False):
    path, suffix = Path(repo_path), f" (branch: {branch})" if is_sc and branch != "main" else ""
    name = 'SmartCash' if is_sc else 'YOLOv5'
    
    if not path.exists():
        try: 
            subprocess.run(["git", "clone"] + (["-b", branch] if is_sc and branch != "main" else []) + [repo_url], check=True, capture_output=True)
            return True, f"{name}{suffix} berhasil di-clone"
        except: return False, f"Gagal clone {name}{suffix}"
    else:
        try: 
            if is_sc and branch != "main": subprocess.run(["git", "checkout", branch], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(["git", "pull"], cwd=repo_path, check=True, capture_output=True)
            return True, f"{name}{suffix} berhasil di-update"
        except: return False, f"Gagal update {name}{suffix}"

def clone_or_update_repos(b):
    with output:
        clear_output()
        if not url_input.value.strip(): return display(create_alert("URL repository tidak boleh kosong", "warning"))
        display(HTML("<h2 style='margin:15px;'>🚀 Clone/Update Repository</h2>"))
        
        # Process repos
        yolo_ok, yolo_msg = repo_action("https://github.com/ultralytics/yolov5.git", "yolov5")
        display(create_alert(yolo_msg, "success" if yolo_ok else "error"))
        if not yolo_ok: return
        
        sc_path = url_input.value.split("/")[-1].replace(".git", "")
        sc_ok, sc_msg = repo_action(url_input.value.strip(), sc_path, branch_dropdown.value, True)
        display(create_alert(sc_msg, "success" if sc_ok else "error"))
        if not sc_ok: return
        
        display(HTML("""<div style="padding:15px;margin:10px;background-color:#2c3e50;border-left:5px solid #2ecc71;color:white"><h3>🔄 Langkah Selanjutnya</h3><ol><li>Klik <b>Runtime</b> di menu atas</li><li>Pilih <b>Restart runtime</b></li><li>Klik <b>Yes</b> pada konfirmasi</li></ol></div>"""))

# UI components
title = widgets.HTML("""<div style="background:#f8f9fa;padding:15px;border-radius:5px;border-left:5px solid #3498db;margin-bottom:15px"><h1 style="margin:0;color:#2F58CD">🚀 Setup Proyek SmartCash</h1><p style="margin:5px 0;color: black">Clone atau update repository project</p></div>""")
url_input = widgets.Text(value='https://github.com/masdevid/smartcash.git', description='🔗 URL:', style={'description_width': 'initial'}, layout={'width': '70%', 'margin': '10px 0'})
branch_dropdown = widgets.Dropdown(options=['main', 'migration', 'dev'], value='main', description='🌿', style={'description_width': 'initial'}, layout={'width': '30%', 'margin': '10px 0'})
clone_button = widgets.Button(description='Clone/Update Repository', button_style='primary', icon='sync', layout={'width': '100%', 'margin': '10px 0'})
output = widgets.Output(layout={'width': '100%', 'border': '1px solid #ddd', 'min_height': '200px', 'margin': '10px 0'})

clone_button.on_click(clone_or_update_repos)
display(widgets.VBox([title, widgets.HBox([url_input, branch_dropdown]), clone_button, output], layout={'width': '100%', 'padding': '15px'}))
def err_alert(msg): display(alert_error = HTML(f"<div style='padding:10px; background:#f8d7da; color:#721c24; border-radius:5px'><h3 style='color:inherit'>❌ Error Inisialisasi</h3><p>{str(e)}</p></div>"))