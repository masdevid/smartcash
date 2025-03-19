"""
Cell 1.1 - Repository Clone
File: smartcash/ui/cells/setup/cell_1_1_repository_clone.py
Deskripsi: Clone/update repository YOLOv5 dan SmartCash dengan pilihan branch
"""

import subprocess
from pathlib import Path
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import os

if not os.path.exists('/content/drive/MyDrive'):
  from google.colab import drive
  drive.mount('/content/drive')

def create_alert(msg, type='info'): return HTML(f"""<div style="padding:10px;margin:10px;background-color:{'#d4edda' if type=='success' else '#fff3cd' if type=='warning' else '#f8d7da' if type=='error' else '#d1ecf1'};color:{'#155724' if type=='success' else '#856404' if type=='warning' else '#721c24' if type=='error' else '#0c5460'};border-radius:4px">{"✅ " if type=='success' else "⚠️ " if type=='warning' else "❌ " if type=='error' else "📘 "}{msg}</div>""")

def clone_or_update_repos(b):
    with output:
        clear_output()
        if not url_input.value.strip(): return display(create_alert("URL repository tidak boleh kosong", "warning"))
        display(HTML("<h2 style='margin:15px;'>🚀 Clone/Update Repository</h2>"))

        # YOLOv5 repository
        try:
            if not Path("yolov5").exists():
                subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"], check=True, capture_output=True)
                display(create_alert("YOLOv5 berhasil di-clone", "success"))
            else:
                subprocess.run(["git", "pull"], cwd="yolov5", check=True, capture_output=True)
                display(create_alert("YOLOv5 berhasil di-update", "success"))
        except Exception as e: display(create_alert(f"Gagal {'clone' if not Path('yolov5').exists() else 'update'} YOLOv5: {str(e)}", "error")); return

        # SmartCash repository
        sc_path, branch = url_input.value.split("/")[-1].replace(".git", ""), branch_dropdown.value
        branch_info = f" (branch: {branch})" if branch != "main" else ""
        try:
            if not Path(sc_path).exists():
                subprocess.run(["git", "clone"] + (["-b", branch] if branch != "main" else []) + [url_input.value.strip()], check=True, capture_output=True)
            else:
                if branch != "main": subprocess.run(["git", "checkout", branch], cwd=sc_path, check=True, capture_output=True)
                subprocess.run(["git", "pull"], cwd=sc_path, check=True, capture_output=True)
            display(create_alert(f"SmartCash{branch_info} berhasil di-{'clone' if not Path(sc_path).exists() else 'update'}", "success"))
        except Exception as e: display(create_alert(f"Gagal {'clone' if not Path(sc_path).exists() else 'update'} SmartCash{branch_info}: {str(e)}", "error")); return

        display(HTML("""<div style="padding:15px;margin:10px;background-color:#2c3e50;border-left:5px solid #2ecc71;"><h3>🔄 Langkah Selanjutnya</h3><ol><li>Klik <b>Runtime</b> di menu atas</li><li>Pilih <b>Restart runtime</b></li><li>Klik <b>Yes</b> pada konfirmasi</li></ol></div>"""))

# UI components with repo input and branch dropdown inline
title = widgets.HTML("""<div style="background:#f8f9fa;padding:15px;border-radius:5px;border-left:5px solid #3498db;margin-bottom:15px"><h1 style="margin:0;color:#2c3e50">🚀 Setup Proyek SmartCash</h1><p style="margin:5px 0;color:#7f8c8d">Clone atau update repository project</p></div>""")
url_input = widgets.Text(value='https://github.com/masdevid/smartcash.git', description='🔗 URL:', style={'description_width': 'initial'}, layout={'width': '70%', 'margin': '10px 0'})
branch_dropdown = widgets.Dropdown(options=['main', 'migration', 'dev'], value='main', description='🌿', style={'description_width': 'initial'}, layout={'width': '30%', 'margin': '10px 0'})
clone_button = widgets.Button(description='Clone/Update Repository', button_style='primary', icon='sync', layout={'width': '100%', 'margin': '10px 0'})
output = widgets.Output(layout={'width': '100%', 'border': '1px solid #ddd', 'min_height': '200px', 'margin': '10px 0'})

clone_button.on_click(clone_or_update_repos)
display(widgets.VBox([
    title,
    widgets.HBox([url_input, branch_dropdown]),
    clone_button,
    output
], layout={'width': '100%', 'padding': '15px'}))


# !pip install -q ipywidgets tqdm pyyaml #silent install top priority packages.