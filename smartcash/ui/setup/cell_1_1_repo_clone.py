"""
File: smartcash/ui/cells/setup/cell_1_1_repository_clone.py
Deskripsi: Clone/update repository YOLOv5 dan SmartCash dengan pilihan branch
"""

import subprocess, os; from pathlib import Path; from IPython.display import display, HTML, clear_output; import ipywidgets as widgets

# Install packages and mount Drive in one-liners
subprocess.run(["pip", "install", "-q", "ipywidgets", "tqdm", "pyyaml"], check=True)
if not os.path.exists('/content/drive/MyDrive'): from google.colab import drive; drive.mount('/content/drive')

def create_alert(msg, type='info'):
    styles = {'success': ('#d4edda', '#155724', '✅'), 'warning': ('#fff3cd', '#856404', '⚠️'), 'error': ('#f8d7da', '#721c24', '❌'), 'info': ('#d1ecf1', '#0c5460', '📘')}
    bg, text, icon = styles.get(type, styles['info']); return HTML(f"<div style='padding:10px;margin:10px;background-color:{bg};color:{text};border-radius:4px'>{icon} {msg}</div>")

def repo_action(repo_url, repo_path, branch=None, is_sc=False):
    path, name = Path(repo_path), 'SmartCash' if is_sc else 'YOLOv5'; suffix = f" (branch: {branch})" if is_sc and branch != "main" else ""
    try:
        cmd = ["git", "clone"] + (["-b", branch] if is_sc and branch != "main" else []) + [repo_url] if not path.exists() else ["git", "pull"]
        if path.exists() and is_sc and branch != "main": subprocess.run(["git", "checkout", branch], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(cmd, cwd=repo_path if path.exists() else None, check=True, capture_output=True); return True, f"{name}{suffix} berhasil di-{'clone' if not path.exists() else 'update'}"
    except subprocess.CalledProcessError as e: return False, f"Gagal {'clone' if not path.exists() else 'update'} {name}{suffix}: {e.stderr.decode()}"

def clone_or_update_repos(b):
    with output:
        clear_output()
        if not url_input.value.strip(): return display(create_alert("URL repository tidak boleh kosong", "warning"))
        display(HTML("<h2 style='margin:15px;'>🚀 Clone/Update Repository</h2>"))
        for repo in [("https://github.com/ultralytics/yolov5.git", "yolov5"), (url_input.value.strip(), url_input.value.split("/")[-1].replace(".git", ""), branch_dropdown.value, True)]:
            ok, msg = repo_action(*repo); display(create_alert(msg, "success" if ok else "error"))
            if not ok: return
        display(HTML("""<div style="padding:15px;margin:10px;background-color:#2c3e50;border-left:5px solid #2ecc71;color:white"><h3>🔄 Langkah Selanjutnya</h3><ol><li>Klik <b>Runtime</b> di menu atas</li><li>Pilih <b>Restart runtime</b></li><li>Klik <b>Yes</b> pada konfirmasi</li></ol></div>"""))

# UI setup with one-liner definitions
title = widgets.HTML("""<div style="background:#f8f9fa;padding:15px;border-radius:5px;border-left:5px solid #3498db;margin-bottom:15px"><h1 style="margin:0;color:#2F58CD">🚀 Setup Proyek SmartCash</h1><p style="margin:5px 0;color: black">Clone atau update repository project</p></div>""")
url_input = widgets.Text(value='https://github.com/masdevid/smartcash.git', description='🔗 URL:', style={'description_width': 'initial'}, layout={'width': '70%', 'margin': '10px 0'})
branch_dropdown = widgets.Dropdown(options=['main', 'migration', 'dev'], value='main', description='🌿', style={'description_width': 'initial'}, layout={'width': '30%', 'margin': '10px 0'})
clone_button = widgets.Button(description='Clone/Update Repository', button_style='primary', icon='sync', layout={'width': '100%', 'margin': '10px 0'}); clone_button.on_click(clone_or_update_repos)
output = widgets.Output(layout={'width': '100%', 'border': '1px solid #ddd', 'min_height': '200px', 'margin': '10px 0'})
display(widgets.VBox([title, widgets.HBox([url_input, branch_dropdown]), clone_button, output], layout={'width': '100%', 'padding': '15px'}))