"""
Cell 1.1 - Repository Clone
Menangani proses clone/update repository YOLOv5 dan SmartCash.
"""

import subprocess
from pathlib import Path
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

def create_status_alert(msg, alert_type='info', icon=None):
    colors = {'info': ('#0c5460', '#d1ecf1', 'ℹ️'), 'success': ('#155724', '#d4edda', '✅'),
              'warning': ('#856404', '#fff3cd', '⚠️'), 'error': ('#721c24', '#f8d7da', '❌')}
    color, bg, default_icon = colors.get(alert_type, colors['info'])
    return HTML(f"""<div style="padding:10px;margin:10px;background-color:{bg};color:{color};
                     border-radius:4px;display:flex;align-items:center;"><div style="margin-right:10px;
                     font-size:1.2em;">{icon or default_icon}</div><div>{msg}</div></div>""")

def get_git_status(repo_path):
    if not Path(repo_path).exists(): return "not_exists"
    try:
        subprocess.run(["git", "fetch"], cwd=repo_path, check=True, capture_output=True, text=True)
        result = subprocess.run(["git", "status", "-uno"], cwd=repo_path, check=True, capture_output=True, text=True)
        return "behind" if "Your branch is behind" in result.stdout else "up_to_date" if "Your branch is up to date" in result.stdout else "unknown"
    except subprocess.CalledProcessError: return "status_error"

def update_repo(repo_path):
    try: return True, subprocess.run(["git", "pull"], cwd=repo_path, check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as e: return False, str(e)

def clone_or_update_repos(b):
    with output:
        clear_output()
        if not smartcash_url.value.strip(): return display(create_status_alert("⚠️ URL repository tidak boleh kosong", "warning"))
        display(HTML("<h2 style='margin-left:10px;'>🚀 Clone/Update Repository</h2>"))
        repos = [{"repo_url": "https://github.com/ultralytics/yolov5.git", "repo_path": "yolov5", "repo_name": "YOLOv5"},
                 {"repo_url": smartcash_url.value.strip(), "repo_path": smartcash_url.value.split("/")[-1].replace(".git", ""), "repo_name": "SmartCash"}]
        all_success = all(clone_or_update_single_repo(**repo) for repo in repos)
        if all_success:
            display(create_status_alert("✅ Semua repository berhasil diproses. Silakan restart runtime.", "success"))
            display(HTML("""<div style="padding:15px;margin:15px 10px;background-color:#f0f0f0;color:#000;
                             border-left:5px solid #2ecc71;border-radius:4px;"><h3 style="color:#000;">🔄 Langkah Selanjutnya</h3>
                             <ol><li>Klik <b>Runtime</b> di menu atas</li><li>Pilih <b>Restart runtime</b></li>
                             <li>Klik <b>Yes</b> pada konfirmasi</li></ol></div>"""))

def clone_or_update_single_repo(repo_url, repo_path, repo_name):
    repo_status = get_git_status(repo_path)
    try:
        if repo_status == "not_exists":
            subprocess.run(["git", "clone", repo_url], check=True, capture_output=True, text=True)
            display(create_status_alert(f"✅ {repo_name} berhasil di-clone", "success"))
        elif repo_status == "behind":
            success, msg = update_repo(repo_path)
            display(create_status_alert(f"✅ {repo_name} berhasil diupdate", "success") if success else
                  create_status_alert(f"❌ Gagal update {repo_name}: {msg}", "error"))
            return success
        elif repo_status == "up_to_date": display(create_status_alert(f"✅ {repo_name} sudah dalam versi terbaru", "success"))
        else: display(create_status_alert(f"⚠️ Status {repo_name} tidak dapat ditentukan", "warning"))
        return True
    except subprocess.CalledProcessError as e: display(create_status_alert(f"❌ Gagal memproses {repo_name}: {e.stderr}", "error")); return False

# UI Components
title = widgets.HTML("""<div style="background-color:#f8f9fa;padding:15px;border-radius:5px;border-left:5px solid #3498db;
                         margin-bottom:15px;"><h1 style="margin:0;color:#2c3e50;padding:10px;">🚀 Setup Proyek SmartCash</h1>
                         <p style="margin:5px 0;color:#7f8c8d;">Clone atau update repository yang diperlukan untuk proyek</p></div>""")
smartcash_url = widgets.Text(value='https://github.com/masdevid/smartcash.git', description='🔗 URL Repository:',
                             placeholder='Masukkan URL repository SmartCash', style={'description_width': 'initial'},
                             layout=widgets.Layout(width='100%', margin='10px 0'))
clone_button = widgets.Button(description='Clone/Update Repository', button_style='primary', icon='sync',
                              layout=widgets.Layout(width='100%', margin='10px 0'))
output = widgets.Output(layout=widgets.Layout(width='100%', border='1px solid #ddd', min_height='200px', margin='10px 0'))
clone_button.on_click(clone_or_update_repos)
display(widgets.VBox([title, smartcash_url, clone_button, output], layout=widgets.Layout(width='100%', padding='15px')))