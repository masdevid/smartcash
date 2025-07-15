"""
File: smartcash/ui/cells/setup/cell_1_1_repository_clone.py
Deskripsi: Clone/update repository YOLOv5 dan SmartCash dengan pilihan branch
"""
import subprocess, os; from pathlib import Path; from IPython.display import display, HTML, clear_output; import ipywidgets as widgets
re_execute = lambda: get_ipython().run_cell_magic('javascript', '', 'Jupyter.notebook.execute_cell_and_select_next(this)')
def run_cmd(cmd): r = subprocess.run(cmd, shell=True, capture_output=True, text=True); return r.stdout if not r.returncode else (_ for _ in ()).throw(Exception(r.stderr))
def setup(click=False):
    b, btn, p, s = (widgets.ToggleButtons(options=['dev', 'main'], value='dev', layout={'width': '320px', 'margin': '0 10px 0 0'}), widgets.Button(description='Go', button_style='info', layout={'width': '80px', 'margin': '0 10px 0 0'}), widgets.FloatProgress(min=0, max=5, layout={'flex': '1', 'margin': '0 10px 0 0'}),widgets.HTML(value='<span style="color:#666">Ready</span>', layout={'width': '120px'}))
    c, o = (widgets.HBox([widgets.Label('🚀', layout={'padding': '0 10px'}), b, p, s, btn], layout={'width': '100%', 'display': 'flex', 'align_items': 'center', 'justify_content': 'space-between', 'padding': '10px 4px', 'border': '1px solid #ddd'}),widgets.Output())
    def on_click(_):
        with o:
            clear_output(); s.value, p.bar_style = '<span style="color:orange">Working...</span>', ''
            try:
                for i, cmd in enumerate(['pip uninstall smartcash -qy','rm -rf smartcash yolov5', f'git clone -b {b.value} https://github.com/masdevid/smartcash.git', 'cd smartcash && pip install -q -e .', 'git clone https://github.com/ultralytics/yolov5.git']):
                    run_cmd(cmd); p.value = i + 1
                p.bar_style, s.value = 'success', '<span style="color:green">✅ Done, Restart!</span>'
                re_execute()
            except:
                p.bar_style, s.value = 'danger', '<span style="color:red">❌ Error</span>'
    btn.on_click(on_click); display(widgets.VBox([c, o]))
    if (click): btn.click(); return
setup()