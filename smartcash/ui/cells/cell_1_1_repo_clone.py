"""
File: smartcash/ui/cells/setup/cell_1_1_repository_clone.py
Deskripsi: Clone/update repository YOLOv5 dan SmartCash dengan pilihan branch
"""
import subprocess, shutil, re; from pathlib import Path; from IPython.display import display, clear_output; import ipywidgets as w; from IPython import get_ipython

def run(cmd, cb=None, step=0, total=3):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    out = []
    for line in iter(p.stdout.readline, ''):
        out.append(line)
        if cb and '%' in line and ('Receiving objects:' in line or 'Resolving deltas:' in line):
            cb(int(re.search(r'(\d+)%', line).group(1)) / 100, step, total)
    p.wait()
    if p.returncode != 0: raise Exception(f"Command failed: {p.returncode}\n{''.join(out)}")
    return ''.join(out)

def setup(click=False):
    i, b, btn, p, s = (w.Label('🚀', layout={'padding':'0 10px'}),w.ToggleButtons(options=['dev','main'], value='dev', layout={'width':'320px','margin':'0 10px 0 0'}),w.Button(description='Go', button_style='info', layout={'width':'80px','margin':'0 10px 0 0'}),w.FloatProgress(min=0, max=1, layout={'flex':'1','margin':'0 10px 0 0'}),w.HTML(value='<span style="color:#666">Ready</span>', layout={'width':'120px'}))
    def update(pct, step, total): p.value = (step + min(pct, 1)) / total
    def on_click(_):
        with w.Output():
            clear_output(); s.value, p.bar_style = '<span style="color:orange">Working...</span>', ''
            try:
                if Path('smartcash').exists(): run('pip uninstall -y smartcash')
                [shutil.rmtree(d, ignore_errors=True) for d in ['smartcash', 'yolov5'] if Path(d).exists()]
                
                s.value = '<span style="color:orange">Cloning SmartCash...</span>'
                run(f'git clone --progress -b {b.value} https://github.com/masdevid/smartcash.git', update, 0)
                
                s.value = '<span style="color:orange">Installing SmartCash...</span>'
                run('cd smartcash && pip install -q -e .', None); update(1, 1, 3)
                
                s.value = '<span style="color:orange">Cloning YOLOv5...</span>'
                run('git clone --progress https://github.com/ultralytics/yolov5.git', update, 2)
                
                p.value, p.bar_style, s.value = 1, 'success', '<span style="color:green">✅ Done, Restart!</span>'
                get_ipython().run_cell_magic('javascript', '', 'Jupyter.notebook.execute_cell_and_select_next(this)')
            except Exception as e:
                p.bar_style, s.value = 'danger', f'<span style="color:red">❌ {str(e)}</span>'
    
    btn.on_click(on_click)
    display(w.HBox([i, b, p, s, btn], layout=w.Layout(padding='10px')))
    if click: btn.click()

setup()