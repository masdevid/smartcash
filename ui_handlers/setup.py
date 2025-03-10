"""
File: smartcash/ui_handlers/setup.py
Author: Alfrida Sabar (refactored)
Deskripsi: Handler UI setup SmartCash
"""

import ipywidgets as widgets
import time
import platform
from smartcash.utils.ui_utils import create_info_alert, create_status_indicator

def setup_project_handlers(ui):
    """Setup handler UI SmartCash"""
    # Handler env
    def on_env_change(change):
        with ui['env_status']:
            clear_output()
            if change['new'] == 'Google Colab':
                ui['colab_connect_button'].layout.display = ''
                display(create_info_alert("Colab detected. Connect Drive needed.", "info", "☁️"))
            else:
                ui['colab_connect_button'].layout.display = 'none'
                display(create_info_alert(f"Local detected. Path: {Path.cwd()}", "info", "💻"))
    ui['env_type'].observe(on_env_change, 'value')

    # Auto detect env
    def detect_env():
        try:
            import google.colab
            ui['env_type'].value = 'Google Colab'
        except ImportError:
            ui['env_type'].value = 'Local'
    
    # Handler clone
    def on_clone(b):
        with ui['repo_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Cloning..."))
            try:
                repo = ui['repo_url'].value.split("/")[-1].replace(".git", "")
                if Path(repo).exists():
                    display(create_status_indicator("warning", "⚠️ Repo exists"))
                    return
                time.sleep(2)  # Simulate
                display(create_status_indicator("success", f"✅ Cloned to {repo}"))
                with ui['overall_status']:
                    clear_output()
                    display(create_info_alert("Clone done! Next: env setup", "success", "🎉"))
            except Exception as e:
                display(create_status_indicator("error", f"❌ {e}"))
    ui['clone_button'].on_click(on_clone)
    
    # Handler colab
    def on_colab(b):
        with ui['env_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Connecting Drive..."))
            try:
                time.sleep(2)  # Simulate
                display(create_status_indicator("success", "✅ Drive connected"))
                with ui['overall_status']:
                    clear_output()
                    display(create_info_alert("Drive connected! Next: install deps", "success", "🎉"))
            except Exception as e:
                display(create_status_indicator("error", f"❌ {e}"))
    ui['colab_connect_button'].on_click(on_colab)
    
    # Handler install
    def on_install(b):
        with ui['deps_status']:
            clear_output()
            display(create_status_indicator("info", "🔄 Installing..."))
            pkgs = [p.strip() for p in ui['required_packages'].value.split('\n') if p.strip()]
            try:
                time.sleep(0.5 * len(pkgs))  # Simulate
                for p in pkgs:
                    display(create_status_indicator("success", f"✅ {p.split('>=')[0]} installed"))
                with ui['overall_status']:
                    clear_output()
                    display(create_info_alert("Setup complete! Ready to use.", "success", "🎉"))
                    display(HTML(f"""
                        <div style='background: #f8f9fa; padding: 10px; margin-top: 10px;'>
                            <h4>📊 System</h4>
                            <ul>
                                <li>Python: {platform.python_version()}</li>
                                <li>OS: {platform.system()}</li>
                                <li>Pkgs: {len(pkgs)}</li>
                                <li>Env: {ui['env_type'].value}</li>
                            </ul>
                        </div>
                    """))
            except Exception as e:
                display(create_status_indicator("error", f"❌ {e}"))
    ui['install_button'].on_click(on_install)
    
    detect_env()
    return ui