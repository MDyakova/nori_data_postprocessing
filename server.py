from flask import Flask, render_template, jsonify, request
import os, string
from postprocessing import start
import threading

app = Flask(__name__, template_folder='templates')

def available_windows_drives():
    drives = [f"{d}:" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
    return drives or ["C:", "D:", "Z:"]

def list_subfolders(base_path):
    try:
        return sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    except Exception:
        return []

def join_windows_path(drive, rel):
    return os.path.join(drive + os.sep, rel.lstrip("/\\"))

@app.route('/')
def index():
    defaults = {
        'drive_letter': 'Z:',
        'data_folder': 'NoRI\\Masha\\20250423 Ahmed Colon Cancer D14 NoRI',
        'stitched_files_folder': 'NoRI\\Masha\\Stitched',
        'powersetting': 'UP',
        'file_separator': '_MAP',
        'calibration_directories': 'NoRI\\Calibration Archive',
        'network_path': 'research.files.med.harvard.edu\Sysbio'
    }
    return render_template('home.html', defaults=defaults)

@app.get('/api/drives')
def api_drives():
    return jsonify({'drives': available_windows_drives()})

@app.get('/api/calibration-folders')
def api_calibration():
    drive = request.args.get('drive', 'Z:')
    cal_dir = request.args.get('dir', r'\\NoRI\\Calibration Archive')
    base = join_windows_path(drive, cal_dir)
    return jsonify({'base_path': base, 'folders': list_subfolders(base)})

@app.get("/api/data-folders")
def api_data_folders():
    drive = request.args.get("drive", "Z:")
    data_folder = request.args.get("data_folder", r"\\NoRI\\Masha")
    base = os.path.join(drive + os.sep, data_folder.lstrip("\\/"))
    try:
        folders = [f for f in os.listdir(base) if os.path.isdir(os.path.join(base, f))]
    except Exception as e:
        folders = []
    return jsonify({"base_path": base, "folders": sorted(folders)})

progress_log = []
status = {"running": False, "done": False, "error": None}

def notify(msg):
    print(msg)
    progress_log.append(msg)

@app.post('/submit')
def submit():
    data = request.form.to_dict(flat=True)
    data['selected_folders'] = request.form.getlist('selected_folders')

    progress_log.clear()
    status.update({"running": True, "done": False, "error": None})

    def worker():
        try:
            from postprocessing import start
            start(data, notify)
            status.update({"running": False, "done": True})
        except Exception as e:
            status.update({"running": False, "done": True, "error": str(e)})
            # notify was already called in main, but ensure at least one line:
            if not progress_log or not progress_log[-1].startswith("❌ ERROR"):
                notify(f"❌ ERROR: {e}")

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"status": "started"})

@app.get('/progress')
def progress():
    return jsonify({
        "running": status["running"],
        "done": status["done"],
        "error": status["error"],
        "log": progress_log,
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # <- avoid double init in dev
