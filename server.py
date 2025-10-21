from flask import Flask, render_template, jsonify, request
import os, string
from postprocessing import start

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

@app.post('/submit')
def submit():
    data = {k: v for k, v in request.form.items()}
    # Checkbox values come as multiple "folders" entries:
    selected = request.form.getlist("selected_folders")
    data["selected_folders"] = selected
    start(data)
    # for msg in start(data):
    #     yield f"data: {msg}\n\n"
    return jsonify({"status": "ok", "message": 'Script finished'})

if __name__ == '__main__':
    app.run(debug=True)
