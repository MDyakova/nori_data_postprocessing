"""
Code launch flask server for NORI images postprocessing
"""
import os
import string
import threading
import time
import multiprocessing as mp
from typing import Optional
from flask import Flask, render_template, jsonify, request

app = Flask(__name__, template_folder="templates")

progress_log = []
status = {"running": False, "done": False, "error": None, "cancelled": False}
current_proc: Optional[mp.Process] = None
log_queue: Optional[mp.Queue] = None
cancelled_by_user = False

def notify(msg: str):
    """Server-side log helper (used for UI/system messages)."""
    print(msg)
    progress_log.append(msg)

def available_windows_drives():
    """Check available remote drivers"""
    drives = [f"{d}:" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
    return drives or ["C:", "D:", "Z:"]


def list_subfolders(base_path):
    """Create list with subfolders of data"""
    try:
        return sorted(
            [
                d
                for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d))
            ]
        )
    except Exception:
        return []


def join_windows_path(drive, rel):
    """Work with windows path"""
    return os.path.join(drive + os.sep, rel.lstrip("/\\"))


# -------- Subprocess target (DO NOT EDIT) ------
def _run_postprocessing(data: dict, q: mp.Queue):
    """
    Runs in a separate process. We pass a 'notify' that pushes messages
    into a queue so the main app can display them live.
    """
    from postprocessing import start
    def notify_child(msg: str):
        """Notifications start"""
        try:
            q.put(msg, block=False)
        except Exception:
            pass  # avoid blocking if queue is full/unavailable

    # keep your start(...) unchanged:
    start(data, notify_child)


# ------------------- Routes --------------------
@app.route("/")
def index():
    """Collect default parameters"""
    defaults = {
        "drive_letter": "Z:",
        "data_folder": r"NoRI\Masha\20250423 Ahmed Colon Cancer D14 NoRI",
        "stitched_files_folder": r"NoRI\Masha\Stitched",
        "powersetting": "UP",
        "file_separator": "_MAP",
        "subfolder_suffix": "",
        "calibration_directories": r"NoRI\Calibration Archive",
        "network_path": r"research.files.med.harvard.edu\Sysbio",
    }
    return render_template("home.html", defaults=defaults)


@app.get("/api/drives")
def api_drives():
    """Get available drivers"""
    return jsonify({"drives": available_windows_drives()})


@app.get("/api/calibration-folders")
def api_calibration():
    """Calibration directories"""
    drive = request.args.get("drive", "Z:")
    cal_dir = request.args.get("dir", r"\\NoRI\\Calibration Archive")
    base = join_windows_path(drive, cal_dir)
    return jsonify({"base_path": base, "folders": list_subfolders(base)})


@app.get("/api/data-folders")
def api_data_folders():
    """Data directories"""
    drive = request.args.get("drive", "Z:")
    data_folder = request.args.get("data_folder", r"\\NoRI\\Masha")
    base = os.path.join(drive + os.sep, data_folder.lstrip("\\/"))
    try:
        folders = [f for f in os.listdir(base) if os.path.isdir(os.path.join(base, f))]
    except Exception:
        folders = []
    return jsonify({"base_path": base, "folders": sorted(folders)})


# ---------------- Worker Thread ----------------
def worker(data):
    """
    Spawns a child process that runs postprocessing.start(data, notify_child).
    Streams logs back via a queue. On Stop, we kill the process and raise a
    controlled error so the except branch is taken (logging ❌ ERROR).
    """
    global current_proc, log_queue, cancelled_by_user

    try:
        from postprocessing import (
            start,
        )  # import to match your original structure (not used directly)

        # fresh state
        cancelled_by_user = False

        # queue for child->parent logs
        log_queue = mp.Queue()
        current_proc = mp.Process(target=_run_postprocessing, args=(data, log_queue))
        current_proc.start()

        # stream logs while process is alive
        while current_proc.is_alive():
            try:
                # poll logs frequently
                msg = log_queue.get(timeout=0.2)
                progress_log.append(msg)
            except Exception:
                pass

        # drain remaining logs after exit
        drained = True
        end_time = time.time() + 0.5
        while drained and time.time() < end_time:
            try:
                msg = log_queue.get_nowait()
                progress_log.append(msg)
            except Exception:
                drained = False

        # If user pressed Stop, force error path
        if cancelled_by_user:
            raise RuntimeError("Stopped by user")

        # Normal completion
        status.update({"running": False, "done": True})

    except Exception as e:
        status.update({"running": False, "done": True, "error": str(e)})
        if not progress_log or not progress_log[-1].startswith("❌ ERROR"):
            notify(f"❌ ERROR: {e}")
    finally:
        # cleanup
        try:
            if current_proc is not None and current_proc.is_alive():
                current_proc.terminate()
                current_proc.join(timeout=2)
        except Exception:
            pass


@app.post("/submit")
def submit():
    """Start process"""
    # collect form data
    data = request.form.to_dict(flat=True)
    data["selected_folders"] = request.form.getlist("selected_folders")

    # reset status
    progress_log.clear()
    status.update({"running": True, "done": False, "error": None, "cancelled": False})

    # start worker thread that manages the subprocess
    t = threading.Thread(target=worker, args=(data,), daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.post("/stop")
def stop():
    """Stop process"""
    global cancelled_by_user
    if status["running"]:
        cancelled_by_user = True
        status["cancelled"] = True
        notify("⏹️ Stop requested… terminating the running process.")
        try:
            if current_proc is not None and current_proc.is_alive():
                current_proc.terminate()
                # don't join here long; the worker thread will handle it
        except Exception as e:
            notify(f"⚠️ Stop encountered an issue: {e}")
        return jsonify({"ok": True, "message": "Stop requested"})
    return jsonify({"ok": False, "message": "No process running"})


@app.get("/progress")
def progress():
    """Progress bar"""
    return jsonify(
        {
            "running": status["running"],
            "done": status["done"],
            "cancelled": status["cancelled"],
            "error": status["error"],
            "log": progress_log,
            "can_stop": status["running"],
        }
    )


# ---------------- Entry point ------------------
if __name__ == "__main__":
    # Windows-safe start method for multiprocessing
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    app.run(debug=True, use_reloader=False)  # avoid double init
