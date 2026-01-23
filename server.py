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


# -------- Subprocess target (child process) -----
def _run_postprocessing(data: dict, q: mp.Queue):
    """
    Runs in a separate process. We pass a 'notify' that pushes messages
    into a queue so the main app can display them live.
    """
    from postprocessing import start

    def notify_child(msg: str):
        """Send log messages back to parent via queue."""
        try:
            q.put({"type": "log", "msg": msg}, block=False)
        except Exception:
            pass  # avoid blocking if queue is full/unavailable

    try:
        # keep your start(...) unchanged, except it receives notify_child
        start(data, notify_child)
        # notify parent about success (optional)
        try:
            q.put({"type": "status", "status": "ok"}, block=False)
        except Exception:
            pass
    except Exception as e:
        # send detailed error text to the parent so it can show in UI
        try:
            q.put(
                {
                    "type": "error",
                    "msg": f"{type(e).__name__}: {e}",
                },
                block=False,
            )
        except Exception:
            pass
        # re-raise so the child process exits with a non-zero exit code
        raise


def _handle_queue_message(msg, error_state):
    """
    Helper: handle one message from the child process queue.
    msg: can be dict (typed) or plain string.
    error_state: dict with {"error_seen": bool}
    """
    if msg is None:
        return

    # Structured message from child
    if isinstance(msg, dict):
        mtype = msg.get("type")
        if mtype == "log":
            text = msg.get("msg", "")
            if text:
                progress_log.append(text)
        elif mtype == "error":
            error_state["error_seen"] = True
            err_text = msg.get("msg", "Unknown error")
            progress_log.append(f"❌ ERROR: {err_text}")
        elif mtype == "status":
            # Optional: show status message
            status_text = msg.get("status", "")
            if status_text:
                progress_log.append(f"ℹ️ Status: {status_text}")
    else:
        # Old style raw string message
        progress_log.append(str(msg))


# ---------------- Worker Thread ----------------
def worker(data):
    """
    Spawns a child process that runs postprocessing.start(data, notify_child).
    Streams logs back via a queue. On Stop, or if the child crashes,
    we go through the error path (logging ❌ ERROR and setting status["error"]).
    """
    global current_proc, log_queue, cancelled_by_user

    try:
        # Import to match your original structure (not used directly)
        from postprocessing import start  # noqa: F401

        # fresh state
        cancelled_by_user = False

        # queue for child->parent logs
        log_queue = mp.Queue()
        current_proc = mp.Process(target=_run_postprocessing, args=(data, log_queue))
        current_proc.start()

        error_state = {"error_seen": False}

        # stream logs while process is alive
        while current_proc.is_alive():
            try:
                msg = log_queue.get(timeout=0.2)
            except Exception:
                msg = None
            _handle_queue_message(msg, error_state)

        # drain remaining logs after exit
        drained = True
        end_time = time.time() + 0.5
        while drained and time.time() < end_time:
            try:
                msg = log_queue.get_nowait()
            except Exception:
                drained = False
                break
            _handle_queue_message(msg, error_state)

        exitcode = current_proc.exitcode

        # If user pressed Stop, force error path
        if cancelled_by_user:
            raise RuntimeError("Stopped by user")

        # If child crashed or reported an error, treat as failure
        if error_state["error_seen"] or exitcode not in (0, None):
            raise RuntimeError(
                f"Post-processing failed (exit code {exitcode}). See log for details."
            )

        # Normal completion
        status.update({"running": False, "done": True, "error": None})
        notify("✅ Processing complete.")

    except Exception as e:
        status.update(
            {
                "running": False,
                "done": False,
                "error": str(e),
                "cancelled": cancelled_by_user,
            }
        )
        # If the last message is not already an ERROR line, add one
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


# ------------------- Routes --------------------
@app.route("/")
def index():
    """Collect default parameters"""
    defaults = {
        "drive_letter": "N:",
        "data_folder": r"NoRI\Masha\20250423 Ahmed Colon Cancer D14 NoRI",
        "stitched_files_folder": r"NoRI\Masha\Stitched",
        "powersetting": "UP",
        "file_separator": "_MAP",
        "subfolder_suffix": "",
        "calibration_directories": r"NoRI\Calibration Archive",
        "network_path": r"research.files.med.harvard.edu\Sysbio",
        "fluorescent_tag" : '_IF_'
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


@app.post("/submit")
def submit():
    """Start process"""
    # collect form data
    data = request.form.to_dict(flat=True)
    data["selected_folders"] = request.form.getlist("selected_folders")

    # reset status
    progress_log.clear()
    status.update(
        {"running": True, "done": False, "error": None, "cancelled": False}
    )

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
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)  # avoid double init
