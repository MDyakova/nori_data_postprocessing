# imagej_session.py
import os, threading, jpype

JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.24.8-hotspot"
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = JAVA_HOME + r"\bin;" + os.environ["PATH"]

_init_lock = threading.Lock()
_ij = None

def get_ij():
    """Initialize Fiji exactly once and reuse it."""
    global _ij
    if _ij is not None:
        return _ij
    with _init_lock:
        if _ij is not None:
            return _ij
        import imagej  # import here, after env is set
        # If JVM already started (e.g., previous run), reuse it; don't try to start again
        if not jpype.isJVMStarted():
            _ij = imagej.init('sc.fiji:fiji', mode='headless')
        else:
            # Attach to existing JVM; imagej.init is idempotent in this case
            _ij = imagej.init('sc.fiji:fiji', mode='headless')
        return _ij