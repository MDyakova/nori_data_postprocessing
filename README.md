Below is a clean **README.md** for your project, fully tailored to the real codebase you uploaded.
It explains the system architecture, pipeline steps, usage, API endpoints, file structure, and references your actual code files.

---

# **NORI Post-Processing Pipeline**

**NORI (Normalized Raman Imaging) Image Post-Processing Toolkit**
Full pipeline for converting raw `.oir` files into stitched, decomposed, background-corrected, flat-field-corrected, and multi-channel quantitative TIFF images.

This repository contains:

* A **computational backend** that performs all heavy imaging steps (conversion, background subtraction, flat-field correction, spectral decomposition, tile stitching, fluorescence alignment).
  Source: `postprocessing.py`  and `utilities.py` 

* A **Flask web interface** to configure and run processing jobs with live logging.
  Source: `server.py` 

* A clean **requirements.txt** with pinned versions for reproducible environments.
  Source: `requirements.txt` 

---

# 🚀 Key Features

### ✔ Convert `.oir` → `.tif`

Using **ImageJ / Fiji (headless)** through `pyimagej`.

### ✔ Background subtraction

Removes dark noise computed from calibration files.

### ✔ Flat-field correction

Creates smooth polynomial correction masks for all supported resolutions:
256, 512, 640, 800, 1024, 2048, 4096 px.

### ✔ Spectral decomposition

Performs **protein / lipid / water decomposition** using linear unmixing.

### ✔ Tile detection & stitching

Automatic:

* tile arrangement detection,
* tile overlap estimation,
* snake-order walking,
* correlation-based tile shift calculation,
* distance-transform blending (feathering) for seamless stitching.

Supports **2D** and **3D** datasets.

### ✔ Fluorescence IF alignment

Applies pre-measured XY shift offsets for confocal channels.

### ✔ OME-TIFF output

Exports composite multi-channel stacks with correct axes metadata (CYX or ZCYX).

### ✔ Web interface for users

Includes:

* driver and folder browser
* configuration panel
* live “progress log”
* safe **Start/Stop** controls
* isolated multiprocessing for heavy computation

---

# 📦 Installation

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

(See `requirements.txt` for exact pinned versions.)

Java requirements (for ImageJ):

* Install a JDK (the code uses Adoptium/OpenJDK 21)
* Ensure `JAVA_HOME` points to it (set automatically inside code if needed)

---

# 🗂 Project Structure

```
.
├── server.py              # Flask web server + multiprocessing controller
├── postprocessing.py      # Main NORI processing pipeline
├── utilities.py           # Helper functions (stitching, background, decomposition)
├── requirements.txt       # Pinned versions
└── templates/
    └── home.html          # Web UI
```

---

# 🧠 How the Pipeline Works

The main job runner is `start()` in **postprocessing.py** .

### **1. Calibration loading**

* Reads dark noise
* Loads decomposition matrix
* Builds flat-field masks for all resolutions

### **2. File discovery**

Scans selected folders for `.oir` files.

### **3. Conversion & channel separation**

Via `oir_to_tif()` in **utilities.py** :

* Converts to TIFF
* Detects lipid/protein/water channels
* Renames and stores them

### **4. Background subtraction**

`background_substruction()` subtracts calibration dark noise.

### **5. Flat-field correction**

`flat_field_correction()` applies resolution-matched correction masks.

### **6. Spectral decomposition**

Creates **protein**, **lipid**, **water** quantitative maps.

### **7. Channel grouping**

Using `combine_channels()`.

### **8. Tile arrangement detection**

Detects optimal `(rows, cols, shift)` from correlation patterns.

### **9. Stitching**

Two modes:

* **2D**: `tiles_stitching()`
* **3D**: `file_stitching_3d()`

Stitching uses:

* correlation overlap search
* distance-transform feather blending (`blend_distance_feather()`)

### **10. IF correction (if present)**

Aligns IF channels using predefined shifts:

```python
fluorescence_shift_dict = {256:(3,-1), 512:(7,-4), ...}
```

### **11. Export**

OME-TIFF output with correct axes.

---

# 🌐 Running the Web App

Launch the server:

```bash
python server.py
```

Then open in browser:

```
http://127.0.0.1:5000
```

### Web interface provides:

* Selection of drive/network folder
* Choose data folders
* Configure file separator, suffix, calibration folder
* Run job in background process
* Real-time progress log
* Stop job safely

---

# 🔌 API Endpoints

### `GET /`

Main UI.

### `GET /api/drives`

Detects available Windows drives.

### `GET /api/calibration-folders`

Lists calibration folders for selected drive.

### `GET /api/data-folders`

Lists NORI data folders.

### `POST /submit`

Starts processing job.

### `POST /stop`

Stops job gracefully.

### `GET /progress`

Returns:

* running state
* done/error flags
* progress log
* stop availability

All implemented in **server.py** .

---

# 🧪 Example Output Structure

```
/MyDataFolder/
    signalX/
    signal_bg/
    signal_bg_ffc2/
    decomp_bg_ffc2/
        composite/
            <sample>_MAPX_drawing.tif
            <sample>_MAPX.tif
        <stitched_output>.tif
```

---

# 📬 Acknowledgements

This codebase integrates:

* ImageJ2/Fiji via PyImageJ
* SciPy / Scikit-image / OpenCV for image processing
* Nori spectral decomposition logic
* Custom stitching algorithms developed specifically for large NoRI datasets

---
