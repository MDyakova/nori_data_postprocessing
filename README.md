# **NORI Post-Processing Pipeline**

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
## Citation

If you use this toolkit in your research, please cite Cellpose and trackpy as appropriate and reference this repository.

---
## License

MIT License

---

### 📫 **Contact**
For questions or contributions, please contact:
**Mariia Diakova**
- GitHub: [MDyakova](https://github.com/MDyakova)
- email: m.dyakova.ml@gmail.com

