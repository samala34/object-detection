# Multi-Stream Office Floor Monitor (YOLO & SAHI Engine)

A modern, multi-threaded Windows desktop application designed for real-time office floor monitoring and object detection. Powered by **YOLO (YOLO11 & YOLO26)** and **SAHI (Slicing Aided Hyper Inference)**, this tool allows operators to monitor multiple video channels simultaneously, log detected objects with confidence ratings, and capture violation/detection frames automatically.

---

## Key Features

* **Multi-Channel Parallel Processing:** Open multiple video streams in separate tabs and run AI inference concurrently.
* **YOLO Architecture Selection:** Toggle dynamically between **YOLO11** and **YOLO26** across all variants (`n`, `s`, `m`, `l`, `x`).
* **On-Demand Model Downloader:** Integrated background downloader to fetch missing `.pt` model weights directly from the interface.
* **SAHI Sliced Inference:** Slices high-resolution frames into $512 \times 512$ tiles with overlapping predictions for accurate small-object detection.
* **Automated Frame Storage:** Automatically saves annotated frames to isolated subfolders (`output_detections/<video_name>/`) whenever object counts change.
* **Logging & Reporting:** Exports live stream logs to CSV and generates a performance summary log (`processing_summary_log.csv`) upon video completion.

---

## System Requirements

* **OS:** Windows 10 or Windows 11 (64-bit)
* **Python:** Python 3.10 to 3.13
* **Hardware:** Multi-core CPU (GPU support available if CUDA and PyTorch GPU drivers are installed)

---

## Installation & Setup

### 1. Clone the Repository & Set Up Virtual Environment

```bash
# Clone repository or extract project source files
cd object-detection

# Create a virtual environment
python -m venv venv

# Activate the virtual environment (Windows Command Prompt)
venv\Scripts\activate

# Activate using PowerShell
.\venv\Scripts\Activate.ps1

```

### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install customtkinter opencv-python pillow sahi ultralytics
or 
pip install -r requirements.txt

```

---

## Running the Application

To start the application directly using Python:

```bash
python app.py

```

### Quick Start Guide:

1. Select your desired **Model** (`YOLO11` or `YOLO26`) and **Variant** (`n`, `s`, `m`, `l`, `x`) from the top control bar.
2. If the model weights are not found locally, click **📥 Download Model**.
3. Click **Browse Video** inside a channel tab to load an `.mp4`, `.avi`, or `.mkv` video file.
4. Choose the target COCO object classes you want to detect (or click **All** / **None**).
5. Click **Start Processing** to launch the detection engine.
6. Click **+ Add Video Channel** to monitor additional videos in parallel tabs.

---

## Packaging to Standalone Executable (`.exe`)

To package this application into a standalone Windows `.exe` that runs on computers without Python installed:

### Step 1: Pre-download Default Model Weights

Ensure at least one default weight file (e.g., `yolo26n.pt`) is present in your project directory:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"

```


## Output Directories & Log Files

* **`output_detections/<video_name>/`**: Contains `.png` snapshots of annotated video frames saved whenever object counts change.
* **`processing_summary_log.csv`**: A structured performance report appended automatically when a video stream finishes processing (contains video duration, total processing time, FPS, and saved frame counts).
* **Exported Log CSVs**: Custom session log files generated via the **Export CSV** button inside the active stream tab.

