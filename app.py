import os
import sys
import cv2
import csv
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image

# SAHI & Ultralytics imports
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

# Set UI Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Complete COCO Dataset 80 Classes
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class VideoMonitorTab(ctk.CTkFrame):
    """Represents a single video monitoring session inside a tab."""
    def __init__(self, parent, tab_title, remove_tab_callback, shared_model_func, object_options):
        super().__init__(parent, fg_color="transparent")
        
        self.tab_title = tab_title
        self.remove_tab_callback = remove_tab_callback
        self.get_shared_model = shared_model_func
        self.object_options = object_options
        
        # Tab-Specific State
        self.video_path = None
        self.video_name = "Unassigned"
        self.is_processing = False
        self.logged_data = []
        self.output_dir = "output_detections"

        # Build UI layout for this specific tab
        self._create_layout()

    def _create_layout(self):
        # Sidebar Controls for this tab
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=8)
        self.sidebar.pack(side="left", fill="y", padx=(0, 10), pady=10)
        self.sidebar.pack_propagate(False)

        # 1. File Selection
        ctk.CTkLabel(self.sidebar, text="1. Source Video", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(15, 5))
        self.btn_select = ctk.CTkButton(self.sidebar, text="Browse Video", command=self.select_video)
        self.btn_select.pack(padx=15, pady=5, fill="x")

        self.lbl_file_status = ctk.CTkLabel(self.sidebar, text="No file selected", font=ctk.CTkFont(size=11), text_color="gray")
        self.lbl_file_status.pack(padx=15, pady=(0, 10))

        # 2. Objects Selection
        header_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        header_frame.pack(fill="x", padx=15, pady=(10, 5))
        
        ctk.CTkLabel(header_frame, text="2. Target Objects", font=ctk.CTkFont(weight="bold")).pack(side="left")

        btn_clear_all = ctk.CTkButton(header_frame, text="None", width=40, height=20, fg_color="gray30", command=self.deselect_all_classes)
        btn_clear_all.pack(side="right", padx=(2, 0))
        btn_select_all = ctk.CTkButton(header_frame, text="All", width=35, height=20, fg_color="gray30", command=self.select_all_classes)
        btn_select_all.pack(side="right")

        scroll_frame = ctk.CTkScrollableFrame(self.sidebar, height=180, fg_color="transparent")
        scroll_frame.pack(fill="x", padx=10, pady=5)

        self.checkbox_vars = {}
        default_active = ["person", "chair", "laptop", "cell phone", "bottle", "keyboard", "mouse", "book"]
        
        for obj in sorted(self.object_options):
            default_val = True if obj in default_active else False
            var = ctk.BooleanVar(value=default_val)
            chk = ctk.CTkCheckBox(scroll_frame, text=obj.capitalize(), variable=var)
            chk.pack(anchor="w", padx=10, pady=3)
            self.checkbox_vars[obj] = var

        # Action Buttons
        self.btn_start = ctk.CTkButton(
            self.sidebar, text="Start Processing", fg_color="#1f538d", hover_color="#14375e", command=self.start_processing_thread
        )
        self.btn_start.pack(padx=15, pady=(15, 5), fill="x")

        self.btn_stop = ctk.CTkButton(
            self.sidebar, text="Stop Processing", fg_color="#a83232", hover_color="#7a2323", state="disabled", command=self.stop_processing
        )
        self.btn_stop.pack(padx=15, pady=5, fill="x")

        self.btn_close = ctk.CTkButton(
            self.sidebar, text="✖ Remove This Channel", fg_color="gray30", hover_color="#8b0000", command=self.close_tab
        )
        self.btn_close.pack(padx=15, pady=(15, 5), fill="x", side="bottom")

        # Main Workspace
        self.workspace = ctk.CTkFrame(self, fg_color="transparent")
        self.workspace.pack(side="right", fill="both", expand=True, pady=10)

        # Video Preview
        self.video_box = ctk.CTkFrame(self.workspace, fg_color="black", height=380, corner_radius=8)
        self.video_box.pack(side="top", fill="x", expand=False)
        self.video_box.pack_propagate(False)

        self.video_label = ctk.CTkLabel(self.video_box, text="[ Video Stream Preview ]\nSelect a video and click 'Start Processing'")
        self.video_label.pack(fill="both", expand=True)

        # Stream Log Display
        self.log_frame = ctk.CTkFrame(self.workspace, corner_radius=8)
        self.log_frame.pack(side="bottom", fill="both", expand=True, pady=(10, 0))

        log_header = ctk.CTkFrame(self.log_frame, fg_color="transparent")
        log_header.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(log_header, text="Live Session Log", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")

        ctk.CTkButton(log_header, text="Open Folder", width=85, height=22, fg_color="#43a047", command=self.open_output_folder).pack(side="right", padx=(5, 0))
        ctk.CTkButton(log_header, text="Export CSV", width=80, height=22, fg_color="#2e7d32", command=self.export_log_csv).pack(side="right", padx=(5, 0))
        ctk.CTkButton(log_header, text="Clear Log", width=75, height=22, fg_color="gray30", command=self.clear_log).pack(side="right")

        self.txt_log = ctk.CTkTextbox(self.log_frame, font=ctk.CTkFont(family="Consolas", size=11))
        self.txt_log.pack(fill="both", expand=True, padx=10, pady=(0, 5))
        self.txt_log.insert("1.0", "Session Ready...\n")
        self.txt_log.configure(state="disabled")

    def select_all_classes(self):
        for var in self.checkbox_vars.values():
            var.set(True)

    def deselect_all_classes(self):
        for var in self.checkbox_vars.values():
            var.set(False)

    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            self.video_path = file_path
            self.video_name = os.path.splitext(os.path.basename(file_path))[0]
            
            self.output_dir = os.path.join("output_detections", self.video_name)
            os.makedirs(self.output_dir, exist_ok=True)

            filename = os.path.basename(file_path)
            self.lbl_file_status.configure(text=f"Loaded: {filename[:22]}...", text_color="#4CAF50")

    def get_selected_targets(self):
        return [obj for obj, var in self.checkbox_vars.items() if var.get()]

    def open_output_folder(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        os.startfile(os.path.abspath(self.output_dir))

    def clear_log(self):
        self.logged_data.clear()
        self.txt_log.configure(state="normal")
        self.txt_log.delete("1.0", tk.END)
        self.txt_log.configure(state="disabled")

    def export_log_csv(self):
        if not self.logged_data:
            messagebox.showinfo("Export", "No logged detections available in this session.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if save_path:
            try:
                with open(save_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Frame Filename", "Class", "Confidence", "Confidence Level"])
                    writer.writerows(self.logged_data)
                messagebox.showinfo("Success", f"Session log saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export CSV:\n{str(e)}")

    def start_processing_thread(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file for this tab.")
            return

        selected_targets = self.get_selected_targets()
        if not selected_targets:
            messagebox.showwarning("Warning", "Please select at least one object category.")
            return

        self.is_processing = True
        self.btn_start.configure(state="disabled")
        self.btn_select.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        
        self.clear_log()
        threading.Thread(target=self.process_video, daemon=True).start()

    def stop_processing(self):
        self.is_processing = False
        self.btn_start.configure(state="normal")
        self.btn_select.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def close_tab(self):
        if self.is_processing:
            if not messagebox.askyesno("Confirm Remove", "Video processing is active on this channel. Stop and remove?"):
                return
            self.stop_processing()
        
        self.remove_tab_callback(self.tab_title)

    def log_and_print_summary(self, video_duration_seconds, total_processing_time, processed_frames, saved_frames_count):
        """Prints report to console terminal and appends entry to summary CSV log."""
        processing_speed_fps = (processed_frames / total_processing_time) if total_processing_time > 0 else 0.0

        print("\n" + "="*40)
        print("          PROCESSING REPORT          ")
        print("="*40)
        print(f"Original Video Length : {video_duration_seconds / 60:.2f} minutes")
        print(f"Total Time Taken      : {total_processing_time / 60:.2f} minutes ({total_processing_time:.1f} seconds)")
        print(f"Total Frames Analyzed : {processed_frames} frames (SAHI slices)")
        print(f"Processing Speed      : {processing_speed_fps:.2f} frames per second")
        print(f"Total Frames Saved    : {saved_frames_count} images (Synced 1:1 with log)")
        print("="*40 + "\n")

        csv_log_file = "processing_summary_log.csv"
        file_exists = os.path.exists(csv_log_file)

        try:
            with open(csv_log_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "Video Name", "Video Duration (min)", "Time Taken (min)", 
                        "Time Taken (sec)", "Frames Analyzed", "FPS Speed", "Synced Frames Saved"
                    ])

                writer.writerow([
                    self.video_name,
                    f"{video_duration_seconds / 60:.2f}",
                    f"{total_processing_time / 60:.2f}",
                    f"{total_processing_time:.1f}",
                    processed_frames,
                    f"{processing_speed_fps:.2f}",
                    saved_frames_count
                ])
        except Exception as err:
            print(f"[Warning] Unable to write summary log CSV: {err}")

    def process_video(self):
        detection_model = self.get_shared_model()
        if not detection_model:
            self.stop_processing()
            return

        start_time = time.time()
        processed_frames = 0
        saved_frames_count = 0
        video_duration_seconds = 0.0

        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                messagebox.showerror("Error", f"Failed to open video stream:\n{self.video_name}")
                self.stop_processing()
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_vid_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0 and total_vid_frames > 0:
                video_duration_seconds = total_vid_frames / fps

            # OPTIMIZATION #1: Video Stride Sampling (Runs AI once per second)
            stride = max(1, int(fps)) if fps > 0 else 25

            # OPTIMIZATION #2: Efficient Slicing Grid (640x640 with 10% overlap reduces slices by ~60%)
            slice_size = 640
            overlap_ratio = 0.10

            while cap.isOpened() and self.is_processing:
                current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames that do not align with the stride interval
                if current_frame_pos % stride != 0:
                    continue

                processed_frames += 1

                # Calculate minute, second, and millisecond timestamps
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                total_seconds = int(msec // 1000)
                mins = total_seconds // 60
                secs = total_seconds % 60
                millis = int(msec % 1000)

                timestamp_str = f"{mins:02d}:{secs:02d}.{millis:03d}"

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_h, frame_w, _ = rgb_frame.shape

                # SAHI Sliced Prediction with Optimized Grid Geometry
                result = get_sliced_prediction(
                    rgb_frame,
                    detection_model,
                    slice_height=slice_size,
                    slice_width=slice_size,
                    overlap_height_ratio=overlap_ratio,
                    overlap_width_ratio=overlap_ratio,
                    postprocess_type="GREEDYNMM",
                    postprocess_match_metric="IOU",
                    postprocess_match_threshold=0.5
                )

                active_targets = self.get_selected_targets()
                frame_detections = []
                annotated_frame = rgb_frame.copy()

                for object_prediction in result.object_prediction_list:
                    class_name = object_prediction.category.name.lower()
                    score = object_prediction.score.value

                    if class_name not in active_targets:
                        continue

                    bbox = object_prediction.bbox
                    x1 = max(0, min(int(bbox.minx), frame_w - 1))
                    y1 = max(0, min(int(bbox.miny), frame_h - 1))
                    x2 = max(0, min(int(bbox.maxx), frame_w - 1))
                    y2 = max(0, min(int(bbox.maxy), frame_h - 1))

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 128), 2)
                    cv2.putText(
                        annotated_frame, f"{class_name.capitalize()} {int(score * 100)}%", 
                        (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2
                    )

                    frame_detections.append({"class": class_name.capitalize(), "score": score})

                # Save snapshot image on EVERY evaluated stride frame where objects exist
                saved_filename = "N/A"
                if frame_detections:
                    saved_frames_count += 1
                    saved_filename = f"frame_{mins:02d}m_{secs:02d}s_{millis:03d}ms.png"
                    save_path = os.path.join(self.output_dir, saved_filename)
                    
                    bgr_save = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, bgr_save)

                # Log entry referencing the exact saved filename for analytical traceability
                self.append_clean_log(timestamp_str, saved_filename, frame_detections)

                # OPTIMIZATION #3: Render GUI Preview only for sampled stride frames
                img = Image.fromarray(annotated_frame)
                widget_w = max(self.video_box.winfo_width(), 100)
                widget_h = max(self.video_box.winfo_height(), 100)
                img = img.resize((widget_w, widget_h), Image.Resampling.LANCZOS)

                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(widget_w, widget_h))
                self.video_label.configure(image=ctk_img, text="")
                self.video_label.image = ctk_img

            cap.release()

            # Output Performance Summary Report
            total_processing_time = time.time() - start_time
            if processed_frames > 0:
                self.log_and_print_summary(video_duration_seconds, total_processing_time, processed_frames, saved_frames_count)

        except Exception as err:
            messagebox.showerror("Execution Error", f"Error on stream '{self.video_name}':\n{str(err)}")
        finally:
            self.stop_processing()

    def append_clean_log(self, timestamp, filename, detections):
        """Formats stream detections into a clear log layout mapped 1:1 to saved image filenames."""
        self.txt_log.configure(state="normal")
        if not detections:
            log_line = f"[{timestamp}] - No active targets detected.\n"
        else:
            grouped = {}
            for d in detections:
                cls, score = d["class"], d["score"]
                grouped.setdefault(cls, []).append(score)

            log_line = f"⏱ [{timestamp}] Total Objects: {len(detections)} | Image: {filename}\n"
            for cls_name, scores in grouped.items():
                score_strings = []
                for score in scores:
                    pct = int(score * 100)
                    level = "HIGH" if score >= 0.70 else ("MED" if score >= 0.50 else "LOW")
                    score_strings.append(f"{pct}% ({level})")
                    self.logged_data.append([timestamp, filename, cls_name, f"{pct}%", level])

                log_line += f"   • {cls_name} ({len(scores)}x) → Confidence: {', '.join(score_strings)}\n"

        log_line += "-" * 55 + "\n"
        self.txt_log.insert(tk.END, log_line)
        self.txt_log.see(tk.END)
        self.txt_log.configure(state="disabled")


class OfficeFloorMonitorApp(ctk.CTk):
    """Main Application Window with Dynamic Model Selection & Downloader."""
    def __init__(self):
        super().__init__()

        self.title("Multi-Stream Office Floor Monitor - YOLO Engine")
        self.geometry("1250x800")
        self.minsize(1050, 720)

        self.shared_detection_model = None
        self.loaded_model_filename = None
        self.tab_count = 0
        self.active_tabs = {}
        self.object_options = COCO_CLASSES

        self._create_top_bar()

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=5)

        self.add_new_tab()

    def _create_top_bar(self):
        self.top_bar = ctk.CTkFrame(self, height=50, corner_radius=0)
        self.top_bar.pack(side="top", fill="x")

        ctk.CTkLabel(self.top_bar, text="AI Floor Monitor", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left", padx=15)

        ctk.CTkLabel(self.top_bar, text="Model:", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=(10, 2))
        self.combo_model = ctk.CTkOptionMenu(
            self.top_bar, values=["YOLO11", "YOLO26"], width=95, command=self.on_model_selection_change
        )
        self.combo_model.set("YOLO26")
        self.combo_model.pack(side="left", padx=2)

        ctk.CTkLabel(self.top_bar, text="Variant:", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=(8, 2))
        self.combo_variant = ctk.CTkOptionMenu(
            self.top_bar, values=["n", "s", "m", "l", "x"], width=65, command=self.on_model_selection_change
        )
        self.combo_variant.set("n")
        self.combo_variant.pack(side="left", padx=2)

        self.lbl_model_status = ctk.CTkLabel(self.top_bar, text="", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_model_status.pack(side="left", padx=10)

        self.btn_download_model = ctk.CTkButton(
            self.top_bar, text="📥 Download Model", width=130, fg_color="#d97706", hover_color="#b45309", command=self.download_selected_model
        )
        
        self.btn_add_tab = ctk.CTkButton(
            self.top_bar, text="+ Add Video Channel", width=140, fg_color="#2e7d32", hover_color="#1b5e20", command=self.add_new_tab
        )
        self.btn_add_tab.pack(side="right", padx=15, pady=8)

        self.check_selected_model_availability()

    def get_selected_model_filename(self):
        model_arch = self.combo_model.get().lower()
        variant = self.combo_variant.get().lower()
        return f"{model_arch}{variant}"

    def resolve_model_file_path(self, base_name):
        # OPTIMIZATION #4: Prefer OpenVINO directory if present for hardware acceleration
        openvino_dir = f"{base_name}_openvino_model"
        pt_file = f"{base_name}.pt"

        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            bundled_ov = os.path.join(sys._MEIPASS, openvino_dir)
            bundled_pt = os.path.join(sys._MEIPASS, pt_file)
            if os.path.exists(bundled_ov):
                return bundled_ov, "openvino"
            if os.path.exists(bundled_pt):
                return bundled_pt, "ultralytics"

        if os.path.exists(openvino_dir):
            return openvino_dir, "openvino"
        elif os.path.exists(pt_file):
            return pt_file, "ultralytics"

        return pt_file, "ultralytics"

    def check_selected_model_availability(self):
        base_name = self.get_selected_model_filename()
        resolved_path, model_type = self.resolve_model_file_path(base_name)

        if os.path.exists(resolved_path):
            tag = "OpenVINO" if model_type == "openvino" else "PyTorch"
            self.lbl_model_status.configure(text=f"✓ {os.path.basename(resolved_path)} ({tag}) Ready", text_color="#4CAF50")
            self.btn_download_model.pack_forget()
            return True
        else:
            self.lbl_model_status.configure(text=f"⚠ {base_name}.pt Missing", text_color="#EF4444")
            self.btn_download_model.pack(side="left", padx=5)
            return False

    def on_model_selection_change(self, _choice=None):
        base_name = self.get_selected_model_filename()
        if base_name != self.loaded_model_filename:
            self.shared_detection_model = None
            self.loaded_model_filename = None
        self.check_selected_model_availability()

    def download_selected_model(self):
        base_name = self.get_selected_model_filename()
        filename = f"{base_name}.pt"
        
        self.btn_download_model.configure(state="disabled", text="Downloading...")
        self.lbl_model_status.configure(text=f"⏳ Fetching {filename}...", text_color="#EAB308")

        def run_download():
            try:
                YOLO(filename)
                self.after(0, lambda: self.on_download_finished(True, filename))
            except Exception as err:
                self.after(0, lambda: self.on_download_finished(False, str(err)))

        threading.Thread(target=run_download, daemon=True).start()

    def on_download_finished(self, success, result_info):
        self.btn_download_model.configure(state="normal", text="📥 Download Model")
        if success:
            messagebox.showinfo("Success", f"Model '{result_info}' downloaded successfully!")
            self.check_selected_model_availability()
        else:
            messagebox.showerror("Download Error", f"Failed to download weights:\n{result_info}")
            self.check_selected_model_availability()

    def get_shared_model(self):
        base_name = self.get_selected_model_filename()
        resolved_path, model_type = self.resolve_model_file_path(base_name)

        if not os.path.exists(resolved_path):
            messagebox.showwarning("Model Missing", f"Model file/directory '{base_name}' is missing.\nPlease click 'Download Model' at the top bar first.")
            return None

        if not self.shared_detection_model or self.loaded_model_filename != base_name:
            try:
                self.shared_detection_model = AutoDetectionModel.from_pretrained(
                    model_type=model_type,
                    model_path=resolved_path,
                    confidence_threshold=0.25,
                    device="cpu"
                )
                self.loaded_model_filename = base_name
            except Exception as e:
                messagebox.showerror("Model Error", f"Failed to load SAHI model from '{resolved_path}':\n{str(e)}")
                return None

        return self.shared_detection_model

    def add_new_tab(self):
        self.tab_count += 1
        tab_title = f"Stream #{self.tab_count}"
        
        self.tabview.add(tab_title)
        tab_container = self.tabview.tab(tab_title)

        monitor_frame = VideoMonitorTab(
            tab_container, tab_title, self.remove_tab, self.get_shared_model, COCO_CLASSES
        )
        monitor_frame.pack(fill="both", expand=True)
        
        self.active_tabs[tab_title] = monitor_frame
        self.tabview.set(tab_title)

    def remove_tab(self, tab_title):
        if tab_title in self.active_tabs:
            del self.active_tabs[tab_title]
            self.tabview.delete(tab_title)

        if not self.active_tabs:
            self.tab_count = 0
            self.add_new_tab()


if __name__ == "__main__":
    app = OfficeFloorMonitorApp()
    app.mainloop()