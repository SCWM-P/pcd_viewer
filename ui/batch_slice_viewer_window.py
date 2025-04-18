# -*- coding: utf-8 -*-
# pcd_viewer/ui/batch_slice_viewer_window.py

import os
import json
import datetime
import numpy as np
import pyvista as pv
import cv2
import open3d as o3d
import traceback
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QListWidget,
                             QListWidgetItem, QPushButton, QSplitter, QGroupBox,
                             QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
                             QMessageBox, QAbstractItemView, QProgressBar, QSpacerItem,
                             QSizePolicy, QProgressDialog, QApplication, QTabWidget,
                             QComboBox, QStackedWidget, QMenu, QSlider, QFormLayout,
                             QFrame) # Added QTabWidget, QComboBox, QStackedWidget, QMenu, QSlider, QFormLayout, QFrame
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import QPixmap, QImage, QIcon, QPainter, QColor, QAction
from pyvistaqt import QtInteractor

# 导入项目模块
from ..utils.point_cloud_handler import PointCloudHandler
from ..utils.stylesheet_manager import StylesheetManager
from .. import DEBUG_MODE # Import global DEBUG_MODE flag

# --- Helper Functions ---

def get_overall_xy_bounds(slices_dict):
    """Calculates the overall XY bounding box for a dictionary of slices."""
    all_bounds_xy = []
    valid_slice_found = False
    for slice_data in slices_dict.values():
        if slice_data is not None and slice_data.n_points > 0:
            b = slice_data.bounds
            if b[0] < b[1] and b[2] < b[3]: # Check if bounds are valid
                 all_bounds_xy.extend(b[0:4]) # Extend with xmin, xmax, ymin, ymax
                 valid_slice_found = True

    if not valid_slice_found or not all_bounds_xy:
        if DEBUG_MODE: print("DEBUG: get_overall_xy_bounds - No valid slice bounds found.")
        return None

    xmin = min(all_bounds_xy[0::4])
    xmax = max(all_bounds_xy[1::4])
    ymin = min(all_bounds_xy[2::4])
    ymax = max(all_bounds_xy[3::4])

    x_range = xmax - xmin if xmax > xmin else 0.1
    y_range = ymax - ymin if ymax > ymin else 0.1
    padding = max(x_range * 0.05, y_range * 0.05, 0.1) # Ensure minimum padding

    return [xmin - padding, xmax + padding, ymin - padding, ymax + padding]

def render_slice_to_image(slice_data, size, overall_xy_bounds=None, is_thumbnail=True):
    """Renders a single slice to a NumPy image array using an off-screen plotter."""
    if DEBUG_MODE: print(f"DEBUG: render_slice_to_image called. is_thumbnail={is_thumbnail}, size={size}")
    if slice_data is None or slice_data.n_points == 0:
        if DEBUG_MODE:
            print("DEBUG: render_slice_to_image - Empty slice data.")
        return None, {}
    plotter = None
    try:
        img_width, img_height = size if isinstance(size, tuple) else (size.width(), size.height())
        if DEBUG_MODE:
            print(f"DEBUG: render_slice_to_image - Creating off-screen plotter with size {img_width}x{img_height}")
        plotter = pv.Plotter(off_screen=True, window_size=[img_width, img_height])
        plotter.set_background('white')
        if DEBUG_MODE: print(f"DEBUG: render_slice_to_image - Plotter created: {plotter}")
        if DEBUG_MODE: print("DEBUG: render_slice_to_image - Adding mesh...")
        if is_thumbnail: actor = plotter.add_mesh(slice_data, color='darkgrey', point_size=1)
        else:
            if 'colors' in slice_data.point_data: actor = plotter.add_mesh(slice_data, scalars='colors', rgb=True, point_size=2)
            else: actor = plotter.add_mesh(slice_data, color='blue', point_size=2)
        if DEBUG_MODE: print(f"DEBUG: render_slice_to_image - Mesh added: {actor is not None}")
        if DEBUG_MODE: print("DEBUG: render_slice_to_image - Setting view_xy()")
        plotter.view_xy()
        bounds_to_use = None
        if overall_xy_bounds:
            zmin = slice_data.bounds[4]; zmax = slice_data.bounds[5]
            # Ensure z range is valid
            if zmax < zmin: zmax = zmin + 1e-6
            bounds_to_use = overall_xy_bounds + [zmin, zmax]
            if DEBUG_MODE: print(f"DEBUG: render_slice_to_image - Resetting camera using OVERALL XY bounds: {bounds_to_use}")
        elif slice_data and slice_data.bounds[0] < slice_data.bounds[1]:
            bounds_to_use = slice_data.bounds
            if DEBUG_MODE: print(f"DEBUG: render_slice_to_image - Resetting camera using SLICE bounds: {bounds_to_use}")
        if bounds_to_use and bounds_to_use[0]<bounds_to_use[1] and bounds_to_use[2]<bounds_to_use[3]: # Check validity before reset
            plotter.reset_camera(bounds=bounds_to_use)
            if DEBUG_MODE: print("DEBUG: render_slice_to_image - Camera reset done.")
        else:
             if DEBUG_MODE: print(f"DEBUG: render_slice_to_image - Invalid bounds for camera reset: {bounds_to_use}")

        if DEBUG_MODE: print("DEBUG: render_slice_to_image - Taking screenshot...")
        img_np = plotter.screenshot(return_img=True)
        if DEBUG_MODE: print(f"DEBUG: render_slice_to_image - Screenshot taken, shape: {img_np.shape if img_np is not None else 'None'}")
        cam = plotter.camera
        if DEBUG_MODE: print(f"DEBUG: render_slice_to_image - Getting camera parameters. Camera object: {cam}")
        view_params = {
            "position": list(cam.position),"focal_point": list(cam.focal_point),"up": list(cam.up),
            "parallel_projection": cam.parallel_projection,"parallel_scale": cam.parallel_scale,
            "slice_bounds": list(slice_data.bounds),"render_window_size": [img_width, img_height],
        }
        if DEBUG_MODE: print(f"DEBUG: render_slice_to_image - View params collected: {view_params}")
        return img_np, view_params
    except Exception as e:
        print(f"ERROR: Error rendering slice to image: {e}")
        if DEBUG_MODE: traceback.print_exc(); return None, {}
    finally:
        if plotter:
            if DEBUG_MODE: print("DEBUG: render_slice_to_image - Closing plotter.")
            try: plotter.close()
            except Exception as close_e: print(f"ERROR: Exception while closing plotter in render_slice_to_image: {close_e}")
        if DEBUG_MODE: print("DEBUG: render_slice_to_image finished.")

def create_density_heatmap(density_matrix, colormap_name='viridis', vmin=None, vmax=None):
    """Generates a QPixmap heatmap from a 2D density matrix."""
    if density_matrix is None or density_matrix.size == 0: return QPixmap()
    try:
        if vmin is None: vmin = np.min(density_matrix)
        if vmax is None: vmax = np.max(density_matrix)
        if vmax <= vmin: vmax = vmin + 1e-6 # Avoid division by zero
        # Use np.errstate to suppress potential warnings during normalization if density_matrix contains NaNs etc.
        with np.errstate(divide='ignore', invalid='ignore'):
             normalized_matrix = (density_matrix - vmin) / (vmax - vmin)
             normalized_matrix[~np.isfinite(normalized_matrix)] = 0 # Handle NaN/inf resulting from division by zero or input NaNs
        normalized_matrix = np.clip(normalized_matrix, 0, 1)
        cmap = plt.get_cmap(colormap_name); colored_matrix_rgba = cmap(normalized_matrix, bytes=True)
        height, width, _ = colored_matrix_rgba.shape
        # Ensure data is contiguous C-order array for QImage
        contiguous_data = np.ascontiguousarray(colored_matrix_rgba.data)
        q_img = QImage(contiguous_data, width, height, width * 4, QImage.Format.Format_RGBA8888)
        if q_img.isNull(): print("ERROR: create_density_heatmap - QImage creation failed."); return QPixmap()
        return QPixmap.fromImage(q_img)
    except Exception as e: print(f"ERROR: Failed to create density heatmap: {e}"); return QPixmap()


# --- Background Threads ---
class SliceProcessingThread(QThread):
    progress = pyqtSignal(int, str)
    slice_ready = pyqtSignal(int, object, tuple)
    thumbnail_ready = pyqtSignal(int, QPixmap, dict)
    finished = pyqtSignal(bool)
    def __init__(self, point_cloud, num_slices, thickness, limit_thickness, thumbnail_size, parent=None):
        super().__init__(parent); self.point_cloud = point_cloud; self.num_slices = num_slices; self.thickness_param = thickness
        self.limit_thickness = limit_thickness; self.thumbnail_size = thumbnail_size; self._is_running = True
    def run(self):
        if DEBUG_MODE: print("DEBUG: SliceProcessingThread run started.")
        if self.point_cloud is None or self.num_slices <= 0 or self.thickness_param <= 0: self.finished.emit(False); return
        try:
            bounds = self.point_cloud.bounds; min_z, max_z = bounds[4], bounds[5]; total_height = max_z - min_z
            if total_height <= 0: self.finished.emit(False); return
            all_points = self.point_cloud.points; has_colors = 'colors' in self.point_cloud.point_data
            if has_colors: all_colors = self.point_cloud['colors']
            step = total_height / self.num_slices; current_start_z = min_z; actual_thickness = self.thickness_param
            if self.limit_thickness:
                max_allowed_thickness = step
                if actual_thickness > max_allowed_thickness: actual_thickness = max_allowed_thickness
            total_steps = self.num_slices * 2; generated_slices = []; height_ranges = []
            for i in range(self.num_slices):
                if not self._is_running: raise InterruptedError("Stopped")
                self.progress.emit(int((i + 1) / total_steps * 100), f"生成切片 {i+1}/{self.num_slices}")
                slice_start_z = current_start_z; slice_end_z = slice_start_z + actual_thickness
                slice_end_z = min(slice_end_z, max_z + 1e-6); slice_start_z = min(slice_start_z, slice_end_z)
                indices = np.where((all_points[:, 2] >= slice_start_z) & (all_points[:, 2] <= slice_end_z))[0]
                height_ranges.append((slice_start_z, slice_end_z))
                if len(indices) > 0:
                    slice_points = all_points[indices]; slice_cloud = pv.PolyData(slice_points)
                    if has_colors and len(all_colors)==len(all_points): slice_cloud['colors'] = all_colors[indices]
                    generated_slices.append(slice_cloud); self.slice_ready.emit(i, slice_cloud, (slice_start_z, slice_end_z))
                else:
                    generated_slices.append(None); self.slice_ready.emit(i, None, (slice_start_z, slice_end_z))
                current_start_z += step
            temp_slices_dict = {i: s for i, s in enumerate(generated_slices)}
            overall_xy_bounds = get_overall_xy_bounds(temp_slices_dict)
            for i in range(self.num_slices):
                if not self._is_running: raise InterruptedError("Stopped")
                self.progress.emit(int((self.num_slices + i + 1) / total_steps * 100), f"生成缩略图 {i+1}/{self.num_slices}")
                slice_data = generated_slices[i]
                img_np, view_params = render_slice_to_image(slice_data, self.thumbnail_size, overall_xy_bounds, is_thumbnail=True)
                metadata = {"index": i,"height_range": height_ranges[i],"view_params": view_params,"is_empty": slice_data is None or slice_data.n_points == 0}
                if img_np is not None:
                    try:
                        h, w, ch = img_np.shape; image_data_bytes = img_np.tobytes()
                        q_img = QImage(image_data_bytes, w, h, w * ch, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_img)
                        if pixmap.isNull(): raise ValueError("Created QPixmap is null")
                        scaled_pixmap = pixmap.scaled(self.thumbnail_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        self.thumbnail_ready.emit(i, scaled_pixmap, metadata)
                    except Exception as qimage_err:
                         print(f"ERROR: SliceProcessingThread - Failed QImage/QPixmap thumbnail {i}: {qimage_err}")
                         placeholder_pixmap = QPixmap(self.thumbnail_size); placeholder_pixmap.fill(Qt.GlobalColor.darkRed); self.thumbnail_ready.emit(i, placeholder_pixmap, metadata)
                else:
                    placeholder_pixmap = QPixmap(self.thumbnail_size); placeholder_pixmap.fill(Qt.GlobalColor.lightGray); painter = QPainter(placeholder_pixmap); painter.drawText(placeholder_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, f"Slice {i}\n(Empty)"); painter.end(); self.thumbnail_ready.emit(i, placeholder_pixmap, metadata)
            self.finished.emit(True)
        except InterruptedError: print("INFO: Slice thread stopped."); self.finished.emit(False)
        except Exception as e: print(f"ERROR: Slice thread error: {e}"); self.finished.emit(False)
    def stop(self): self._is_running = False

class DensityProcessingThread(QThread):
    progress = pyqtSignal(int, str)
    density_map_ready = pyqtSignal(int, np.ndarray, QPixmap, dict)
    finished = pyqtSignal(bool)
    def __init__(self, slices_dict, overall_xy_bounds, grid_resolution, colormap_name, parent=None):
        super().__init__(parent); self.slices_dict = slices_dict; self.overall_xy_bounds = overall_xy_bounds
        self.grid_resolution = grid_resolution; self.colormap_name = colormap_name; self._is_running = True
    def run(self):
        if DEBUG_MODE: print("DEBUG: DensityProcessingThread run started.")
        if not self.slices_dict or self.overall_xy_bounds is None: self.finished.emit(False); return
        xmin, xmax, ymin, ymax = self.overall_xy_bounds; bins = [self.grid_resolution, self.grid_resolution]; range_xy = [[xmin, xmax], [ymin, ymax]]
        num_slices = len(self.slices_dict); sorted_indices = sorted(self.slices_dict.keys()); max_density = 0; all_matrices = {}
        try:
            if DEBUG_MODE: print("DEBUG: DensityProcessingThread - First pass: Calculating densities...")
            for i, index in enumerate(sorted_indices):
                if not self._is_running: raise InterruptedError("Stopped")
                self.progress.emit(int(((i + 1) / (num_slices * 2)) * 100), f"计算密度 {index+1}/{num_slices}")
                slice_data = self.slices_dict.get(index)
                if slice_data is not None and slice_data.n_points > 0:
                    points_xy = slice_data.points[:, 0:2]
                    density_matrix, _, _ = np.histogram2d(points_xy[:, 0], points_xy[:, 1], bins=bins, range=range_xy)
                    all_matrices[index] = density_matrix; current_max = np.max(density_matrix); max_density = max(max_density, current_max)
                else: all_matrices[index] = np.zeros(bins)
            if DEBUG_MODE: print(f"DEBUG: DensityProcessingThread - Max density found: {max_density}")
            if DEBUG_MODE: print("DEBUG: DensityProcessingThread - Second pass: Generating heatmaps...")
            for i, index in enumerate(sorted_indices):
                 if not self._is_running: raise InterruptedError("Stopped")
                 self.progress.emit(int(((num_slices + i + 1) / (num_slices * 2)) * 100), f"生成热力图 {index+1}/{num_slices}")
                 density_matrix = all_matrices[index]; heatmap_pixmap = create_density_heatmap(density_matrix, self.colormap_name, vmin=0, vmax=max_density)
                 density_params = {"grid_resolution": self.grid_resolution,"colormap": self.colormap_name,"xy_bounds": self.overall_xy_bounds,"max_density_scale": max_density}
                 self.density_map_ready.emit(index, density_matrix, heatmap_pixmap, density_params)
            self.finished.emit(True)
        except InterruptedError: print("INFO: Density thread stopped."); self.finished.emit(False)
        except Exception as e: print(f"ERROR: Density thread error: {e}"); self.finished.emit(False)
    def stop(self): self._is_running = False

class BatchLogicOpThread(QThread):
    progress = pyqtSignal(int, str) # percentage, status
    result_ready = pyqtSignal(dict, dict) # results_matrices {key: matrix}, results_pixmaps {key: pixmap}
    finished = pyqtSignal(bool) # success

    def __init__(self, density_matrices_dict, operation, step_k, loop, diff_colormap, parent=None):
        super().__init__(parent)
        self.density_matrices = density_matrices_dict
        self.operation = operation
        self.step_k = step_k
        self.loop = loop
        self.diff_colormap = diff_colormap
        self._is_running = True
        if DEBUG_MODE: print(f"DEBUG: BatchLogicOpThread initialized. Op={operation}, k={step_k}, loop={loop}")

    def run(self):
        if DEBUG_MODE: print("DEBUG: BatchLogicOpThread run started.")
        if not self.density_matrices or self.step_k <= 0:
            self.finished.emit(False); return

        results_matrices = {}
        results_pixmaps = {}
        sorted_indices = sorted(self.density_matrices.keys())
        n = len(sorted_indices)
        total_ops = n if not self.loop else n # Number of operations to perform

        try:
            for i in range(n):
                if not self._is_running: raise InterruptedError("Stopped")
                self.progress.emit(int(((i + 1) / total_ops) * 100), f"计算批量运算 {i+1}/{total_ops}")

                idx_i = sorted_indices[i]
                idx_j_raw = i + self.step_k # Index of the second operand

                if not self.loop and idx_j_raw >= n:
                    if DEBUG_MODE: print(f"DEBUG: Batch op skipping index {i}, step {self.step_k} goes out of bounds (loop=False).")
                    continue # Skip if looping is off and index goes out of bounds

                idx_j = sorted_indices[idx_j_raw % n] # Apply modulo for looping or normal indexing

                matrix_i = self.density_matrices.get(idx_i)
                matrix_j = self.density_matrices.get(idx_j)

                result_key = f"Op({idx_j},{idx_i})" # Key represents Op(B, A) where B=i+k

                if matrix_i is None or matrix_j is None or matrix_i.shape != matrix_j.shape:
                    print(f"Warning: Skipping batch op for indices {idx_i}, {idx_j} due to missing or mismatched matrices.")
                    results_matrices[result_key] = None # Store None to indicate failure for this pair
                    results_pixmaps[result_key] = QPixmap()
                    continue

                # Perform Operation (Similar to _compute_logic_operation)
                result_matrix = None
                colormap = 'binary' # Default for logical ops
                vmin, vmax = 0, 1

                if self.operation == "差分 (A-B)": # Interpreted as a[i+k] - a[i]
                    result_matrix = matrix_j - matrix_i
                    abs_max = np.max(np.abs(result_matrix)) if np.any(result_matrix) else 1
                    vmin, vmax = -abs_max, abs_max
                    colormap = self.diff_colormap
                elif self.operation == "差分 (B-A)": # Interpreted as a[i] - a[i+k]
                     result_matrix = matrix_i - matrix_j
                     abs_max = np.max(np.abs(result_matrix)) if np.any(result_matrix) else 1
                     vmin, vmax = -abs_max, abs_max
                     colormap = self.diff_colormap
                elif self.operation == "并集 (A | B)":
                    result_matrix = np.logical_or(matrix_i > 0, matrix_j > 0).astype(float)
                elif self.operation == "交集 (A & B)":
                    result_matrix = np.logical_and(matrix_i > 0, matrix_j > 0).astype(float)
                # Add other ops...

                if result_matrix is not None:
                    results_matrices[result_key] = result_matrix
                    results_pixmaps[result_key] = create_density_heatmap(result_matrix, colormap, vmin=vmin, vmax=vmax)
                else:
                     results_matrices[result_key] = None
                     results_pixmaps[result_key] = QPixmap()

            if DEBUG_MODE: print(f"DEBUG: BatchLogicOpThread finished successfully with {len(results_matrices)} results.")
            self.result_ready.emit(results_matrices, results_pixmaps)
            self.finished.emit(True)

        except InterruptedError: print("INFO: Batch logic thread stopped."); self.finished.emit(False)
        except Exception as e: print(f"ERROR: Batch logic thread error: {e}"); self.finished.emit(False)

    def stop(self): self._is_running = False


# --- Main Window Class ---
class BatchSliceViewerWindow(QWidget):
    BITMAP_EXPORT_RESOLUTION = (1024, 1024)
    DEFAULT_DENSITY_RESOLUTION = 512
    AVAILABLE_COLORMAPS = sorted([cm for cm in plt.colormaps() if not cm.endswith("_r")]) # Get base colormaps
    DIVERGING_COLORMAPS = ['RdBu', 'bwr', 'coolwarm', 'seismic', 'PiYG', 'PRGn', 'BrBG', 'PuOr']

    def __init__(self, point_cloud, source_filename="Unknown", parent=None):
        super().__init__(parent)
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow __init__ started.")
        self.setWindowTitle("批量切片查看器")
        self.setMinimumSize(1100, 750)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setStyleSheet(StylesheetManager.get_light_theme())

        self.original_point_cloud = point_cloud
        self.source_filename = source_filename
        # Data storage
        self.current_slices = {}
        self.slice_metadata = {}
        self.density_matrices = {}
        self.density_pixmaps = {}
        self.density_params = {} # Stores params used for the *last* density calculation
        self.logic_op_result_matrix = None # For single A-B op
        self.logic_op_result_pixmap = None # For single A-B op
        self.batch_op_results_matrices = {} # {key: matrix}
        self.batch_op_results_pixmaps = {} # {key: pixmap}
        self.batch_op_keys = [] # Ordered keys for slider access
        # UI State
        self.selected_slice_a = None
        self.selected_slice_b = None
        self.current_display_mode = "3D" # "3D", "DensitySlice", "DensitySingleResult", "DensityBatchResult"
        self.current_display_slice_index = 0 # Index for DensitySlice mode
        self.current_batch_result_index = 0 # Index for DensityBatchResult mode
        # Threads & Timers
        self.slice_processing_thread = None
        self.density_processing_thread = None
        self.batch_logic_op_thread = None
        self.batch_play_timer = QTimer(self)
        self.batch_play_timer.timeout.connect(self._advance_batch_slider)
        self.progress_dialog = None
        self.plotter = None

        self.setup_ui()
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow __init__ finished.")

    # --- UI Setup ---
    def setup_ui(self):
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow setup_ui started.")
        main_layout = QHBoxLayout(self)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)

        # --- Left Panel (Thumbnails + Context Menu) ---
        self.setup_left_panel()

        # --- Center Panel (Stacked View) ---
        self.setup_center_panel()

        # --- Right Panel (Tabbed Controls) ---
        self.setup_right_panel()
        self.splitter.setSizes([280, 600, 300]) # Adjusted right panel size
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow setup_ui finished.")

    def setup_left_panel(self):
        # ... (Implementation remains the same, includes context menu) ...
        left_panel = QWidget(); left_layout = QVBoxLayout(left_panel); left_layout.setContentsMargins(5, 5, 5, 5)
        list_group = QGroupBox("切片预览 (顶视图)"); list_group_layout = QVBoxLayout(list_group)
        self.slice_list_widget = QListWidget(); self.slice_list_widget.setViewMode(QListWidget.ViewMode.IconMode); self.slice_list_widget.setIconSize(QSize(128, 128)); self.slice_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust); self.slice_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection); self.slice_list_widget.itemSelectionChanged.connect(self._on_selection_changed); self.slice_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu); self.slice_list_widget.customContextMenuRequested.connect(self._show_list_context_menu)
        list_group_layout.addWidget(self.slice_list_widget)
        list_button_layout = QHBoxLayout(); select_all_btn = QPushButton("全选"); select_all_btn.clicked.connect(self.slice_list_widget.selectAll); deselect_all_btn = QPushButton("全不选"); deselect_all_btn.clicked.connect(self.slice_list_widget.clearSelection); export_selected_btn = QPushButton("导出选中"); export_selected_btn.clicked.connect(self._export_selected_data)
        list_button_layout.addWidget(select_all_btn); list_button_layout.addWidget(deselect_all_btn); list_button_layout.addStretch(); list_button_layout.addWidget(export_selected_btn)
        list_group_layout.addLayout(list_button_layout); left_layout.addWidget(list_group); self.splitter.addWidget(left_panel)


    def setup_center_panel(self):
        # ... (Implementation remains the same) ...
        center_panel = QWidget(); center_layout = QVBoxLayout(center_panel); center_layout.setContentsMargins(0, 0, 0, 0)
        self.center_stacked_widget = QStackedWidget(); center_layout.addWidget(self.center_stacked_widget)
        self.plotter_widget = QWidget(); plotter_layout = QVBoxLayout(self.plotter_widget); plotter_layout.setContentsMargins(0,0,0,0)
        try:
            self.plotter = QtInteractor(parent=self.plotter_widget); plotter_layout.addWidget(self.plotter); QTimer.singleShot(200, self._initialize_plotter_view)
        except Exception as e:
             print(f"ERROR: Failed to create QtInteractor: {e}"); self.plotter = None
             error_label = QLabel(f"无法初始化3D视图...\n错误: {e}"); error_label.setAlignment(Qt.AlignmentFlag.AlignCenter); plotter_layout.addWidget(error_label)
        self.center_stacked_widget.addWidget(self.plotter_widget)
        self.density_view_label = QLabel("请先计算密度图"); self.density_view_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.density_view_label.setScaledContents(False)
        self.center_stacked_widget.addWidget(self.density_view_label); self.splitter.addWidget(center_panel)

    def setup_right_panel(self):
        # ... (Implementation remains the same, calls tab setup methods) ...
        right_panel = QWidget(); right_panel.setMinimumWidth(300); right_panel.setMaximumWidth(450); right_layout = QVBoxLayout(right_panel); right_layout.setContentsMargins(5, 5, 5, 5)
        self.right_tab_widget = QTabWidget(); right_layout.addWidget(self.right_tab_widget)
        view_control_tab = QWidget(); vc_layout = QVBoxLayout(view_control_tab); self.setup_view_control_tab(vc_layout); self.right_tab_widget.addTab(view_control_tab, "视图控制")
        density_tab = QWidget(); density_layout = QVBoxLayout(density_tab); self.setup_density_analysis_tab(density_layout); self.right_tab_widget.addTab(density_tab, "密度分析")
        self.right_tab_widget.currentChanged.connect(self._handle_tab_change); self.splitter.addWidget(right_panel)

    def setup_view_control_tab(self, layout):
        # ... (Implementation mostly the same, ensure tooltips updated) ...
        slicing_group = QGroupBox("切片参数")
        slicing_layout = QVBoxLayout(slicing_group)
        num_slices_layout = QHBoxLayout()
        num_slices_layout.addWidget(QLabel("切片数量:"))
        self.num_slices_spin = QSpinBox()
        self.num_slices_spin.setRange(1, 500)
        self.num_slices_spin.setValue(10)
        num_slices_layout.addWidget(self.num_slices_spin); slicing_layout.addLayout(num_slices_layout)
        thickness_layout = QHBoxLayout()
        thickness_layout.addWidget(QLabel("单片厚度 (米):"))
        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setRange(0.01, 5.0)
        self.thickness_spin.setSingleStep(0.01)
        self.thickness_spin.setValue(0.10)
        self.thickness_spin.setDecimals(3)
        thickness_layout.addWidget(self.thickness_spin)
        slicing_layout.addLayout(thickness_layout)
        self.limit_thickness_check = QCheckBox("无重叠")
        self.limit_thickness_check.setChecked(True)
        self.limit_thickness_check.setToolTip("确保切片厚度不超过 (总高度/切片数)")
        slicing_layout.addWidget(self.limit_thickness_check)
        layout.addWidget(slicing_group)

        # 3D Visualization Parameters Group
        viz_group = QGroupBox("3D视图参数")
        viz_layout = QVBoxLayout(viz_group)
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("垂直偏移:"))
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(0.0, 10.0)
        self.offset_spin.setSingleStep(0.1)
        self.offset_spin.setValue(0.5)
        self.offset_spin.valueChanged.connect(self._update_3d_view_presentation)
        offset_layout.addWidget(self.offset_spin)
        viz_layout.addLayout(offset_layout)
        point_size_layout = QHBoxLayout()
        point_size_layout.addWidget(QLabel("点大小:"))
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 10)
        self.point_size_spin.setValue(2)
        self.point_size_spin.valueChanged.connect(self._update_3d_view_presentation)
        point_size_layout.addWidget(self.point_size_spin)
        viz_layout.addLayout(point_size_layout)
        self.use_color_check = QCheckBox("显示原始颜色")
        self.use_color_check.setChecked(True)
        self.use_color_check.stateChanged.connect(self._update_3d_view_presentation)
        viz_layout.addWidget(self.use_color_check)
        layout.addWidget(viz_group)
        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout(action_group)
        generate_slices_btn = QPushButton("生成切片并预览")
        generate_slices_btn.setToolTip("计算切片数据和缩略图，更新3D视图")
        generate_slices_btn.clicked.connect(self._start_slice_processing)
        action_layout.addWidget(generate_slices_btn)
        export_all_btn = QPushButton("导出所有数据")
        export_all_btn.setToolTip("导出所有切片的PCD、位图、元数据和密度数据(如果已计算)")
        export_all_btn.clicked.connect(self._export_all_data)
        action_layout.addWidget(export_all_btn); layout.addWidget(action_group)
        layout.addStretch(); close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close); layout.addWidget(close_btn)


    def setup_density_analysis_tab(self, layout):
        """Populates the 'Density Analysis' tab with refined controls."""
        # Density Calculation Group
        density_calc_group = QGroupBox("密度计算"); dcg_layout = QFormLayout(density_calc_group)
        self.density_resolution_combo = QComboBox(); self.density_resolution_combo.addItems(["256x256", "512x512", "1024x1024", "2048x2048"]); self.density_resolution_combo.setCurrentText(f"{self.DEFAULT_DENSITY_RESOLUTION}x{self.DEFAULT_DENSITY_RESOLUTION}"); dcg_layout.addRow("密度网格分辨率:", self.density_resolution_combo)
        self.density_colormap_combo = QComboBox(); self.density_colormap_combo.addItems(self.AVAILABLE_COLORMAPS); self.density_colormap_combo.setCurrentText("viridis"); dcg_layout.addRow("颜色映射 (主):", self.density_colormap_combo)
        self.update_density_btn = QPushButton("计算/更新密度图"); self.update_density_btn.setToolTip("计算所有切片的密度矩阵和热力图"); self.update_density_btn.clicked.connect(self._start_density_processing); dcg_layout.addRow(self.update_density_btn); layout.addWidget(density_calc_group)

        # Display Control Group
        display_group = QGroupBox("显示控制"); dg_layout = QFormLayout(display_group)
        self.density_display_combo = QComboBox(); self.density_display_combo.addItems(["选中切片", "单次运算结果", "批量运算结果"]); self.density_display_combo.setEnabled(False); self.density_display_combo.currentIndexChanged.connect(self._update_display_mode); dg_layout.addRow("显示内容:", self.density_display_combo)
        self.diff_colormap_combo = QComboBox(); self.diff_colormap_combo.addItems(self.DIVERGING_COLORMAPS); self.diff_colormap_combo.setCurrentText("RdBu"); dg_layout.addRow("颜色映射 (差分):", self.diff_colormap_combo)
        # Add colorbar checkbox later if needed
        layout.addWidget(display_group)

        # Single Logic Operation Group
        logic_op_group = QGroupBox("单次逻辑运算"); log_layout = QVBoxLayout(logic_op_group)
        slice_a_layout = QHBoxLayout(); slice_a_layout.addWidget(QLabel("切片 A:")); self.slice_a_label = QLabel("未选"); self.slice_a_label.setAlignment(Qt.AlignmentFlag.AlignCenter); slice_a_layout.addWidget(self.slice_a_label, 1); self.clear_slice_a_btn = QPushButton("清除"); self.clear_slice_a_btn.clicked.connect(lambda: self._clear_logic_operand('A')); slice_a_layout.addWidget(self.clear_slice_a_btn); log_layout.addLayout(slice_a_layout)
        slice_b_layout = QHBoxLayout(); slice_b_layout.addWidget(QLabel("切片 B:")); self.slice_b_label = QLabel("未选"); self.slice_b_label.setAlignment(Qt.AlignmentFlag.AlignCenter); slice_b_layout.addWidget(self.slice_b_label, 1); self.clear_slice_b_btn = QPushButton("清除"); self.clear_slice_b_btn.clicked.connect(lambda: self._clear_logic_operand('B')); slice_b_layout.addWidget(self.clear_slice_b_btn); log_layout.addLayout(slice_b_layout)
        op_layout = QHBoxLayout(); op_layout.addWidget(QLabel("操作:")); self.logic_op_combo = QComboBox(); self.logic_op_combo.addItems(["差分 (A-B)", "差分 (B-A)", "并集 (A | B)", "交集 (A & B)"]); op_layout.addWidget(self.logic_op_combo, 1); log_layout.addLayout(op_layout)
        self.compute_logic_op_btn = QPushButton("计算单次运算"); self.compute_logic_op_btn.clicked.connect(self._compute_logic_operation); self.compute_logic_op_btn.setEnabled(False); log_layout.addWidget(self.compute_logic_op_btn); layout.addWidget(logic_op_group)

        # Batch Logic Operation Group
        batch_op_group = QGroupBox("批量逻辑运算"); bog_layout = QVBoxLayout(batch_op_group)
        batch_op_param_layout = QFormLayout()
        self.batch_op_combo = QComboBox(); self.batch_op_combo.addItems(["差分 (a[i+k] - a[i])", "差分 (a[i] - a[i+k])", "并集", "交集"]); batch_op_param_layout.addRow("运算类型:", self.batch_op_combo)
        self.batch_op_step_spin = QSpinBox(); self.batch_op_step_spin.setRange(1, 50); self.batch_op_step_spin.setValue(1); batch_op_param_layout.addRow("步长 k:", self.batch_op_step_spin)
        self.batch_op_loop_check = QCheckBox("循环应用"); self.batch_op_loop_check.setChecked(False); batch_op_param_layout.addRow("", self.batch_op_loop_check)
        bog_layout.addLayout(batch_op_param_layout)
        self.execute_batch_op_btn = QPushButton("执行批量运算"); self.execute_batch_op_btn.clicked.connect(self._execute_batch_logic_op); self.execute_batch_op_btn.setEnabled(False); bog_layout.addWidget(self.execute_batch_op_btn)
        # Batch Result Preview
        self.batch_result_slider = QSlider(Qt.Orientation.Horizontal); self.batch_result_slider.setEnabled(False); self.batch_result_slider.valueChanged.connect(self._update_batch_preview_slider); bog_layout.addWidget(self.batch_result_slider)
        batch_play_layout = QHBoxLayout(); self.batch_result_label = QLabel("结果: 0/0"); batch_play_layout.addWidget(self.batch_result_label, 1); self.batch_play_btn = QPushButton("播放"); self.batch_play_btn.setCheckable(True); self.batch_play_btn.toggled.connect(self._toggle_batch_play); self.batch_play_btn.setEnabled(False); batch_play_layout.addWidget(self.batch_play_btn); bog_layout.addLayout(batch_play_layout)
        layout.addWidget(batch_op_group)

        layout.addStretch()

    # --- Initialization & View Update ---
    # ... (_initialize_plotter_view remains the same) ...
    def _initialize_plotter_view(self):
        if self.plotter is None: return
        try:
            if DEBUG_MODE: print("DEBUG: _initialize_plotter_view - Setting background...")
            self.plotter.set_background("white")
            self.plotter.add_text("请在右侧面板设置参数并点击“生成切片”", name="init_text")
            self.plotter.render()
        except Exception as e: print(f"ERROR: Plotter init failed: {e}")

    def _handle_tab_change(self, index):
        """Switches the center view and updates display mode."""
        if DEBUG_MODE: print(f"DEBUG: Tab changed to index {index}")
        if index == 0: # View Control Tab
            self.current_display_mode = "3D"
            self.center_stacked_widget.setCurrentIndex(0)
            if self.plotter: self.plotter.render() # Ensure plotter renders if switched back
        elif index == 1: # Density Analysis Tab
            # Don't force display mode here, let _update_display_mode handle it
            # based on density_display_combo's current value
            self.center_stacked_widget.setCurrentIndex(1)
            self._update_density_display_label() # Update label based on combo

    def _update_display_mode(self):
        """Updates the internal display mode state and refreshes the center view."""
        selected_text = self.density_display_combo.currentText()
        if DEBUG_MODE: print(f"DEBUG: Display mode combo changed to: {selected_text}")
        if selected_text == "选中切片":
            self.current_display_mode = "DensitySlice"
        elif selected_text == "单次运算结果":
            self.current_display_mode = "DensitySingleResult"
        elif selected_text == "批量运算结果":
            self.current_display_mode = "DensityBatchResult"
            # Reset slider to beginning when switching to batch view
            self.current_batch_result_index = 0
            self.batch_result_slider.setValue(0)
        self._update_center_view() # Update display based on new mode

    def _update_center_view(self):
        """Updates the central widget display based on current_display_mode."""
        if self.current_display_mode == "3D":
            if self.center_stacked_widget.currentIndex() != 0:
                 if DEBUG_MODE: print("DEBUG: _update_center_view - Switching to 3D view (stack index 0)")
                 self.center_stacked_widget.setCurrentIndex(0)
            # 3D view updates happen via _update_3d_view_presentation
        else: # DensitySlice, DensitySingleResult, or DensityBatchResult
            if self.center_stacked_widget.currentIndex() != 1:
                 if DEBUG_MODE: print("DEBUG: _update_center_view - Switching to Density view (stack index 1)")
                 self.center_stacked_widget.setCurrentIndex(1)
            self._update_density_display_label()

    # def _update_density_display_label(self):
    #     """Updates the content of the density view QLabel based on current mode."""
    #     pixmap_to_show = QPixmap()
    #     label_text = ""
    #     if DEBUG_MODE: print(f"DEBUG: _update_density_display_label - Mode: '{self.current_display_mode}'")
    #
    #     if self.current_display_mode == "DensitySlice":
    #         selected_items = self.slice_list_widget.selectedItems()
    #         target_index = -1
    #         if selected_items: target_index = selected_items[0].data(Qt.ItemDataRole.UserRole)
    #         elif self.density_pixmaps: target_index = sorted(self.density_pixmaps.keys())[0] if self.density_pixmaps else -1
    #
    #         if target_index != -1 and target_index in self.density_pixmaps:
    #             self.current_display_slice_index = target_index
    #             pixmap_to_show = self.density_pixmaps.get(target_index, QPixmap())
    #             label_text = f"切片 {target_index} 密度图"
    #         else: label_text = f"切片密度图不可用 (索引: {target_index})"
    #
    #     elif self.current_display_mode == "DensitySingleResult":
    #         if self.logic_op_result_pixmap and not self.logic_op_result_pixmap.isNull():
    #             pixmap_to_show = self.logic_op_result_pixmap
    #             label_text = "单次逻辑运算结果"
    #         else: label_text = "无单次逻辑运算结果"
    #
    #     elif self.current_display_mode == "DensityBatchResult":
    #         if self.batch_op_keys: # Check if batch results exist
    #             current_key = self.batch_op_keys[self.current_batch_result_index]
    #             pixmap_to_show = self.batch_op_results_pixmaps.get(current_key, QPixmap())
    #             label_text = f"批量运算结果: {current_key}"
    #             # Update slider label
    #             self.batch_result_label.setText(f"结果: {self.current_batch_result_index + 1}/{len(self.batch_op_keys)}")
    #         else: label_text = "无批量运算结果"
    #     else: # Default or 3D mode (shouldn't reach here if called correctly)
    #         label_text = "请先计算密度图"
    #
    #
    #     if not pixmap_to_show.isNull():
    #         scaled_pixmap = pixmap_to_show.scaled(self.density_view_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    #         self.density_view_label.setPixmap(scaled_pixmap); self.density_view_label.setText("")
    #     else:
    #          self.density_view_label.setText(label_text); self.density_view_label.setPixmap(QPixmap())

    # --- Processing ---
    def _start_slice_processing(self):
        # ... (Implementation from previous response, ensure clear logic added) ...
        if DEBUG_MODE: print("DEBUG: _start_slice_processing called.")
        # (Checks for plotter, point_cloud, running thread)
        if self.plotter is None or self.original_point_cloud is None or self.original_point_cloud.n_points == 0 or (self.slice_processing_thread and self.slice_processing_thread.isRunning()): return
        if DEBUG_MODE: print("DEBUG: _start_slice_processing - Clearing previous results.")
        self.slice_list_widget.clear(); self.current_slices.clear()
        self.slice_metadata.clear(); self.density_matrices.clear()
        self.density_pixmaps.clear(); self.density_params.clear()
        self.batch_op_results_matrices.clear()
        self.batch_op_results_pixmaps.clear()
        self.batch_op_keys.clear()
        self.selected_slice_a = None; self.selected_slice_b = None
        self.logic_op_result_matrix = None
        self.logic_op_result_pixmap = None
        self._update_logic_op_ui()
        self._update_batch_op_ui()
        try:
            self.plotter.clear(); self.plotter.remove_actor("init_text", render=False)
            self.plotter.add_text("正在生成切片...", name="status_text")
            self.plotter.render()
            QApplication.processEvents()
        except Exception as e: print(f"ERROR: Failed to clear plotter: {e}")
        num_slices = self.num_slices_spin.value(); thickness = self.thickness_spin.value(); limit_thickness = self.limit_thickness_check.isChecked(); thumbnail_size = self.slice_list_widget.iconSize()
        self.progress_dialog = QProgressDialog("正在处理切片...", "取消", 0, 100, self); self.progress_dialog.setWindowTitle("切片处理")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True); self.progress_dialog.setAutoReset(True)
        self.progress_dialog.canceled.connect(self._cancel_processing)
        QTimer.singleShot(50, self.progress_dialog.show)
        self.slice_processing_thread = SliceProcessingThread(
            self.original_point_cloud, num_slices, thickness, limit_thickness, thumbnail_size
        )
        self.slice_processing_thread.progress.connect(self._update_progress)
        self.slice_processing_thread.slice_ready.connect(self._collect_slice_data)
        self.slice_processing_thread.thumbnail_ready.connect(self._add_thumbnail_item)
        self.slice_processing_thread.finished.connect(self._slice_processing_finished)
        self.slice_processing_thread.start()
        if DEBUG_MODE: print("DEBUG: _start_slice_processing - Thread started.")

    def _start_density_processing(self):
        # ... (Implementation from previous response, ensure clear logic added) ...
        if DEBUG_MODE: print("DEBUG: _start_density_processing called.")
        if not self.current_slices: QMessageBox.warning(self, "无切片", "请先生成切片数据。"); return
        if self.density_processing_thread and self.density_processing_thread.isRunning(): QMessageBox.warning(self, "处理中", "正在计算密度图..."); return
        if DEBUG_MODE: print("DEBUG: _start_density_processing - Clearing previous density and logic op results.")
        self.density_matrices.clear(); self.density_pixmaps.clear(); self.density_params.clear(); self.logic_op_result_matrix = None; self.logic_op_result_pixmap = None; self.batch_op_results_matrices.clear(); self.batch_op_results_pixmaps.clear(); self.batch_op_keys.clear()
        self.selected_slice_a = None; self.selected_slice_b = None; self._update_logic_op_ui(); self._update_batch_op_ui()
        try:
            if self.density_display_combo.count() > 0: self.density_display_combo.setCurrentIndex(0) # Reset display mode
        except Exception as e: print(f"Warning: Error resetting density display combo: {e}")
        self._update_center_view() # Update view to show status/clear old view
        resolution_text = self.density_resolution_combo.currentText(); grid_resolution = int(resolution_text.split('x')[0]) if 'x' in resolution_text else self.DEFAULT_DENSITY_RESOLUTION; colormap_name = self.density_colormap_combo.currentText(); overall_xy_bounds = get_overall_xy_bounds(self.current_slices)
        if overall_xy_bounds is None: QMessageBox.warning(self, "无有效边界", "无法计算有效XY边界。"); return
        self.progress_dialog = QProgressDialog("正在计算密度图...", "取消", 0, 100, self); self.progress_dialog.setWindowTitle("密度计算"); self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal); self.progress_dialog.setAutoClose(True); self.progress_dialog.setAutoReset(True); self.progress_dialog.canceled.connect(self._cancel_processing)
        QTimer.singleShot(50, self.progress_dialog.show)

        # Start thread
        if DEBUG_MODE: print(f"DEBUG: Starting DensityProcessingThread. Resolution={grid_resolution}, Colormap={colormap_name}")
        self.density_processing_thread = DensityProcessingThread(self.current_slices, overall_xy_bounds, grid_resolution, colormap_name)
        self.density_processing_thread.progress.connect(self._update_progress); self.density_processing_thread.density_map_ready.connect(self._collect_density_data); self.density_processing_thread.finished.connect(self._density_processing_finished)
        self.density_processing_thread.start()

    def _update_progress(self, value, message):
        """Update progress dialog (connected to thread signals)."""
        if self.progress_dialog:
            try:
                # Ensure value is within range, sometimes threads might emit slightly off
                clamped_value = max(0, min(value, 100))
                self.progress_dialog.setValue(clamped_value)
                self.progress_dialog.setLabelText(message)
                # Allow UI to refresh during updates
                QApplication.processEvents()
            except RuntimeError:  # Handle case where dialog might have been closed
                if DEBUG_MODE: print("DEBUG: Progress dialog accessed after deletion.")
                self.progress_dialog = None  # Reset reference

    def _cancel_processing(self):
        """Cancel any running background thread (connected to progress dialog)."""
        if DEBUG_MODE: print("DEBUG: _cancel_processing called from progress dialog.")
        if self.slice_processing_thread and self.slice_processing_thread.isRunning():
            print("INFO: Canceling slice processing...")
            self.slice_processing_thread.stop()
        if self.density_processing_thread and self.density_processing_thread.isRunning():
            print("INFO: Canceling density processing...")
            self.density_processing_thread.stop()
        if self.batch_logic_op_thread and self.batch_logic_op_thread.isRunning():
            print("INFO: Canceling batch logic op processing...")
            self.batch_logic_op_thread.stop()
        # Progress dialog cancellation signal handles closing the dialog itself

    def _collect_slice_data(self, index, slice_data, height_range):
        """Collect slice data from the SliceProcessingThread."""
        if DEBUG_MODE: print(
            f"DEBUG: _collect_slice_data received for index {index}. Data valid: {slice_data is not None}")
        self.current_slices[index] = slice_data
        # Note: Corresponding metadata (height range) is now collected when thumbnail is ready

    def _add_thumbnail_item(self, index, pixmap, metadata):
        """Add thumbnail item to the list (connected to SliceProcessingThread)."""
        if DEBUG_MODE: print(
            f"DEBUG: _add_thumbnail_item received for index {index}. Pixmap valid: {not pixmap.isNull()}")
        item = QListWidgetItem(f"Slice {index}")
        item.setIcon(QIcon(pixmap))
        item.setData(Qt.ItemDataRole.UserRole, index)
        # Insert items in sorted order based on index
        # Find the correct position to insert the new item
        insert_row = 0
        for i in range(self.slice_list_widget.count()):
            existing_item = self.slice_list_widget.item(i)
            existing_index = existing_item.data(Qt.ItemDataRole.UserRole)
            if index > existing_index:
                insert_row = i + 1
            else:
                break  # Found the position
        self.slice_list_widget.insertItem(insert_row, item)

        # Store metadata (includes height_range now)
        self.slice_metadata[index] = metadata
        if DEBUG_MODE: print(f"DEBUG: _add_thumbnail_item - Item inserted for index {index}, metadata stored.")

    def _execute_batch_logic_op(self):
        """Starts the background thread for batch logic operations."""
        if DEBUG_MODE: print("DEBUG: _execute_batch_logic_op called.")
        if not self.density_matrices: QMessageBox.warning(self, "无密度数据", "请先计算密度图。"); return
        if self.batch_logic_op_thread and self.batch_logic_op_thread.isRunning(): QMessageBox.warning(self, "处理中", "正在执行批量运算..."); return

        # Clear previous batch results
        self.batch_op_results_matrices.clear(); self.batch_op_results_pixmaps.clear(); self.batch_op_keys.clear()
        self._update_batch_op_ui() # Disable slider etc.

        operation_text = self.batch_op_combo.currentText()
        # Map UI text to internal operation identifier if needed, or pass text directly
        operation = operation_text.split(" ")[0] # Simple mapping for now
        step_k = self.batch_op_step_spin.value()
        loop = self.batch_op_loop_check.isChecked()
        diff_colormap = self.diff_colormap_combo.currentText()

        self.progress_dialog = QProgressDialog("正在执行批量运算...", "取消", 0, 100, self); self.progress_dialog.setWindowTitle("批量运算"); self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal); self.progress_dialog.setAutoClose(True); self.progress_dialog.setAutoReset(True); self.progress_dialog.canceled.connect(self._cancel_processing)
        QTimer.singleShot(50, self.progress_dialog.show)

        if DEBUG_MODE: print(f"DEBUG: Starting BatchLogicOpThread. Op={operation}, k={step_k}, loop={loop}")
        self.batch_logic_op_thread = BatchLogicOpThread(self.density_matrices, operation, step_k, loop, diff_colormap)
        self.batch_logic_op_thread.progress.connect(self._update_progress); self.batch_logic_op_thread.result_ready.connect(self._collect_batch_logic_result); self.batch_logic_op_thread.finished.connect(self._batch_logic_op_finished)
        self.batch_logic_op_thread.start()

    # --- Data Collection & Finish Callbacks ---
    # ... (_update_progress, _cancel_processing, _collect_slice_data, _add_thumbnail_item) ...
    def _collect_density_data(self, index, density_matrix, heatmap_pixmap, density_params):
        # ... (Implementation remains the same) ...
        if DEBUG_MODE:
            print(f"DEBUG: _collect_density_data for index {index}. Matrix shape: {density_matrix.shape}, Pixmap valid: {not heatmap_pixmap.isNull()}")
        self.density_matrices[index] = density_matrix
        self.density_pixmaps[index] = heatmap_pixmap
        self.density_params[index] = density_params

    def _collect_batch_logic_result(self, results_matrices, results_pixmaps):
        """Collect batch logic operation results."""
        if DEBUG_MODE: print(f"DEBUG: _collect_batch_logic_result received {len(results_matrices)} results.")
        self.batch_op_results_matrices = results_matrices
        self.batch_op_results_pixmaps = results_pixmaps
        self.batch_op_keys = sorted(results_pixmaps.keys()) # Store sorted keys for slider

    # --- Processing Finished Callbacks ---
    def _slice_processing_finished(self, success):
        # ... (Implementation from previous response, enables density button) ...
        if DEBUG_MODE: print(f"DEBUG: _slice_processing_finished called. Success: {success}")
        if self.progress_dialog:
            try: self.progress_dialog.setValue(100)
            except RuntimeError: self.progress_dialog = None
        self.slice_processing_thread = None
        if self.plotter:
            try: self.plotter.remove_actor("status_text", render=False);
            except Exception as e: print(f"WARNING: Could not remove status text: {e}")
        if success:
            print(f"INFO: Successfully processed {len(self.current_slices)} slices.")
            self._update_3d_view_presentation()
            self.update_density_btn.setEnabled(True) # Enable density calculation now
            # Enable density display combo ONLY if density data already existed (unlikely here)
            # self.density_display_combo.setEnabled(bool(self.density_pixmaps))
        else:
             was_canceled = False; # ... check progress_dialog ...
             if was_canceled: QMessageBox.information(self, "已取消", "切片处理已取消.")
             else: QMessageBox.warning(self, "处理失败", "切片处理失败.")
             if self.plotter: self.plotter.clear(); self.plotter.add_text("处理失败或取消",...); self.plotter.render()
        if self.progress_dialog:
            try: self.progress_dialog.close();
            except RuntimeError: pass; self.progress_dialog = None
        self._update_batch_op_ui() # Update batch button state based on slice availability

    def _density_processing_finished(self, success):
        # ... (Implementation from previous response, resets display mode) ...
        if DEBUG_MODE: print(f"DEBUG: _density_processing_finished called. Success: {success}")
        if self.progress_dialog:
            try: self.progress_dialog.setValue(100);
            except RuntimeError: self.progress_dialog = None
        self.density_processing_thread = None
        if success:
            print(f"INFO: Successfully processed densities for {len(self.density_matrices)} slices.")
            self.density_display_combo.setEnabled(True)
            first_available_index = sorted(self.density_pixmaps.keys())[0] if self.density_pixmaps else -1
            if first_available_index != -1:
                 self.current_display_mode = "DensitySlice"; self.density_display_combo.setCurrentText("选中切片")
                 self.current_display_slice_index = first_available_index
                 if DEBUG_MODE: print(f"DEBUG: _density_processing_finished - Setting view to Slice {first_available_index}")
            else:
                 self.current_display_mode = ""; self.density_view_label.setText("无有效密度图生成")
            self._update_center_view(); self._update_logic_op_ui(); self._update_batch_op_ui()
        else:
            was_canceled = False; # ... check progress_dialog ...
            if was_canceled: QMessageBox.information(self, "已取消", "密度计算已取消。")
            else: QMessageBox.warning(self, "处理失败", "密度计算过程中发生错误。")
            self.density_view_label.setText("密度计算失败或取消")
        if self.progress_dialog:
            try: self.progress_dialog.close();
            except RuntimeError: pass; self.progress_dialog = None

    def _batch_logic_op_finished(self, success):
        """Called when batch logic operation finishes."""
        if DEBUG_MODE: print(f"DEBUG: _batch_logic_op_finished called. Success: {success}")
        if self.progress_dialog:
            try: self.progress_dialog.setValue(100);
            except RuntimeError: self.progress_dialog = None
        self.batch_logic_op_thread = None

        if success and self.batch_op_results_pixmaps:
            print(f"INFO: Successfully finished batch logic operation with {len(self.batch_op_results_pixmaps)} results.")
            self._update_batch_op_ui() # Enable slider/play
            # Switch display to show the first batch result
            self.density_display_combo.setCurrentText("批量运算结果") # This triggers _update_display_mode
            self.current_batch_result_index = 0
            self.batch_result_slider.setValue(0)
            # _update_center_view will be called by the combo change signal
        else:
            if self.progress_dialog and self.progress_dialog.wasCanceled(): QMessageBox.information(self, "已取消", "批量运算已取消。")
            else: QMessageBox.warning(self, "处理失败", "批量逻辑运算失败或未产生结果。")
            self.batch_op_results_matrices.clear(); self.batch_op_results_pixmaps.clear(); self.batch_op_keys.clear()
            self._update_batch_op_ui() # Ensure UI is disabled/reset

        if self.progress_dialog:
            try: self.progress_dialog.close();
            except RuntimeError: pass;
            self.progress_dialog = None

    # --- UI Update Helpers ---
    # ... (_update_3d_view_presentation - Needs careful check to ensure it uses self.current_slices correctly) ...
    # _update_3d_view_presentation (ensure it only uses self.current_slices, no density logic needed here)
    def _update_3d_view_presentation(self):
        if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation called.")
        if self.plotter is None: return
        # --- Ensure we are showing the 3D view stack page ---
        if self.center_stacked_widget.currentIndex() != 0:
             # Don't update if 3D view is not visible
             # Or alternatively, switch to it: self.center_stacked_widget.setCurrentIndex(0)
             if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - 3D view not active, skipping update.")
             return
        try: self.plotter.clear_actors()
        except Exception as e: print(f"ERROR: Failed clear actors: {e}")
        if not self.current_slices: self.plotter.add_text("无切片", name="status_text"); self.plotter.render(); return
        offset = self.offset_spin.value(); ps = self.point_size_spin.value(); use_color = self.use_color_check.isChecked()
        actors = []; current_offset = 0.0; sorted_indices = sorted(self.current_slices.keys()); all_bounds = []
        for i in sorted_indices:
            slice_data = self.current_slices.get(i)
            if slice_data is None or slice_data.n_points == 0: current_offset += offset; continue
            offset_slice = slice_data.copy(deep=True); offset_slice.points[:, 2] += current_offset
            if offset_slice.bounds[0] < offset_slice.bounds[1]: all_bounds.extend(offset_slice.bounds)
            try:
                if 'colors' in offset_slice.point_data and use_color: actor = self.plotter.add_mesh(offset_slice, scalars='colors', rgb=True, point_size=ps)
                else: actor = self.plotter.add_mesh(offset_slice, color='grey', point_size=ps)
                if actor: actors.append(actor)
            except Exception as e: print(f"ERROR: Adding mesh {i}: {e}")
            current_offset += offset
        if actors:
            try:
                if all_bounds:
                     bounds = [min(all_bounds[0::6]),max(all_bounds[1::6]),min(all_bounds[2::6]),max(all_bounds[3::6]),min(all_bounds[4::6]),max(all_bounds[5::6])]
                     self.plotter.reset_camera(bounds=bounds)
                else: self.plotter.reset_camera()
                # Keep current view unless explicitly reset
                # self.plotter.view_vector([1, -1, 0.5], viewup=[0, 0, 1])
            except Exception as e: print(f"ERROR: Resetting camera: {e}")
        elif self.current_slices: self.plotter.add_text("所有切片均为空", name="status_text")
        try: self.plotter.render()
        except Exception as e: print(f"ERROR: Rendering presentation: {e}")

    def _update_density_display_label(self):
        pixmap_to_show = QPixmap(); label_text = ""
        if DEBUG_MODE: print(f"DEBUG: _update_density_display_label - Mode: '{self.current_display_mode}'")
        if self.current_display_mode == "DensitySlice":
            target_index = self.current_display_slice_index # Use state variable
            if target_index in self.density_pixmaps:
                pixmap_to_show = self.density_pixmaps.get(target_index, QPixmap())
                label_text = f"切片 {target_index} 密度图"
            else: label_text = f"切片密度图不可用 (索引: {target_index})"
        elif self.current_display_mode == "DensitySingleResult":
            if self.logic_op_result_pixmap and not self.logic_op_result_pixmap.isNull(): pixmap_to_show = self.logic_op_result_pixmap; label_text = "单次逻辑运算结果"
            else: label_text = "无单次逻辑运算结果"
        elif self.current_display_mode == "DensityBatchResult":
            if self.batch_op_keys and 0 <= self.current_batch_result_index < len(self.batch_op_keys):
                current_key = self.batch_op_keys[self.current_batch_result_index]
                pixmap_to_show = self.batch_op_results_pixmaps.get(current_key, QPixmap())
                label_text = f"批量运算结果: {current_key}"
                self.batch_result_label.setText(f"结果: {self.current_batch_result_index + 1}/{len(self.batch_op_keys)}")
            else: label_text = "无批量运算结果"
        else: label_text = "请先计算密度图"
        if not pixmap_to_show.isNull():
            scaled_pixmap = pixmap_to_show.scaled(self.density_view_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.density_view_label.setPixmap(scaled_pixmap); self.density_view_label.setText("")
        else: self.density_view_label.setText(label_text); self.density_view_label.setPixmap(QPixmap())


    # --- Interaction Slots ---
    # ... (resizeEvent) ...
    def resizeEvent(self, event):
        """Handle window resize to update scaled pixmap."""
        super().resizeEvent(event)
        if self.center_stacked_widget.currentIndex() == 1: self._update_density_display_label()


    def _on_selection_changed(self):
        # --- Modified: Only update density slice view ---
        selected_items = self.slice_list_widget.selectedItems()
        if selected_items and self.current_display_mode == "DensitySlice":
            first_selected_index = selected_items[0].data(Qt.ItemDataRole.UserRole)
            if first_selected_index in self.density_pixmaps: # Check if density map exists
                 self.current_display_slice_index = first_selected_index
                 self._update_density_display_label() # Update the label directly
                 if DEBUG_MODE: print(f"DEBUG: List selection changed, density view updated to index {first_selected_index}")

    def _show_list_context_menu(self, pos: QPoint):
        # ... (Implementation remains the same) ...
        item = self.slice_list_widget.itemAt(pos);
        if not item: return
        index = item.data(Qt.ItemDataRole.UserRole); menu = QMenu()
        action_a = QAction(f"设为逻辑运算切片 A (当前: {self.selected_slice_a if self.selected_slice_a is not None else '无'})", self)
        action_b = QAction(f"设为逻辑运算切片 B (当前: {self.selected_slice_b if self.selected_slice_b is not None else '无'})", self)

        action_a.triggered.connect(lambda: self._set_logic_operand('A', index))
        action_b.triggered.connect(lambda: self._set_logic_operand('B', index))
        enabled = index in self.density_matrices
        action_a.setEnabled(enabled); action_b.setEnabled(enabled)
        if not enabled: action_a.setText(f"设为切片 A (密度未计算)"); action_b.setText(f"设为切片 B (密度未计算)")
        menu.addAction(action_a)
        menu.addAction(action_b)
        menu.exec(self.slice_list_widget.mapToGlobal(pos))

    def _set_logic_operand(self, operand, index):
        # ... (Implementation remains the same) ...
        if operand == 'A': self.selected_slice_a = index
        elif operand == 'B': self.selected_slice_b = index
        self._update_logic_op_ui()

    def _clear_logic_operand(self, operand):
        """Clear selection for Slice A or B."""
        if operand == 'A': self.selected_slice_a = None
        elif operand == 'B': self.selected_slice_b = None
        self._update_logic_op_ui()

    def _update_logic_op_ui(self):
        # ... (Implementation remains the same, updates labels and button state) ...
        a_ok = self.selected_slice_a is not None and self.selected_slice_a in self.density_matrices
        b_ok = self.selected_slice_b is not None and self.selected_slice_b in self.density_matrices
        self.slice_a_label.setText(f"{self.selected_slice_a if a_ok else '未选'}{'' if a_ok else ' (无数据)' if self.selected_slice_a is not None else ''}")
        self.slice_b_label.setText(f"{self.selected_slice_b if b_ok else '未选'}{'' if b_ok else ' (无数据)' if self.selected_slice_b is not None else ''}")
        self.compute_logic_op_btn.setEnabled(a_ok and b_ok)

    def _update_batch_op_ui(self):
        """Enable/disable batch operation UI elements based on data."""
        densities_exist = bool(self.density_matrices)
        results_exist = bool(self.batch_op_keys)
        self.execute_batch_op_btn.setEnabled(densities_exist)
        self.batch_result_slider.setEnabled(results_exist)
        self.batch_play_btn.setEnabled(results_exist)
        if results_exist:
            self.batch_result_slider.setRange(0, len(self.batch_op_keys) - 1)
            self.batch_result_slider.setValue(self.current_batch_result_index)
            self.batch_result_label.setText(f"结果: {self.current_batch_result_index + 1}/{len(self.batch_op_keys)}")
        else:
            self.batch_result_slider.setRange(0, 0)
            self.batch_result_label.setText("结果: 0/0")
            self.batch_play_btn.setChecked(False) # Ensure play button is reset


    def _compute_logic_operation(self):
        # ... (Implementation remains the same, sets display mode) ...
        if not self.compute_logic_op_btn.isEnabled(): return
        idx_a = self.selected_slice_a; idx_b = self.selected_slice_b; operation = self.logic_op_combo.currentText(); matrix_a = self.density_matrices.get(idx_a); matrix_b = self.density_matrices.get(idx_b)
        if matrix_a is None or matrix_b is None or matrix_a.shape != matrix_b.shape: QMessageBox.critical(self, "错误", "密度矩阵无效或不匹配。"); return
        result_matrix = None; colormap = self.diff_colormap_combo.currentText() if '差分' in operation else 'binary'; vmin, vmax = 0, 1
        try:
            if operation == "差分 (A-B)": result_matrix = matrix_a - matrix_b
            elif operation == "差分 (B-A)": result_matrix = matrix_b - matrix_a
            elif operation == "并集 (A | B)": result_matrix = np.logical_or(matrix_a > 0, matrix_b > 0).astype(float)
            elif operation == "交集 (A & B)": result_matrix = np.logical_and(matrix_a > 0, matrix_b > 0).astype(float)
            if result_matrix is not None:
                if '差分' in operation: abs_max = np.max(np.abs(result_matrix)) if np.any(result_matrix) else 1; vmin, vmax = -abs_max, abs_max
                self.logic_op_result_matrix = result_matrix; self.logic_op_result_pixmap = create_density_heatmap(result_matrix, colormap, vmin=vmin, vmax=vmax)
                self.density_display_combo.setCurrentText("单次运算结果") # Trigger update
                self._update_center_view()
            else: QMessageBox.warning(self, "未实现", f"操作 '{operation}' 未实现。")
        except Exception as e: print(f"ERROR: Logic op error: {e}"); QMessageBox.critical(self, "计算错误", f"逻辑运算错误:\n{e}"); self.logic_op_result_matrix = None; self.logic_op_result_pixmap = None


    def _update_batch_preview_slider(self, value):
        """Update display when batch slider changes."""
        if self.current_display_mode == "DensityBatchResult" and self.batch_op_keys:
            self.current_batch_result_index = value
            self._update_density_display_label() # Refresh center view

    def _toggle_batch_play(self, checked):
        """Start/stop batch result animation."""
        if checked and self.batch_op_keys:
            interval_ms = 500 # Adjust speed as needed
            self.batch_play_timer.start(interval_ms)
            self.batch_play_btn.setText("暂停")
        else:
            self.batch_play_timer.stop()
            self.batch_play_btn.setText("播放")
            self.batch_play_btn.setChecked(False) # Ensure button state is consistent

    def _advance_batch_slider(self):
        """Increment slider for batch playback."""
        if not self.batch_op_keys:
            self._toggle_batch_play(False) # Stop if no results
            return
        max_index = len(self.batch_op_keys) - 1
        next_index = (self.current_batch_result_index + 1) % (max_index + 1) # Loop
        self.batch_result_slider.setValue(next_index) # This triggers _update_batch_preview_slider

    # --- Export ---
    def _export_data(self, indices_to_export): # Renamed from _export_bitmaps
        # ... (Implementation includes PCD and density export, uses overall_xy_bounds) ...
        if DEBUG_MODE: print(f"DEBUG: _export_data called for indices: {indices_to_export}")
        if not indices_to_export: return
        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not export_dir: return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); base_export_path = os.path.join(export_dir, f"batch_slice_export_{timestamp}"); os.makedirs(base_export_path, exist_ok=True)
        global_params = {"export_time": datetime.datetime.now().isoformat(),"original_point_cloud_source": self.source_filename,"num_slices_param": self.num_slices_spin.value(),"slice_thickness_param": self.thickness_spin.value(),}
        global_params_file = os.path.join(base_export_path, "export_parameters.json")
        with open(global_params_file, 'w', encoding='utf-8') as f: json.dump(global_params, f, ensure_ascii=False, indent=2)
        overall_xy_bounds = get_overall_xy_bounds(self.current_slices)
        export_progress = QProgressDialog("正在导出数据...", "取消", 0, len(indices_to_export), self); export_progress.setWindowTitle("导出进度"); export_progress.setWindowModality(Qt.WindowModality.WindowModal); export_progress.setAutoClose(True); export_progress.setAutoReset(True); export_progress.show()
        exported_count = 0; exported_pcd_count = 0; exported_density_count = 0
        for i, index in enumerate(indices_to_export):
             export_progress.setValue(i)
             if export_progress.wasCanceled(): print("INFO: Export canceled."); break
             slice_data_pv = self.current_slices.get(index); metadata = self.slice_metadata.get(index); density_matrix = self.density_matrices.get(index)
             if metadata is None: print(f"WARNING: Metadata missing for {index}. Skipping."); continue
             img_np = None; view_params_render = None; render_error_msg = None; bitmap_saved = False; pcd_saved = False; density_saved = False
             # Render Bitmap
             if not metadata.get("is_empty", False) and slice_data_pv is not None:
                 export_progress.setLabelText(f"渲染位图 {index}..."); QApplication.processEvents()
                 try: img_np, view_params_render = render_slice_to_image(slice_data_pv, self.BITMAP_EXPORT_RESOLUTION, overall_xy_bounds, False);
                 except Exception as render_err: render_error_msg = f"Bitmap render error: {render_err}"
             # Save Metadata
             meta_filename = os.path.join(base_export_path, f"slice_{index}_metadata.json"); metadata["view_params_render"] = view_params_render if img_np is not None else None
             if render_error_msg: metadata["render_error"] = render_error_msg
             if index in self.density_params: metadata["density_params"] = self.density_params[index] # Add density params if available
             export_data = {"slice_index": index, "metadata": metadata}
             try:
                 with open(meta_filename, 'w', encoding='utf-8') as f: json.dump(export_data, f, ensure_ascii=False, indent=2)
             except Exception as meta_save_err: print(f"ERROR: Saving metadata {index}: {meta_save_err}")
             # Save Bitmap
             if img_np is not None:
                 try:
                     bitmap_filename = os.path.join(base_export_path, f"slice_{index}_bitmap.png"); img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                     if cv2.imwrite(bitmap_filename, img_bgr): bitmap_saved = True
                 except Exception as export_err: print(f"ERROR: Saving bitmap {index}: {export_err}")
             # Save PCD
             if slice_data_pv is not None and slice_data_pv.n_points > 0:
                 pcd_filename = os.path.join(base_export_path, f"slice_{index}.pcd")
                 try:
                     export_progress.setLabelText(f"保存 PCD {index}..."); QApplication.processEvents()
                     points = slice_data_pv.points; o3d_pcd = o3d.geometry.PointCloud(); o3d_pcd.points = o3d.utility.Vector3dVector(points)
                     if 'colors' in slice_data_pv.point_data:
                         colors = slice_data_pv['colors']; o3d_pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
                     if o3d.io.write_point_cloud(pcd_filename, o3d_pcd, write_ascii=False, compressed=True):
                         pcd_saved = True; exported_pcd_count += 1
                 except Exception as pcd_err: print(f"ERROR: Saving PCD {index}: {pcd_err}")
             elif metadata.get("is_empty", False): pcd_saved = True
             # Save Density Data
             if density_matrix is not None:
                 try:
                     export_progress.setLabelText(f"保存密度数据 {index}..."); QApplication.processEvents()
                     matrix_filename = os.path.join(base_export_path, f"slice_{index}_density_matrix.npy"); np.save(matrix_filename, density_matrix)
                     heatmap_filename = os.path.join(base_export_path, f"slice_{index}_density_heatmap.png")
                     density_pixmap = self.density_pixmaps.get(index) # Get the already generated pixmap
                     if density_pixmap and not density_pixmap.isNull():
                         if density_pixmap.save(heatmap_filename, "PNG"): density_saved = True; exported_density_count += 1
                         else: print(f"ERROR: Saving density heatmap PNG for {index} failed.")
                     else: print(f"WARNING: Density heatmap pixmap not available for {index}.")
                 except Exception as density_err: print(f"ERROR: Saving density data {index}: {density_err}")
             elif index in self.density_matrices: density_saved = True

             if bitmap_saved or pcd_saved or density_saved or metadata.get("is_empty", False): exported_count += 1

        export_progress.setValue(len(indices_to_export))
        QMessageBox.information(self, "导出完成", f"处理完成 {len(indices_to_export)} 项。\n成功导出 {exported_count} 个切片的数据。\n(PCD: {exported_pcd_count}, 密度: {exported_density_count})\n保存在:\n{base_export_path}")

    def _export_selected_data(self): # Renamed from _export_selected_bitmaps
        # ... (Implementation remains the same) ...
        selected_items = self.slice_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "未选择", "请先选择要导出的项。"); return
        indices = sorted([item.data(Qt.ItemDataRole.UserRole) for item in selected_items])
        self._export_data(indices)


    def _export_all_data(self):
        # ... (Implementation remains the same) ...
        if not self.slice_metadata: QMessageBox.warning(self, "无数据", "请先生成切片。"); return
        indices = sorted(list(self.slice_metadata.keys())); self._export_data(indices)


    # --- Close Event ---
    # ... (Implementation remains the same) ...
    def closeEvent(self, event):
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow closeEvent called.")
        self._cancel_processing()
        threads = [self.slice_processing_thread, self.density_processing_thread, self.batch_logic_op_thread]
        running_threads = [t for t in threads if t and t.isRunning()]
        if running_threads:
            print(f"INFO: Waiting for {len(running_threads)} processing thread(s) to finish before closing...")
            for t in running_threads: t.wait(1500) # Wait a bit for each
        if self.plotter:
             if DEBUG_MODE: print("DEBUG: Closing plotter in closeEvent.")
             try: self.plotter.close()
             except Exception as e: print(f"ERROR: Closing plotter: {e}")
        super().closeEvent(event)