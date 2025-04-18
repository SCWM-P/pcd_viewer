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
                             QComboBox, QStackedWidget, QMenu, QFormLayout) # Added QTabWidget, QComboBox, QStackedWidget, QMenu
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import QPixmap, QImage, QIcon, QPainter, QColor, QAction # Added QColor, QAction
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

    # Add a small padding
    x_range = xmax - xmin
    y_range = ymax - ymin
    padding = max(x_range * 0.05, y_range * 0.05, 0.1) # Ensure minimum padding

    return [xmin - padding, xmax + padding, ymin - padding, ymax + padding]

def render_slice_to_image(slice_data, size, overall_xy_bounds=None, is_thumbnail=True):
    """Renders a single slice to a NumPy image array using an off-screen plotter."""
    # ... (Implementation from previous response, ensure it's correct) ...
    if DEBUG_MODE: print(f"DEBUG: render_slice_to_image called. is_thumbnail={is_thumbnail}, size={size}")
    if slice_data is None or slice_data.n_points == 0:
        if DEBUG_MODE: print("DEBUG: render_slice_to_image - Empty slice data.")
        return None, {}
    plotter = None
    try:
        img_width, img_height = size if isinstance(size, tuple) else (size.width(), size.height())
        if DEBUG_MODE: print(f"DEBUG: render_slice_to_image - Creating off-screen plotter with size {img_width}x{img_height}")
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
            bounds_to_use = overall_xy_bounds + [zmin, zmax]
            if DEBUG_MODE: print(f"DEBUG: render_slice_to_image - Resetting camera using OVERALL XY bounds: {bounds_to_use}")
        elif slice_data and slice_data.bounds[0] < slice_data.bounds[1]:
            bounds_to_use = slice_data.bounds
            if DEBUG_MODE: print(f"DEBUG: render_slice_to_image - Resetting camera using SLICE bounds: {bounds_to_use}")
        if bounds_to_use:
            plotter.reset_camera(bounds=bounds_to_use)
            if DEBUG_MODE: print("DEBUG: render_slice_to_image - Camera reset done.")
        else:
             if DEBUG_MODE: print("DEBUG: render_slice_to_image - No valid bounds for camera reset.")

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
    if density_matrix is None or density_matrix.size == 0:
        return QPixmap() # Return empty pixmap

    try:
        # Normalize the density matrix
        if vmin is None: vmin = np.min(density_matrix)
        if vmax is None: vmax = np.max(density_matrix)
        # Avoid division by zero if vmax == vmin
        if vmax <= vmin: vmax = vmin + 1e-6
        normalized_matrix = (density_matrix - vmin) / (vmax - vmin)
        normalized_matrix = np.clip(normalized_matrix, 0, 1) # Ensure values are within [0, 1]

        # Apply colormap
        cmap = plt.get_cmap(colormap_name)
        colored_matrix_rgba = cmap(normalized_matrix, bytes=True) # Get RGBA uint8 array

        # Convert RGBA to QImage
        height, width, _ = colored_matrix_rgba.shape
        q_img = QImage(colored_matrix_rgba.data, width, height, width * 4, QImage.Format.Format_RGBA8888)

        if q_img.isNull():
             print("ERROR: create_density_heatmap - QImage creation failed.")
             return QPixmap()

        return QPixmap.fromImage(q_img)

    except Exception as e:
        print(f"ERROR: Failed to create density heatmap: {e}")
        if DEBUG_MODE: traceback.print_exc()
        return QPixmap()

# --- Background Threads ---

class SliceProcessingThread(QThread):
    # ... (Implementation from previous response, ensure it's correct) ...
    progress = pyqtSignal(int, str)
    slice_ready = pyqtSignal(int, object, tuple)
    thumbnail_ready = pyqtSignal(int, QPixmap, dict)
    finished = pyqtSignal(bool)
    def __init__(self, point_cloud, num_slices, thickness, limit_thickness, thumbnail_size, parent=None):
        super().__init__(parent)
        self.point_cloud = point_cloud
        self.num_slices = num_slices
        self.thickness_param = thickness
        self.limit_thickness = limit_thickness
        self.thickness = thickness
        self.thumbnail_size = thumbnail_size
        self._is_running = True
        if DEBUG_MODE: print(f"DEBUG: SliceProcessingThread initialized. num_slices={num_slices}, thickness={thickness}")
    def run(self):
        if DEBUG_MODE: print("DEBUG: SliceProcessingThread run started.")
        if self.point_cloud is None or self.num_slices <= 0 or self.thickness_param <= 0:
            if DEBUG_MODE: print("DEBUG: SliceProcessingThread run - Invalid parameters or point cloud.")
            self.finished.emit(False); return
        try:
            bounds = self.point_cloud.bounds; min_z, max_z = bounds[4], bounds[5]; total_height = max_z - min_z
            if DEBUG_MODE: print(f"DEBUG: SliceProcessingThread - Point cloud bounds Z: {min_z} to {max_z}, total_height={total_height}")
            if total_height <= 0: self.finished.emit(False); return
            all_points = self.point_cloud.points; has_colors = 'colors' in self.point_cloud.point_data
            if has_colors: all_colors = self.point_cloud['colors']
            step = total_height / self.num_slices; current_start_z = min_z; actual_thickness = self.thickness_param
            if self.limit_thickness:
                max_allowed_thickness = step
                if actual_thickness > max_allowed_thickness:
                    if DEBUG_MODE: print(f"DEBUG: Requested thickness {self.thickness_param} exceeds step {max_allowed_thickness}. Limiting thickness.")
                    actual_thickness = max_allowed_thickness
                else:
                    if DEBUG_MODE: print(f"DEBUG: Requested thickness {self.thickness_param} is within limit.")
            else:
                 if DEBUG_MODE: print("DEBUG: Thickness limit checkbox is off.")
            total_steps = self.num_slices * 2
            if DEBUG_MODE: print(f"DEBUG: SliceProcessingThread - Calculated step={step}, actual_thickness={actual_thickness}")
            generated_slices = []; height_ranges = []
            if DEBUG_MODE: print("DEBUG: SliceProcessingThread - Starting slicing phase...")
            for i in range(self.num_slices):
                if not self._is_running: raise InterruptedError("Processing stopped by user request.")
                self.progress.emit(int((i + 1) / total_steps * 100), f"正在生成切片 {i+1}/{self.num_slices}")
                slice_start_z = current_start_z; slice_end_z = slice_start_z + actual_thickness
                slice_end_z = min(slice_end_z, max_z + 1e-6); slice_start_z = min(slice_start_z, slice_end_z)
                indices = np.where((all_points[:, 2] >= slice_start_z) & (all_points[:, 2] <= slice_end_z))[0]
                height_ranges.append((slice_start_z, slice_end_z))
                if len(indices) > 0:
                    slice_points = all_points[indices]; slice_cloud = pv.PolyData(slice_points)
                    if has_colors: slice_cloud['colors'] = all_colors[indices]
                    generated_slices.append(slice_cloud); self.slice_ready.emit(i, slice_cloud, (slice_start_z, slice_end_z))
                else:
                    generated_slices.append(None); self.slice_ready.emit(i, None, (slice_start_z, slice_end_z))
                current_start_z += step
            temp_slices_dict = {i: s for i, s in enumerate(generated_slices)}
            overall_xy_bounds = get_overall_xy_bounds(temp_slices_dict)
            if DEBUG_MODE: print(f"DEBUG: SliceProcessingThread - Calculated overall XY bounds: {overall_xy_bounds}")
            if DEBUG_MODE: print("DEBUG: SliceProcessingThread - Starting thumbnail generation phase...")
            for i in range(self.num_slices):
                if not self._is_running: raise InterruptedError("Processing stopped by user request.")
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
                         print(f"ERROR: SliceProcessingThread - Failed to create QImage/QPixmap for thumbnail {i}: {qimage_err}")
                         if DEBUG_MODE: traceback.print_exc()
                         placeholder_pixmap = QPixmap(self.thumbnail_size); placeholder_pixmap.fill(Qt.GlobalColor.darkRed)
                         self.thumbnail_ready.emit(i, placeholder_pixmap, metadata)
                else:
                    placeholder_pixmap = QPixmap(self.thumbnail_size); placeholder_pixmap.fill(Qt.GlobalColor.lightGray)
                    painter = QPainter(placeholder_pixmap); painter.drawText(placeholder_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, f"Slice {i}\n(Empty)"); painter.end()
                    self.thumbnail_ready.emit(i, placeholder_pixmap, metadata)
            if DEBUG_MODE: print("DEBUG: SliceProcessingThread - Processing finished successfully.")
            self.finished.emit(True)
        except InterruptedError: print("INFO: Slice processing thread stopped by user."); self.finished.emit(False)
        except Exception as e: print(f"ERROR: Unhandled error during slice processing thread: {e}"); self.finished.emit(False)
    def stop(self): self._is_running = False

class DensityProcessingThread(QThread):
    progress = pyqtSignal(int, str) # percentage, status
    density_map_ready = pyqtSignal(int, np.ndarray, QPixmap, dict) # index, density_matrix, heatmap_pixmap, density_params
    finished = pyqtSignal(bool) # success

    def __init__(self, slices_dict, overall_xy_bounds, grid_resolution, colormap_name, parent=None):
        super().__init__(parent)
        self.slices_dict = slices_dict
        self.overall_xy_bounds = overall_xy_bounds
        self.grid_resolution = grid_resolution
        self.colormap_name = colormap_name
        self._is_running = True
        if DEBUG_MODE: print(f"DEBUG: DensityProcessingThread initialized. Resolution={grid_resolution}, Colormap={colormap_name}")

    def run(self):
        if DEBUG_MODE: print("DEBUG: DensityProcessingThread run started.")
        if not self.slices_dict or self.overall_xy_bounds is None:
            print("ERROR: DensityProcessingThread - No slices or overall bounds provided.")
            self.finished.emit(False); return

        xmin, xmax, ymin, ymax = self.overall_xy_bounds
        bins = [self.grid_resolution, self.grid_resolution]
        range_xy = [[xmin, xmax], [ymin, ymax]]

        num_slices = len(self.slices_dict)
        sorted_indices = sorted(self.slices_dict.keys())

        try:
            max_density = 0 # Track max density across all slices for consistent colormap scaling
            all_matrices = {}

            # First pass: Calculate all density matrices and find max density
            if DEBUG_MODE: print("DEBUG: DensityProcessingThread - First pass: Calculating densities...")
            for i, index in enumerate(sorted_indices):
                if not self._is_running: raise InterruptedError("Processing stopped")
                self.progress.emit(int(((i + 1) / (num_slices * 2)) * 100), f"计算密度 {index+1}/{num_slices}")
                slice_data = self.slices_dict.get(index)
                if slice_data is not None and slice_data.n_points > 0:
                    points_xy = slice_data.points[:, 0:2]
                    density_matrix, _, _ = np.histogram2d(
                        points_xy[:, 0], points_xy[:, 1], bins=bins, range=range_xy
                    )
                    # Rotate and flip for correct image orientation if necessary
                    # density_matrix = np.rot90(density_matrix)
                    # density_matrix = np.flipud(density_matrix)
                    all_matrices[index] = density_matrix
                    current_max = np.max(density_matrix)
                    if current_max > max_density: max_density = current_max
                else:
                    all_matrices[index] = np.zeros(bins) # Empty matrix for empty slices

            if DEBUG_MODE: print(f"DEBUG: DensityProcessingThread - Max density found: {max_density}")

            # Second pass: Generate heatmaps with consistent scaling
            if DEBUG_MODE: print("DEBUG: DensityProcessingThread - Second pass: Generating heatmaps...")
            for i, index in enumerate(sorted_indices):
                 if not self._is_running: raise InterruptedError("Processing stopped")
                 self.progress.emit(int(((num_slices + i + 1) / (num_slices * 2)) * 100), f"生成热力图 {index+1}/{num_slices}")
                 density_matrix = all_matrices[index]
                 heatmap_pixmap = create_density_heatmap(density_matrix, self.colormap_name, vmin=0, vmax=max_density)

                 density_params = {
                     "grid_resolution": self.grid_resolution,
                     "colormap": self.colormap_name,
                     "xy_bounds": self.overall_xy_bounds,
                     "max_density_scale": max_density
                 }
                 self.density_map_ready.emit(index, density_matrix, heatmap_pixmap, density_params)
                 if DEBUG_MODE: print(f"DEBUG: DensityProcessingThread - Heatmap ready for index {index}")


            if DEBUG_MODE: print("DEBUG: DensityProcessingThread finished successfully.")
            self.finished.emit(True)

        except InterruptedError:
             print("INFO: Density processing thread stopped by user.")
             self.finished.emit(False)
        except Exception as e:
            print(f"ERROR: Unhandled error during density processing thread: {e}")
            if DEBUG_MODE: traceback.print_exc()
            self.finished.emit(False)

    def stop(self):
        if DEBUG_MODE: print("DEBUG: DensityProcessingThread stop requested.")
        self._is_running = False


# --- Main Window Class ---
class BatchSliceViewerWindow(QWidget):
    BITMAP_EXPORT_RESOLUTION = (1024, 1024)
    DEFAULT_DENSITY_RESOLUTION = 512
    AVAILABLE_COLORMAPS = plt.colormaps() # Get available matplotlib colormaps

    def __init__(self, point_cloud, source_filename="Unknown", parent=None):
        super().__init__(parent)
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow __init__ started.")
        self.setWindowTitle("批量切片查看器")
        self.setMinimumSize(1100, 750) # Slightly larger default size
        self.setWindowFlags(Qt.WindowType.Window)
        self.setStyleSheet(StylesheetManager.get_light_theme())

        self.original_point_cloud = point_cloud
        self.source_filename = source_filename
        # Data storage
        self.current_slices = {} # {index: pv.PolyData or None}
        self.slice_metadata = {} # {index: metadata_dict} - includes thumb view params, height, etc.
        self.density_matrices = {} # {index: np.ndarray}
        self.density_pixmaps = {} # {index: QPixmap}
        self.density_params = {} # {index: params_dict} - resolution, cmap, bounds, scale used
        self.logic_op_result_matrix = None
        self.logic_op_result_pixmap = None
        # UI State
        self.selected_slice_a = None
        self.selected_slice_b = None
        self.current_density_display_mode = "Slice" # "Slice" or "Result"
        self.current_density_display_index = 0 # Index of slice to show in density mode
        # Threads
        self.slice_processing_thread = None
        self.density_processing_thread = None
        self.progress_dialog = None
        self.plotter = None

        self.setup_ui()
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow __init__ finished.")

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

        # --- Splitter Setup ---
        self.splitter.setSizes([280, 600, 280]) # Adjusted sizes slightly
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow setup_ui finished.")

    def setup_left_panel(self):
        """Sets up the left panel with the thumbnail list."""
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        list_group = QGroupBox("切片预览 (顶视图)")
        list_group_layout = QVBoxLayout(list_group)
        self.slice_list_widget = QListWidget()
        self.slice_list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.slice_list_widget.setIconSize(QSize(128, 128))
        self.slice_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.slice_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.slice_list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        # Add context menu for selecting Slice A/B
        self.slice_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.slice_list_widget.customContextMenuRequested.connect(self._show_list_context_menu)
        list_group_layout.addWidget(self.slice_list_widget)
        list_button_layout = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self.slice_list_widget.selectAll)
        deselect_all_btn = QPushButton("全不选")
        deselect_all_btn.clicked.connect(self.slice_list_widget.clearSelection)
        export_selected_bitmaps_btn = QPushButton("导出选中") # Combined export now
        export_selected_bitmaps_btn.clicked.connect(self._export_selected_data) # Connect to combined export
        list_button_layout.addWidget(select_all_btn)
        list_button_layout.addWidget(deselect_all_btn)
        list_button_layout.addStretch()
        list_button_layout.addWidget(export_selected_bitmaps_btn)
        list_group_layout.addLayout(list_button_layout)
        left_layout.addWidget(list_group)
        self.splitter.addWidget(left_panel)

    def setup_center_panel(self):
        """Sets up the center panel with the stacked widget for 3D/2D views."""
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        self.center_stacked_widget = QStackedWidget()
        center_layout.addWidget(self.center_stacked_widget)

        # --- Page 0: 3D Plotter ---
        self.plotter_widget = QWidget() # Container widget for plotter might be needed
        plotter_layout = QVBoxLayout(self.plotter_widget)
        plotter_layout.setContentsMargins(0,0,0,0)
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow setup_ui - Creating QtInteractor...")
        try:
            self.plotter = QtInteractor(parent=self.plotter_widget) # Parent is container
            if DEBUG_MODE: print(f"DEBUG: BatchSliceViewerWindow setup_ui - QtInteractor created: {self.plotter}")
            plotter_layout.addWidget(self.plotter)
            QTimer.singleShot(200, self._initialize_plotter_view)
        except Exception as e:
             print(f"ERROR: Failed to create QtInteractor: {e}")
             if DEBUG_MODE: traceback.print_exc()
             self.plotter = None # Ensure plotter is None
             # Add error label directly to the stack page if plotter fails
             error_label = QLabel(f"无法初始化3D视图。\n错误: {e}")
             error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
             error_label.setWordWrap(True)
             plotter_layout.addWidget(error_label) # Add to the container layout

        self.center_stacked_widget.addWidget(self.plotter_widget) # Add container page

        # --- Page 1: 2D Density View ---
        self.density_view_label = QLabel("请先计算密度图")
        self.density_view_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.density_view_label.setScaledContents(False) # Do not scale content, use pixmap scaling
        # Add scroll area for potentially large density maps? Maybe later.
        self.center_stacked_widget.addWidget(self.density_view_label)

        self.splitter.addWidget(center_panel)

    def setup_right_panel(self):
        """Sets up the right panel with the tab widget."""
        right_panel = QWidget()
        right_panel.setMinimumWidth(280) # Adjusted width
        right_panel.setMaximumWidth(400)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)

        self.right_tab_widget = QTabWidget()
        right_layout.addWidget(self.right_tab_widget)

        # --- Tab 1: View Controls ---
        view_control_tab = QWidget()
        vc_layout = QVBoxLayout(view_control_tab)
        self.setup_view_control_tab(vc_layout)
        self.right_tab_widget.addTab(view_control_tab, "视图控制")

        # --- Tab 2: Density Analysis ---
        density_tab = QWidget()
        density_layout = QVBoxLayout(density_tab)
        self.setup_density_analysis_tab(density_layout)
        self.right_tab_widget.addTab(density_tab, "密度分析")

        # Connect tab change signal to update center view
        self.right_tab_widget.currentChanged.connect(self._handle_tab_change)

        self.splitter.addWidget(right_panel)

    def setup_view_control_tab(self, layout):
        """Populates the 'View Control' tab."""
        # Slicing Parameters Group
        slicing_group = QGroupBox("切片参数")
        slicing_layout = QVBoxLayout(slicing_group)
        num_slices_layout = QHBoxLayout()
        num_slices_layout.addWidget(QLabel("切片数量:"))
        self.num_slices_spin = QSpinBox()
        self.num_slices_spin.setRange(1, 500); self.num_slices_spin.setValue(10)
        num_slices_layout.addWidget(self.num_slices_spin)
        slicing_layout.addLayout(num_slices_layout)
        thickness_layout = QHBoxLayout()
        thickness_layout.addWidget(QLabel("单片厚度 (米):"))
        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setRange(0.01, 5.0); self.thickness_spin.setSingleStep(0.01)
        self.thickness_spin.setValue(0.10); self.thickness_spin.setDecimals(3)
        thickness_layout.addWidget(self.thickness_spin)
        slicing_layout.addLayout(thickness_layout)
        self.limit_thickness_check = QCheckBox("无重叠")
        self.limit_thickness_check.setChecked(True)
        self.limit_thickness_check.setToolTip("...") # Tooltip from previous code
        slicing_layout.addWidget(self.limit_thickness_check)
        layout.addWidget(slicing_group)

        # 3D Visualization Parameters Group
        viz_group = QGroupBox("3D视图参数")
        viz_layout = QVBoxLayout(viz_group)
        offset_layout = QHBoxLayout(); offset_layout.addWidget(QLabel("垂直偏移:"))
        self.offset_spin = QDoubleSpinBox(); self.offset_spin.setRange(0.0, 10.0); self.offset_spin.setSingleStep(0.1); self.offset_spin.setValue(0.5)
        self.offset_spin.valueChanged.connect(self._update_3d_view_presentation)
        offset_layout.addWidget(self.offset_spin); viz_layout.addLayout(offset_layout)
        point_size_layout = QHBoxLayout(); point_size_layout.addWidget(QLabel("点大小:"))
        self.point_size_spin = QSpinBox(); self.point_size_spin.setRange(1, 10); self.point_size_spin.setValue(2)
        self.point_size_spin.valueChanged.connect(self._update_3d_view_presentation)
        point_size_layout.addWidget(self.point_size_spin); viz_layout.addLayout(point_size_layout)
        self.use_color_check = QCheckBox("显示原始颜色"); self.use_color_check.setChecked(True)
        self.use_color_check.stateChanged.connect(self._update_3d_view_presentation)
        viz_layout.addWidget(self.use_color_check); layout.addWidget(viz_group)

        # Action Buttons Group
        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout(action_group)
        generate_slices_btn = QPushButton("生成切片并预览"); generate_slices_btn.setToolTip("...")
        generate_slices_btn.clicked.connect(self._start_slice_processing)
        action_layout.addWidget(generate_slices_btn)
        export_all_btn = QPushButton("导出所有数据"); export_all_btn.setToolTip("...")
        export_all_btn.clicked.connect(self._export_all_data)
        action_layout.addWidget(export_all_btn)
        layout.addWidget(action_group)
        layout.addStretch()
        close_btn = QPushButton("关闭"); close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

    def setup_density_analysis_tab(self, layout):
        """Populates the 'Density Analysis' tab."""
        # Density Calculation Group
        density_calc_group = QGroupBox("密度计算")
        dcg_layout = QFormLayout(density_calc_group) # Use QFormLayout for label-widget pairs
        self.density_resolution_combo = QComboBox()
        self.density_resolution_combo.addItems(["256x256", "512x512", "1024x1024", "2048x2048"])
        self.density_resolution_combo.setCurrentText(f"{self.DEFAULT_DENSITY_RESOLUTION}x{self.DEFAULT_DENSITY_RESOLUTION}")
        dcg_layout.addRow("密度网格分辨率:", self.density_resolution_combo)
        self.density_colormap_combo = QComboBox()
        self.density_colormap_combo.addItems(self.AVAILABLE_COLORMAPS)
        self.density_colormap_combo.setCurrentText("viridis") # Default colormap
        dcg_layout.addRow("颜色映射:", self.density_colormap_combo)
        self.update_density_btn = QPushButton("计算/更新密度图")
        self.update_density_btn.setToolTip("根据当前切片数据和设置计算密度图")
        self.update_density_btn.clicked.connect(self._start_density_processing)
        dcg_layout.addRow(self.update_density_btn)
        layout.addWidget(density_calc_group)

        # Display Control Group
        display_group = QGroupBox("显示控制")
        dg_layout = QFormLayout(display_group)
        self.density_display_combo = QComboBox()
        self.density_display_combo.addItems(["选中切片", "逻辑运算结果"]) # Initial options
        self.density_display_combo.setEnabled(False) # Enable after calculation
        self.density_display_combo.currentIndexChanged.connect(self._update_center_view)
        dg_layout.addRow("显示内容:", self.density_display_combo)
        # Add other controls like colorbar toggle if needed later
        layout.addWidget(display_group)

        # Logic Operation Group
        logic_op_group = QGroupBox("逻辑运算")
        log_layout = QVBoxLayout(logic_op_group)
        # Slice A selection
        slice_a_layout = QHBoxLayout()
        self.select_slice_a_btn = QPushButton("选择切片 A")
        self.select_slice_a_btn.clicked.connect(self._select_slice_a)
        self.slice_a_label = QLabel("A: 未选")
        slice_a_layout.addWidget(self.select_slice_a_btn)
        slice_a_layout.addWidget(self.slice_a_label)
        log_layout.addLayout(slice_a_layout)
        # Slice B selection
        slice_b_layout = QHBoxLayout()
        self.select_slice_b_btn = QPushButton("选择切片 B")
        self.select_slice_b_btn.clicked.connect(self._select_slice_b)
        self.slice_b_label = QLabel("B: 未选")
        slice_b_layout.addWidget(self.select_slice_b_btn)
        slice_b_layout.addWidget(self.slice_b_label)
        log_layout.addLayout(slice_b_layout)
        # Operation selection
        op_layout = QHBoxLayout()
        op_layout.addWidget(QLabel("操作:"))
        self.logic_op_combo = QComboBox()
        self.logic_op_combo.addItems(["差分 (A-B)", "差分 (B-A)", "并集 (A | B)", "交集 (A & B)"]) # Example ops
        op_layout.addWidget(self.logic_op_combo)
        log_layout.addLayout(op_layout)
        # Compute button
        self.compute_logic_op_btn = QPushButton("计算逻辑运算")
        self.compute_logic_op_btn.clicked.connect(self._compute_logic_operation)
        self.compute_logic_op_btn.setEnabled(False) # Enable when A and B are selected
        log_layout.addWidget(self.compute_logic_op_btn)
        layout.addWidget(logic_op_group)

        layout.addStretch()

    # --- Initialization ---
    def _initialize_plotter_view(self):
        # ... (Implementation remains the same) ...
        if self.plotter is None:
            if DEBUG_MODE: print("DEBUG: _initialize_plotter_view - Plotter is None, cannot initialize.")
            return
        if DEBUG_MODE: print("DEBUG: _initialize_plotter_view called.")
        try:
            if DEBUG_MODE: print("DEBUG: _initialize_plotter_view - Setting background...")
            self.plotter.set_background("white")
            if DEBUG_MODE: print("DEBUG: _initialize_plotter_view - Adding initial text...")
            self.plotter.add_text("请在右侧面板设置参数并点击“生成切片”", position="upper_left", font_size=12, name="init_text")
            if DEBUG_MODE: print("DEBUG: _initialize_plotter_view - Calling render()...")
            self.plotter.render()
            if DEBUG_MODE: print("DEBUG: _initialize_plotter_view - Initialization render finished.")
        except Exception as e:
            print(f"ERROR: Failed during plotter initialization: {e}")
            if DEBUG_MODE: traceback.print_exc()
            try:
                 self.plotter.add_text(f"渲染初始化错误:\n{e}", position='center', color='red', font_size=10); self.plotter.render()
            except: pass


    # --- Processing ---
    def _start_slice_processing(self):
        # ... (Implementation remains the same, ensures limit_thickness is passed) ...
        if DEBUG_MODE: print("DEBUG: _start_slice_processing called.")
        # (Checks for plotter, point_cloud, running thread)
        if self.plotter is None or self.original_point_cloud is None or self.original_point_cloud.n_points == 0 or (self.slice_processing_thread and self.slice_processing_thread.isRunning()): return
        if DEBUG_MODE: print("DEBUG: _start_slice_processing - Clearing previous results.")
        self.slice_list_widget.clear(); self.current_slices.clear(); self.slice_metadata.clear(); self.density_matrices.clear(); self.density_pixmaps.clear(); self.density_params.clear()
        self.selected_slice_a = None; self.selected_slice_b = None; self._update_logic_op_ui() # Reset logic op state
        try:
            self.plotter.clear(); self.plotter.remove_actor("init_text", render=False)
            self.plotter.add_text("正在生成切片...", position="upper_left", font_size=12, name="status_text")
            self.plotter.render(); QApplication.processEvents()
        except Exception as e: print(f"ERROR: Failed to clear plotter or add text: {e}")
        num_slices = self.num_slices_spin.value(); thickness = self.thickness_spin.value(); limit_thickness = self.limit_thickness_check.isChecked(); thumbnail_size = self.slice_list_widget.iconSize()
        if DEBUG_MODE: print("DEBUG: _start_slice_processing - Setting up progress dialog.")
        self.progress_dialog = QProgressDialog("正在处理切片...", "取消", 0, 100, self); self.progress_dialog.setWindowTitle("切片处理"); self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True); self.progress_dialog.setAutoReset(True); self.progress_dialog.canceled.connect(self._cancel_processing)
        QTimer.singleShot(50, self.progress_dialog.show)
        if DEBUG_MODE: print("DEBUG: _start_slice_processing - Starting SliceProcessingThread.")
        self.slice_processing_thread = SliceProcessingThread(
            self.original_point_cloud,
            num_slices, thickness,
            limit_thickness, thumbnail_size
        )
        self.slice_processing_thread.progress.connect(self._update_progress)
        self.slice_processing_thread.slice_ready.connect(self._collect_slice_data)
        self.slice_processing_thread.thumbnail_ready.connect(self._add_thumbnail_item)
        self.slice_processing_thread.finished.connect(self._slice_processing_finished)
        self.slice_processing_thread.start()
        if DEBUG_MODE: print("DEBUG: _start_slice_processing - Thread started.")

    def _start_density_processing(self):
        """Starts the background thread for density calculation."""
        if DEBUG_MODE: print("DEBUG: _start_density_processing called.")
        if not self.current_slices:
            QMessageBox.warning(self, "无切片", "请先生成切片数据。")
            return
        if self.density_processing_thread and self.density_processing_thread.isRunning():
            QMessageBox.warning(self, "处理中", "当前正在计算密度图，请稍候或取消。")
            return

        # Clear previous density results
        if DEBUG_MODE: print("DEBUG: _start_density_processing - Clearing previous density and logic op results.")
        self.density_matrices.clear()
        self.density_pixmaps.clear()
        self.density_params.clear()
        self.logic_op_result_matrix = None
        self.logic_op_result_pixmap = None
        # Reset selection state for logic ops
        self.selected_slice_a = None
        self.selected_slice_b = None
        self._update_logic_op_ui()  # Update UI to reflect cleared state
        # Set display combo back to slice view initially, handle potential errors if combo is empty
        try:
            if self.density_display_combo.count() > 0:  # Ensure items exist before setting index
                self.density_display_combo.setCurrentIndex(0)  # Index 0 should be "选中切片"
            else:
                # Handle case where combo might not be populated yet (shouldn't happen ideally)
                self.density_display_combo.clear()
                self.density_display_combo.addItems(["选中切片", "逻辑运算结果"])
                self.density_display_combo.setCurrentIndex(0)
                self.density_display_combo.setEnabled(False)  # Keep disabled until data is ready

        except Exception as e:
            print(f"Warning: Error resetting density display combo: {e}")

        # Update center view immediately to show status/clear old view
        self._update_center_view()

        resolution_text = self.density_resolution_combo.currentText()
        try:
            grid_resolution = int(resolution_text.split('x')[0])
        except:
            grid_resolution = self.DEFAULT_DENSITY_RESOLUTION # Fallback
        colormap_name = self.density_colormap_combo.currentText()
        overall_xy_bounds = get_overall_xy_bounds(self.current_slices)

        if overall_xy_bounds is None:
             QMessageBox.warning(self, "无有效边界", "无法计算所有切片的有效XY边界，无法生成密度图。")
             return

        # Setup progress dialog
        self.progress_dialog = QProgressDialog("正在计算密度图...", "取消", 0, 100, self)
        # ... (progress dialog setup as in _start_slice_processing) ...
        self.progress_dialog.setWindowTitle("密度计算")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        self.progress_dialog.canceled.connect(self._cancel_processing)
        QTimer.singleShot(50, self.progress_dialog.show)

        # Start thread
        if DEBUG_MODE: print(f"DEBUG: Starting DensityProcessingThread. Resolution={grid_resolution}, Colormap={colormap_name}")
        self.density_processing_thread = DensityProcessingThread(
            self.current_slices, overall_xy_bounds, grid_resolution, colormap_name
        )
        self.density_processing_thread.progress.connect(self._update_progress)
        self.density_processing_thread.density_map_ready.connect(self._collect_density_data)
        self.density_processing_thread.finished.connect(self._density_processing_finished)
        self.density_processing_thread.start()

    # --- Data Collection Callbacks ---
    def _update_progress(self, value, message):
        # ... (Implementation remains the same) ...
        if self.progress_dialog:
            try: self.progress_dialog.setValue(value); self.progress_dialog.setLabelText(message)
            except RuntimeError: self.progress_dialog = None

    def _cancel_processing(self):
        """Cancel any running background thread."""
        if DEBUG_MODE: print("DEBUG: _cancel_processing called.")
        if self.slice_processing_thread and self.slice_processing_thread.isRunning():
            print("INFO: Canceling slice processing...")
            self.slice_processing_thread.stop()
        if self.density_processing_thread and self.density_processing_thread.isRunning():
            print("INFO: Canceling density processing...")
            self.density_processing_thread.stop()
        # Progress dialog cancellation is handled by its own signal

    def _collect_slice_data(self, index, slice_data, height_range):
        # ... (Implementation remains the same) ...
        if DEBUG_MODE:
            print(f"DEBUG: _collect_slice_data received for index {index}. Data valid: {slice_data is not None}")
        self.current_slices[index] = slice_data

    def _add_thumbnail_item(self, index, pixmap, metadata):
        # ... (Implementation remains the same) ...
        if DEBUG_MODE:
            print(f"DEBUG: _add_thumbnail_item received for index {index}. Pixmap valid: {not pixmap.isNull()}")
        item = QListWidgetItem(f"Slice {index}")
        item.setIcon(QIcon(pixmap))
        item.setData(Qt.ItemDataRole.UserRole, index)
        self.slice_list_widget.addItem(item)
        self.slice_metadata[index] = metadata
        if DEBUG_MODE:
            print(f"DEBUG: _add_thumbnail_item - Item added for index {index}, metadata stored.")


    def _collect_density_data(self, index, density_matrix, heatmap_pixmap, density_params):
        """Collect density data from the thread."""
        if DEBUG_MODE:
            print(f"DEBUG: _collect_density_data for index {index}. Matrix shape: {density_matrix.shape}, Pixmap valid: {not heatmap_pixmap.isNull()}")
        self.density_matrices[index] = density_matrix
        self.density_pixmaps[index] = heatmap_pixmap
        self.density_params[index] = density_params

    # --- Processing Finished Callbacks ---
    def _slice_processing_finished(self, success):
        # ... (Implementation similar to previous, updates 3D view) ...
        if DEBUG_MODE:
            print(f"DEBUG: _slice_processing_finished called. Success: {success}")
        if self.progress_dialog:
            try:
                self.progress_dialog.setValue(100)
            except RuntimeError:
                self.progress_dialog = None
        self.slice_processing_thread = None
        if self.plotter:
             try:
                 self.plotter.remove_actor("status_text", render=False)
             except Exception as e:
                 print(f"WARNING: Could not remove status text: {e}")
        if success:
            print(f"INFO: Successfully processed {len(self.current_slices)} slices.")
            self._update_3d_view_presentation() # Update 3D view first
            # Enable density analysis tab/button now
            self.update_density_btn.setEnabled(True)
            self.density_display_combo.setEnabled(True) # Enable display switch
        else:
            # (Error handling remains similar)
            was_canceled = False # ... (Check progress_dialog.wasCanceled()) ...
            if was_canceled: QMessageBox.information(self, "已取消", "切片处理已取消.")
            else: QMessageBox.warning(self, "处理失败", "切片处理失败.")
            if self.plotter: self.plotter.clear(); self.plotter.add_text("处理失败或取消",...); self.plotter.render()
        if self.progress_dialog:
            try:
                self.progress_dialog.close()
            except RuntimeError:
                pass
            self.progress_dialog = None

    def _density_processing_finished(self, success):
        """Called when density processing finishes."""
        if DEBUG_MODE:
            print(f"DEBUG: _density_processing_finished called. Success: {success}")
        if self.progress_dialog:
            try:
                self.progress_dialog.setValue(100)
            except RuntimeError:
                self.progress_dialog = None
        self.density_processing_thread = None

        if success:
            print(f"INFO: Successfully processed densities for {len(self.density_matrices)} slices.")
            # --- Reset display mode to show a generated slice ---
            self.density_display_combo.setEnabled(True)  # Enable the dropdown
            # Find the first available density map to display
            first_available_index = -1
            sorted_indices = sorted(self.density_pixmaps.keys())
            if sorted_indices:
                first_available_index = sorted_indices[0]
            else:  # Handle case where no density maps were generated at all
                print("Warning: Density processing finished, but no density pixmaps were generated.")

            if first_available_index != -1:
                self.current_density_display_mode = "选中切片"
                self.density_display_combo.setCurrentText(self.current_density_display_mode)  # Set combo text
                self.current_density_display_index = first_available_index
                if DEBUG_MODE: print(
                    f"DEBUG: _density_processing_finished - Setting view to Slice {first_available_index}")
            else:
                # If no pixmaps available, keep showing placeholder/error
                self.current_density_display_mode = ""  # Or some indicator state
                self.density_view_label.setText("无有效密度图生成")
            self._update_center_view()
            self.density_display_combo.setEnabled(True) # Ensure enabled
            self._update_logic_op_ui() # Update logic op buttons based on data
        else:
            # (Error handling similar to slice processing finished)
             was_canceled = False # ... (Check progress_dialog.wasCanceled()) ...
             if was_canceled: QMessageBox.information(self, "已取消", "密度计算已取消。")
             else: QMessageBox.warning(self, "处理失败", "密度计算过程中发生错误。")
             # Optionally clear density view label
             self.density_view_label.setText("密度计算失败或取消")

        if self.progress_dialog:
            try: self.progress_dialog.close()
            except RuntimeError: pass
            self.progress_dialog = None

    # --- UI Update and Interaction ---
    def _handle_tab_change(self, index):
        """Switches the center view when the right tab changes."""
        if DEBUG_MODE: print(f"DEBUG: Tab changed to index {index}")
        if index == 0: # View Control Tab -> Show 3D Plotter
            self.center_stacked_widget.setCurrentIndex(0)
        elif index == 1: # Density Analysis Tab -> Show 2D Label
            self.center_stacked_widget.setCurrentIndex(1)
            # Update the 2D label based on current density display mode/index
            self._update_density_display_label()

    def _update_center_view(self):
        """Updates the central widget display based on tab and density display mode."""
        current_tab_index = self.right_tab_widget.currentIndex()
        if current_tab_index == 0: # 3D View
            if DEBUG_MODE: print("DEBUG: _update_center_view - Switching to 3D view (index 0)")
            self.center_stacked_widget.setCurrentIndex(0)
            # Ensure 3D view is rendered if needed (e.g., if it wasn't visible before)
            if self.plotter:
                try: self.plotter.render()
                except Exception as e: print(f"ERROR: Failed to render plotter on tab switch: {e}")
        elif current_tab_index == 1: # Density View
            if DEBUG_MODE: print("DEBUG: _update_center_view - Switching to Density view (index 1)")
            self.center_stacked_widget.setCurrentIndex(1)
            self._update_density_display_label()

    def _update_density_display_label(self):
        """Updates the content of the density view QLabel."""
        pixmap_to_show = QPixmap()
        display_mode = self.density_display_combo.currentText()
        label_text = "" # Text to show if pixmap is invalid

        if DEBUG_MODE: print(f"DEBUG: _update_density_display_label - Mode: '{display_mode}'")

        if display_mode == "选中切片":
            # Find the currently selected item in the list widget *if possible*
            selected_items = self.slice_list_widget.selectedItems()
            target_index = -1
            if selected_items:
                target_index = selected_items[0].data(Qt.ItemDataRole.UserRole)
            elif self.density_pixmaps: # Fallback to first available if nothing selected
                target_index = sorted(self.density_pixmaps.keys())[0]

            if target_index != -1 and target_index in self.density_pixmaps:
                self.current_density_display_index = target_index # Update state
                pixmap_to_show = self.density_pixmaps[target_index]
                label_text = f"切片 {target_index} 密度图" # Default text if pixmap fails
                if DEBUG_MODE: print(f"DEBUG: Showing density pixmap for index {target_index}")
            else:
                label_text = f"切片密度图不可用\n(索引: {target_index})"
                if DEBUG_MODE: print(f"DEBUG: Density pixmap not available for index {target_index}")

        elif display_mode == "逻辑运算结果":
            if self.logic_op_result_pixmap and not self.logic_op_result_pixmap.isNull():
                pixmap_to_show = self.logic_op_result_pixmap
                label_text = "逻辑运算结果"
                if DEBUG_MODE: print("DEBUG: Showing logic op result pixmap.")
            else:
                label_text = "无逻辑运算结果"
                if DEBUG_MODE: print("DEBUG: Logic op result pixmap not available.")

        else: # Handle unexpected combo text or empty state
            label_text = "请先计算密度图"


        if not pixmap_to_show.isNull():
            scaled_pixmap = pixmap_to_show.scaled(self.density_view_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.density_view_label.setPixmap(scaled_pixmap)
            # Clear text when pixmap is shown successfully
            self.density_view_label.setText("")
        else:
             self.density_view_label.setText(label_text)
             self.density_view_label.setPixmap(QPixmap()) # Ensure no old pixmap remains

    def resizeEvent(self, event):
        """Handle window resize to update scaled pixmap."""
        super().resizeEvent(event)
        # If density view is active, update its pixmap scaling
        if self.center_stacked_widget.currentIndex() == 1:
            self._update_density_display_label()


    def _on_selection_changed(self):
        """Update density display when selection changes in the list."""
        selected_items = self.slice_list_widget.selectedItems()
        if selected_items:
            # Update density display to show the first selected item's density map
            first_selected_index = selected_items[0].data(Qt.ItemDataRole.UserRole)
            if self.current_density_display_mode == "Slice":
                 self.current_density_display_index = first_selected_index
                 self._update_center_view()
                 if DEBUG_MODE: print(f"DEBUG: List selection changed, density view set to index {first_selected_index}")
        else:
             # Optional: Clear display or show a default message if nothing selected
             pass

    def _show_list_context_menu(self, pos: QPoint):
        """Show context menu on the list widget."""
        item = self.slice_list_widget.itemAt(pos)
        if not item: return # No item under cursor

        index = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu()
        action_a = QAction(f"设为逻辑运算切片 A (当前: {self.selected_slice_a if self.selected_slice_a is not None else '无'})", self)
        action_b = QAction(f"设为逻辑运算切片 B (当前: {self.selected_slice_b if self.selected_slice_b is not None else '无'})", self)

        action_a.triggered.connect(lambda: self._set_logic_operand('A', index))
        action_b.triggered.connect(lambda: self._set_logic_operand('B', index))

        # Enable actions only if density data is available for the selected slice
        if index not in self.density_matrices:
            action_a.setEnabled(False)
            action_b.setEnabled(False)
            action_a.setText(f"设为切片 A (密度未计算)")
            action_b.setText(f"设为切片 B (密度未计算)")


        menu.addAction(action_a)
        menu.addAction(action_b)
        menu.exec(self.slice_list_widget.mapToGlobal(pos))

    def _set_logic_operand(self, operand, index):
        """Set Slice A or B for logic operation."""
        if operand == 'A':
            self.selected_slice_a = index
            self.slice_a_label.setText(f"A: {index}")
            if DEBUG_MODE: print(f"DEBUG: Slice A selected: {index}")
        elif operand == 'B':
            self.selected_slice_b = index
            self.slice_b_label.setText(f"B: {index}")
            if DEBUG_MODE: print(f"DEBUG: Slice B selected: {index}")
        self._update_logic_op_ui()

    def _select_slice_a(self):
        """Handles the 'Select Slice A' button click (alternative selection method)."""
        # This provides an alternative way, maybe less intuitive than context menu
        QMessageBox.information(self, "选择切片 A", "请在左侧列表中单击选择一个切片作为操作数 A。")
        # We might need a state machine here to capture the next click on the list
        # For now, relying on context menu is simpler. Button can just be a reminder.

    def _select_slice_b(self):
        QMessageBox.information(self, "选择切片 B", "请在左侧列表中单击选择一个切片作为操作数 B。")

    def _update_logic_op_ui(self):
        """Enable/disable logic operation button based on selections."""
        ready = self.selected_slice_a is not None and self.selected_slice_b is not None \
                and self.selected_slice_a in self.density_matrices \
                and self.selected_slice_b in self.density_matrices
        self.compute_logic_op_btn.setEnabled(ready)
        # Update labels just in case
        self.slice_a_label.setText(f"A: {self.selected_slice_a if self.selected_slice_a is not None else '未选'}")
        self.slice_b_label.setText(f"B: {self.selected_slice_b if self.selected_slice_b is not None else '未选'}")


    def _compute_logic_operation(self):
        """Performs the selected logic operation on density matrices."""
        if not self.compute_logic_op_btn.isEnabled(): return

        if DEBUG_MODE: print("DEBUG: _compute_logic_operation called.")
        idx_a = self.selected_slice_a
        idx_b = self.selected_slice_b
        operation = self.logic_op_combo.currentText()

        matrix_a = self.density_matrices.get(idx_a)
        matrix_b = self.density_matrices.get(idx_b)

        if matrix_a is None or matrix_b is None:
            QMessageBox.critical(self, "错误", f"无法获取切片 {idx_a} 或 {idx_b} 的密度矩阵。")
            return

        if matrix_a.shape != matrix_b.shape:
             QMessageBox.critical(self, "错误", f"切片 {idx_a} 和 {idx_b} 的密度矩阵形状不匹配。")
             return

        result_matrix = None
        try:
            if operation == "差分 (A-B)":
                # Simple subtraction, might result in negative values
                result_matrix = matrix_a - matrix_b
            elif operation == "差分 (B-A)":
                result_matrix = matrix_b - matrix_a
            elif operation == "并集 (A | B)":
                # Treat density > 0 as "present"
                result_matrix = np.logical_or(matrix_a > 0, matrix_b > 0).astype(float) # Result is 0.0 or 1.0
            elif operation == "交集 (A & B)":
                result_matrix = np.logical_and(matrix_a > 0, matrix_b > 0).astype(float) # Result is 0.0 or 1.0
            # Add more operations like XOR etc. if needed

            if result_matrix is not None:
                if DEBUG_MODE: print(f"DEBUG: Logic operation '{operation}' completed. Result matrix shape: {result_matrix.shape}")
                # Generate heatmap for the result
                # Decide on colormap/scaling for the result (might need different scaling)
                result_colormap = 'coolwarm' if '差分' in operation else 'binary' # Example scaling
                # Find appropriate vmin/vmax for the result matrix
                if np.all(result_matrix == 0): # Handle all-zero result
                    vmin_res, vmax_res = 0, 1
                elif '差分' in operation:
                    abs_max = np.max(np.abs(result_matrix))
                    vmin_res, vmax_res = -abs_max, abs_max
                else: # Binary results
                     vmin_res, vmax_res = 0, 1

                self.logic_op_result_matrix = result_matrix # Store raw result
                self.logic_op_result_pixmap = create_density_heatmap(result_matrix, result_colormap, vmin=vmin_res, vmax=vmax_res)

                # Switch view to show result
                self.density_display_combo.setCurrentText("逻辑运算结果")
                self._update_center_view()
            else:
                 QMessageBox.warning(self, "未实现", f"操作 '{operation}' 尚未实现。")

        except Exception as e:
            print(f"ERROR: Error during logic operation '{operation}': {e}")
            if DEBUG_MODE: traceback.print_exc()
            QMessageBox.critical(self, "计算错误", f"逻辑运算时发生错误:\n{e}")
            self.logic_op_result_matrix = None
            self.logic_op_result_pixmap = None

    def _update_3d_view_presentation(self):
        """
        Updates the 3D view presentation based on current slices and
        presentation parameters (offset, size, color) without regenerating slice data.
        """
        if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation called.")
        if self.plotter is None or self.center_stacked_widget.currentIndex() != 0:
            if DEBUG_MODE: print(
                "DEBUG: _update_3d_view_presentation - Plotter is None or not in 3D view mode, returning.")
            # Only update if plotter exists and the 3D view tab is active
            return

        # Ensure we are on the 3D view tab in the center
        self.center_stacked_widget.setCurrentIndex(0)

        try:
            if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - Clearing actors.")
            self.plotter.clear_actors()
        except Exception as e:
            print(f"ERROR: Failed to clear plotter actors in _update_3d_view_presentation: {e}")
            # Attempt to continue if clearing failed

        if not self.current_slices:
            if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - No slices exist to display.")
            self.plotter.add_text("无切片可显示。\n请点击“生成切片并预览”", position="upper_left", font_size=12,
                                  name="status_text")
            try:
                self.plotter.render()
            except Exception as render_err:
                print(f"ERROR: Render failed after clearing actors with no slices: {render_err}")
            return

        # Get current presentation parameters from UI
        offset_value = self.offset_spin.value()
        point_size = self.point_size_spin.value()
        use_color = self.use_color_check.isChecked()
        if DEBUG_MODE: print(
            f"DEBUG: _update_3d_view_presentation - Params: offset={offset_value}, point_size={point_size}, use_color={use_color}")

        actors = []
        current_offset = 0.0
        sorted_indices = sorted(self.current_slices.keys())
        all_bounds = []  # For camera reset

        if DEBUG_MODE: print(
            f"DEBUG: _update_3d_view_presentation - Re-adding meshes for {len(sorted_indices)} slices...")
        for i in sorted_indices:
            slice_data = self.current_slices.get(i)
            # if DEBUG_MODE: print(f"DEBUG: Re-adding slice {i}. Data valid: {slice_data is not None}") # Can be too verbose
            if slice_data is None or slice_data.n_points == 0:
                current_offset += offset_value
                continue

            offset_slice = slice_data.copy(deep=True)  # Work on a copy
            offset_slice.points[:, 2] += current_offset
            # Check bounds validity before adding
            if offset_slice.bounds[0] < offset_slice.bounds[1]:
                all_bounds.extend(offset_slice.bounds)
            # if DEBUG_MODE: print(f"DEBUG: Slice {i} offset applied. Current offset: {current_offset}") # Too verbose

            try:
                # if DEBUG_MODE: print(f"DEBUG: Adding mesh for slice {i}...") # Too verbose
                if 'colors' in offset_slice.point_data and use_color:
                    actor = self.plotter.add_mesh(offset_slice, scalars='colors', rgb=True, point_size=point_size)
                else:
                    actor = self.plotter.add_mesh(offset_slice, color='grey', point_size=point_size)

                if actor:
                    actors.append(actor)
                    # if DEBUG_MODE: print(f"DEBUG: Actor added successfully for slice {i}.") # Too verbose
                else:
                    print(f"WARNING: Failed to add actor for slice {i} (add_mesh returned None).")

            except Exception as e:
                print(f"ERROR: Error adding slice {i} to plotter: {e}")
                if DEBUG_MODE: traceback.print_exc()

            current_offset += offset_value

        if actors:
            if DEBUG_MODE: print(f"DEBUG: _update_3d_view_presentation - {len(actors)} actors added. Resetting camera.")
            try:
                # Reset camera based on the new layout of actors
                if all_bounds:
                    min_x = min(all_bounds[0::6]) if all_bounds[0::6] else 0;
                    max_x = max(all_bounds[1::6]) if all_bounds[1::6] else 1
                    min_y = min(all_bounds[2::6]) if all_bounds[2::6] else 0;
                    max_y = max(all_bounds[3::6]) if all_bounds[3::6] else 1
                    min_z_off = min(all_bounds[4::6]) if all_bounds[4::6] else 0;
                    max_z_off = max(all_bounds[5::6]) if all_bounds[5::6] else 1
                    overall_bounds = [min_x, max_x, min_y, max_y, min_z_off, max_z_off]
                    if DEBUG_MODE: print(f"DEBUG: Resetting camera to overall bounds: {overall_bounds}")
                    self.plotter.reset_camera(bounds=overall_bounds)
                else:
                    self.plotter.reset_camera()  # Fallback

                # Keep the existing view vector unless explicitly changed elsewhere
                # self.plotter.view_vector([1, -1, 0.5], viewup=[0, 0, 1]) # Don't force view change on param update
                if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - Camera reset.")
            except Exception as e:
                print(f"ERROR: Error resetting camera: {e}")
                if DEBUG_MODE: traceback.print_exc()
        elif self.current_slices:  # Check if slices exist but were all empty
            if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - No actors added, slices might be empty.")
            self.plotter.add_text("所有切片均为空。", position="upper_left", font_size=12, name="status_text")
        # else: # No slices exist, message handled earlier or by processing finished

        try:
            if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - Calling plotter.render().")
            self.plotter.render()
            if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - plotter.render() finished.")
        except Exception as e:
            print(f"ERROR: Exception during final plotter.render() in presentation update: {e}")


    # --- Export ---
    def _export_bitmaps(self, indices_to_export):
        # ... (Implementation modified to include density matrix/heatmap export) ...
        if DEBUG_MODE:
            print(f"DEBUG: _export_bitmaps called for indices: {indices_to_export}")
        if not indices_to_export:
            return
        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not export_dir:
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_export_path = os.path.join(export_dir, f"batch_slice_export_{timestamp}")
        os.makedirs(base_export_path, exist_ok=True)
        if DEBUG_MODE:
            print(f"DEBUG: _export_bitmaps - Exporting to: {base_export_path}")
        global_params = {
            "export_time": datetime.datetime.now().isoformat(),
            "original_point_cloud_source": self.source_filename,
            "num_slices_param": self.num_slices_spin.value(),
            "slice_thickness_param": self.thickness_spin.value(),
        }
        global_params_file = os.path.join(base_export_path, "export_parameters.json")
        with open(global_params_file, 'w', encoding='utf-8') as f:
            json.dump(global_params, f, ensure_ascii=False, indent=2)
        overall_xy_bounds = get_overall_xy_bounds(self.current_slices)
        if DEBUG_MODE:
            print(f"DEBUG: _export_bitmaps - Using overall XY bounds for rendering: {overall_xy_bounds}")
        export_progress = QProgressDialog("正在导出数据...", "取消", 0, len(indices_to_export), self); export_progress.setWindowTitle("导出进度"); export_progress.setWindowModality(Qt.WindowModality.WindowModal); export_progress.setAutoClose(True); export_progress.setAutoReset(True); export_progress.show()
        exported_count = 0; exported_pcd_count = 0; exported_density_count = 0
        for i, index in enumerate(indices_to_export):
             export_progress.setValue(i);
             if export_progress.wasCanceled(): print("INFO: Export canceled by user."); break
             if DEBUG_MODE: print(f"DEBUG: _export_bitmaps - Processing index {index}...")
             slice_data_pv = self.current_slices.get(index); metadata = self.slice_metadata.get(index)
             density_matrix = self.density_matrices.get(index); density_pixmap = self.density_pixmaps.get(index)
             density_params_saved = self.density_params.get(index) # Params used for saved pixmap

             if metadata is None: print(f"WARNING: Metadata missing for slice index {index} during export. Skipping."); continue
             img_np = None; view_params_render = None; render_error_msg = None; bitmap_saved = False; pcd_saved = False; density_saved = False
             # --- Render Bitmap ---
             if not metadata.get("is_empty", False) and slice_data_pv is not None:
                 export_progress.setLabelText(f"正在渲染切片位图 {index}...") ; QApplication.processEvents()
                 try:
                     img_np, view_params_render = render_slice_to_image(slice_data_pv, self.BITMAP_EXPORT_RESOLUTION, overall_xy_bounds, is_thumbnail=False)
                     if img_np is None: render_error_msg = "Bitmap rendering failed"
                 except Exception as render_err: render_error_msg = f"Bitmap rendering error: {render_err}"
             else: print(f"INFO: Skipping bitmap rendering for empty slice {index}.")

             # --- Save Metadata (include density params if available) ---
             meta_filename = os.path.join(base_export_path, f"slice_{index}_metadata.json")
             metadata["view_params_render"] = view_params_render if img_np is not None else None
             if render_error_msg: metadata["render_error"] = render_error_msg
             if density_params_saved: metadata["density_params"] = density_params_saved # Add density info
             export_data = {"slice_index": index, "metadata": metadata}
             try:
                 with open(meta_filename, 'w', encoding='utf-8') as f: json.dump(export_data, f, ensure_ascii=False, indent=2)
             except Exception as meta_save_err: print(f"ERROR: Failed to save metadata for slice {index}: {meta_save_err}")

             # --- Save Bitmap ---
             if img_np is not None:
                 try:
                     bitmap_filename = os.path.join(base_export_path, f"slice_{index}_bitmap.png")
                     img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                     if cv2.imwrite(bitmap_filename, img_bgr): bitmap_saved = True
                     else: print(f"ERROR: Failed to save bitmap file: {bitmap_filename}")
                 except Exception as export_err: print(f"ERROR: Failed to save bitmap for slice {index}: {export_err}")

             # --- Save PCD ---
             if slice_data_pv is not None and slice_data_pv.n_points > 0:
                 pcd_filename = os.path.join(base_export_path, f"slice_{index}.pcd")
                 try:
                    export_progress.setLabelText(f"正在保存 PCD {index}...") ; QApplication.processEvents()
                    points = slice_data_pv.points; o3d_pcd = o3d.geometry.PointCloud(); o3d_pcd.points = o3d.utility.Vector3dVector(points)
                    if 'colors' in slice_data_pv.point_data:
                        colors = slice_data_pv['colors']
                        if colors.ndim == 2 and colors.shape[1] == 3 and colors.shape[0] == len(points): o3d_pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
                    if o3d.io.write_point_cloud(pcd_filename, o3d_pcd, write_ascii=False, compressed=True): pcd_saved = True; exported_pcd_count += 1
                    else: print(f"ERROR: Open3D failed to save PCD file: {pcd_filename}")
                 except Exception as pcd_err: print(f"ERROR: Failed to save PCD for slice {index}: {pcd_err}")
             elif metadata.get("is_empty", False): pcd_saved = True # Consider empty slice handled

             # --- Save Density Data ---
             if density_matrix is not None:
                 try:
                     export_progress.setLabelText(f"正在保存密度数据 {index}...") ; QApplication.processEvents()
                     # Save matrix as numpy binary
                     matrix_filename = os.path.join(base_export_path, f"slice_{index}_density_matrix.npy")
                     np.save(matrix_filename, density_matrix)
                     # Save heatmap pixmap as image
                     heatmap_filename = os.path.join(base_export_path, f"slice_{index}_density_heatmap.png")
                     if density_pixmap and not density_pixmap.isNull():
                         density_pixmap.save(heatmap_filename, "PNG")
                         density_saved = True
                         exported_density_count += 1
                     else: print(f"WARNING: Density heatmap pixmap not available or invalid for slice {index}, matrix saved.")
                 except Exception as density_err: print(f"ERROR: Failed to save density data for slice {index}: {density_err}")
             elif index in self.density_matrices: # Check if density was calculated but matrix is None/empty
                  density_saved = True # Consider handled if calculation was attempted

             if bitmap_saved or pcd_saved or density_saved or metadata.get("is_empty", False):
                  exported_count += 1

        export_progress.setValue(len(indices_to_export))
        QMessageBox.information(
            self, "导出完成",
            f"处理完成 {len(indices_to_export)} 个切片。\n"
            f"成功导出 {exported_count} 个项目的数据。\n"
            f"(PCD: {exported_pcd_count}, 密度: {exported_density_count})\n"
            f"文件保存在:\n{base_export_path}"
        )

    def _export_selected_data(self): # Renamed from _export_selected_bitmaps
        selected_items = self.slice_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "未选择", "请先在左侧列表中选择要导出的项。")
            return
        indices_to_export = sorted([item.data(Qt.ItemDataRole.UserRole) for item in selected_items])
        self._export_bitmaps(indices_to_export) # Call the combined export function

    def _export_all_data(self):
        if not self.slice_metadata: # Check metadata instead of slices for export list
            QMessageBox.warning(self, "无数据", "没有有效的切片元数据可供导出。请先生成切片。")
            return
        indices_to_export = sorted(list(self.slice_metadata.keys()))
        self._export_bitmaps(indices_to_export)

    # --- Close Event ---
    def closeEvent(self, event):
        # ... (Implementation remains the same) ...
        if DEBUG_MODE:
            print("DEBUG: BatchSliceViewerWindow closeEvent called.")
        self._cancel_processing()
        if (self.slice_processing_thread and self.slice_processing_thread.isRunning()) or \
           (self.density_processing_thread and self.density_processing_thread.isRunning()):
            print("INFO: Waiting for processing thread(s) to finish before closing...")
            if self.slice_processing_thread:
                self.slice_processing_thread.wait(1500)
            if self.density_processing_thread:
                self.density_processing_thread.wait(1500)

        if self.plotter:
             if DEBUG_MODE: print("DEBUG: Closing plotter in closeEvent.")
             try: self.plotter.close()
             except Exception as e: print(f"ERROR: Exception while closing plotter in closeEvent: {e}")
        super().closeEvent(event)