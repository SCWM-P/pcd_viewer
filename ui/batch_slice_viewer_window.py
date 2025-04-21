# pcd_viewer/ui/batch_slice_viewer_window.py

import os
import json
import datetime
import numpy as np
import pyvista as pv
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QListWidget,
                             QListWidgetItem, QPushButton, QSplitter, QGroupBox,
                             QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
                             QMessageBox, QAbstractItemView, QProgressBar, QSpacerItem,
                             QSizePolicy, QProgressDialog, QApplication, QTabWidget,
                             QComboBox, QStackedWidget, QMenu, QSlider, QFormLayout)  # Added/Verified imports
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import QPixmap, QImage, QIcon, QPainter, QColor, QAction, QFontMetrics  # Added QFontMetrics
from pyvistaqt import QtInteractor

# 导入项目模块
from ..utils.point_cloud_handler import PointCloudHandler
from ..utils.stylesheet_manager import StylesheetManager
from .. import DEBUG_MODE


# --- Helper Functions (Remain the same as previous correct version) ---
def get_overall_xy_bounds(slices_dict):
    all_bounds_xy = []
    valid_slice_found = False
    for slice_data in slices_dict.values():
        if slice_data is not None and slice_data.n_points > 0:
            b = slice_data.bounds
            if b[0] < b[1] and b[2] < b[3]:
                all_bounds_xy.extend(b[0:4])
                valid_slice_found = True
    if not valid_slice_found or not all_bounds_xy: return None
    xmin = min(all_bounds_xy[0::4])
    xmax = max(all_bounds_xy[1::4])
    ymin = min(all_bounds_xy[2::4])
    ymax = max(all_bounds_xy[3::4])
    x_range = xmax - xmin
    y_range = ymax - ymin
    padding = max(x_range * 0.05, y_range * 0.05, 0.1)
    return [xmin - padding, xmax + padding, ymin - padding, ymax + padding]


def render_slice_to_image(slice_data, size, overall_xy_bounds=None, is_thumbnail=True):
    if DEBUG_MODE: print(f"DEBUG: render_slice_to_image called. is_thumbnail={is_thumbnail}, size={size}")
    if slice_data is None or slice_data.n_points == 0: return None, {}
    plotter = None
    try:
        img_width, img_height = size if isinstance(size, tuple) else (size.width(), size.height())
        plotter = pv.Plotter(off_screen=True, window_size=[img_width, img_height])
        plotter.set_background('white')
        if is_thumbnail:
            actor = plotter.add_mesh(slice_data, color='darkgrey', point_size=1)
        else:
            if 'colors' in slice_data.point_data:
                actor = plotter.add_mesh(slice_data, scalars='colors', rgb=True, point_size=2)
            else:
                actor = plotter.add_mesh(slice_data, color='blue', point_size=2)
        plotter.view_xy()
        bounds_to_use = None
        if overall_xy_bounds:
            zmin = slice_data.bounds[4]
            zmax = slice_data.bounds[5]
            bounds_to_use = overall_xy_bounds + [zmin, zmax]
        elif slice_data and slice_data.bounds[0] < slice_data.bounds[1]:
            bounds_to_use = slice_data.bounds
        if bounds_to_use: plotter.reset_camera(bounds=bounds_to_use)
        img_np = plotter.screenshot(return_img=True)
        cam = plotter.camera
        view_params = {
            "position": list(cam.position), "focal_point": list(cam.focal_point), "up": list(cam.up),
            "parallel_projection": cam.parallel_projection, "parallel_scale": cam.parallel_scale,
            "slice_bounds": list(slice_data.bounds), "render_window_size": [img_width, img_height],
        }
        return img_np, view_params
    except Exception as e:
        print(f"ERROR: Error rendering slice: {e}"); return None, {}
    finally:
        if plotter:
            try:
                plotter.close()
            except Exception:
                pass


def create_density_heatmap(density_matrix, colormap_name='viridis', vmin=None, vmax=None):
    # ... (Implementation remains the same as previous correct version) ...
    if density_matrix is None or density_matrix.size == 0: return QPixmap()
    try:
        if vmin is None: vmin = np.min(density_matrix)
        if vmax is None: vmax = np.max(density_matrix)
        if vmax <= vmin: vmax = vmin + 1e-6
        # Handle potential NaN values before normalization
        nan_mask = np.isnan(density_matrix)
        if np.any(nan_mask):
            density_matrix[nan_mask] = vmin  # Replace NaN with min value

        normalized_matrix = (density_matrix - vmin) / (vmax - vmin)
        normalized_matrix = np.clip(normalized_matrix, 0, 1)
        cmap = plt.get_cmap(colormap_name)
        colored_matrix_rgba = cmap(normalized_matrix, bytes=True)
        height, width, _ = colored_matrix_rgba.shape
        image_data = np.require(colored_matrix_rgba, dtype=np.uint8, requirements='C')
        q_img = QImage(image_data, width, height, width * 4, QImage.Format.Format_RGBA8888)
        if q_img.isNull():
            print("ERROR: create_density_heatmap - QImage failed.")
            return QPixmap()
        return QPixmap.fromImage(q_img)
    except Exception as e:
        print(f"ERROR: create_density_heatmap: {e}")
        return QPixmap()


# --- Background Threads (Remain the same logic, ensure correct parameters passed) ---
class SliceProcessingThread(QThread):
    # ... (Signals remain the same) ...
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
        self.thumbnail_size = thumbnail_size
        self._is_running = True

    def run(self):
        # ... (Implementation remains the same logic as previous correct version) ...
        if DEBUG_MODE: print("DEBUG: SliceProcessingThread run started.")
        if self.point_cloud is None or self.num_slices <= 0 or self.thickness_param <= 0:
            self.finished.emit(False)
            return
        try:
            bounds = self.point_cloud.bounds
            min_z, max_z = bounds[4], bounds[5]
            total_height = max_z - min_z
            if total_height <= 0:
                self.finished.emit(False)
                return
            all_points = self.point_cloud.points
            has_colors = 'colors' in self.point_cloud.point_data
            if has_colors: all_colors = self.point_cloud['colors']
            step = total_height / self.num_slices
            current_start_z = min_z
            actual_thickness = self.thickness_param
            if self.limit_thickness:
                max_allowed_thickness = step
                if actual_thickness > max_allowed_thickness: actual_thickness = max_allowed_thickness
            total_steps = self.num_slices * 2
            generated_slices = []
            height_ranges = []
            for i in range(self.num_slices):
                if not self._is_running: raise InterruptedError("Stopped")
                self.progress.emit(int((i + 1) / total_steps * 100), f"切片 {i + 1}/{self.num_slices}")
                slice_start_z = current_start_z
                slice_end_z = slice_start_z + actual_thickness
                slice_end_z = min(slice_end_z, max_z + 1e-6)
                slice_start_z = min(slice_start_z, slice_end_z)
                indices = np.where((all_points[:, 2] >= slice_start_z) & (all_points[:, 2] <= slice_end_z))[0]
                height_ranges.append((slice_start_z, slice_end_z))
                if len(indices) > 0:
                    slice_points = all_points[indices];
                    slice_cloud = pv.PolyData(slice_points)
                    if has_colors: slice_cloud['colors'] = all_colors[indices]
                    generated_slices.append(slice_cloud);
                    self.slice_ready.emit(i, slice_cloud, (slice_start_z, slice_end_z))
                else:
                    generated_slices.append(None); self.slice_ready.emit(i, None, (slice_start_z, slice_end_z))
                current_start_z += step
            temp_slices_dict = {i: s for i, s in enumerate(generated_slices)}
            overall_xy_bounds = get_overall_xy_bounds(temp_slices_dict)
            for i in range(self.num_slices):
                if not self._is_running: raise InterruptedError("Stopped")
                self.progress.emit(int((self.num_slices + i + 1) / total_steps * 100),
                                   f"缩略图 {i + 1}/{self.num_slices}")
                slice_data = generated_slices[i]
                img_np, view_params = render_slice_to_image(slice_data, self.thumbnail_size, overall_xy_bounds,
                                                            is_thumbnail=True)
                metadata = {"index": i, "height_range": height_ranges[i], "view_params": view_params,
                            "is_empty": slice_data is None or slice_data.n_points == 0}
                if img_np is not None:
                    try:
                        h, w, ch = img_np.shape;
                        image_data_bytes = img_np.tobytes()
                        q_img = QImage(image_data_bytes, w, h, w * ch, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_img);
                        scaled_pixmap = pixmap.scaled(self.thumbnail_size, Qt.AspectRatioMode.KeepAspectRatio,
                                                      Qt.TransformationMode.SmoothTransformation)
                        self.thumbnail_ready.emit(i, scaled_pixmap, metadata)
                    except Exception as qimage_err:
                        print(f"ERROR: QImage/QPixmap failed: {qimage_err}"); placeholder_pixmap = QPixmap(
                            self.thumbnail_size); placeholder_pixmap.fill(
                            Qt.GlobalColor.darkRed); self.thumbnail_ready.emit(i, placeholder_pixmap, metadata)
                else:
                    placeholder_pixmap = QPixmap(self.thumbnail_size); placeholder_pixmap.fill(
                        Qt.GlobalColor.lightGray); painter = QPainter(placeholder_pixmap); painter.drawText(
                        placeholder_pixmap.rect(), Qt.AlignmentFlag.AlignCenter,
                        f"Slice {i}\n(Empty)"); painter.end(); self.thumbnail_ready.emit(i, placeholder_pixmap,
                                                                                         metadata)
            self.finished.emit(True)
        except InterruptedError:
            print("INFO: Slice thread stopped."); self.finished.emit(False)
        except Exception as e:
            print(f"ERROR: Slice thread error: {e}"); self.finished.emit(False)

    def stop(self):
        self._is_running = False


class DensityProcessingThread(QThread):
    # ... (Signals remain the same) ...
    progress = pyqtSignal(int, str)
    density_map_ready = pyqtSignal(int, np.ndarray, QPixmap, dict)
    finished = pyqtSignal(bool)

    def __init__(self, slices_dict, overall_xy_bounds, grid_resolution, colormap_name, parent=None):
        super().__init__(parent);
        self.slices_dict = slices_dict;
        self.overall_xy_bounds = overall_xy_bounds
        self.grid_resolution = grid_resolution;
        self.colormap_name = colormap_name;
        self._is_running = True

    # noinspection PyUnresolvedReferences
    def run(self):
        # ... (Implementation remains the same logic as previous correct version) ...
        if DEBUG_MODE: print("DEBUG: DensityProcessingThread run started.")
        if not self.slices_dict or self.overall_xy_bounds is None: self.finished.emit(False); return
        try:
            xmin, xmax, ymin, ymax = self.overall_xy_bounds;
            bins = [self.grid_resolution, self.grid_resolution];
            range_xy = [[xmin, xmax], [ymin, ymax]]
            num_slices = len(self.slices_dict);
            sorted_indices = sorted(self.slices_dict.keys());
            max_density = 0;
            all_matrices = {}
            for i, index in enumerate(sorted_indices):
                if not self._is_running: raise InterruptedError("Stopped")
                self.progress.emit(int(((i + 1) / (num_slices * 2)) * 100), f"计算密度 {index + 1}/{num_slices}")
                slice_data = self.slices_dict.get(index)
                if slice_data is not None and slice_data.n_points > 0:
                    points_xy = slice_data.points[:, 0:2]
                    density_matrix, _, _ = np.histogram2d(points_xy[:, 0], points_xy[:, 1], bins=bins, range=range_xy)
                    all_matrices[index] = density_matrix;
                    current_max = np.max(density_matrix)
                    if current_max > max_density: max_density = current_max
                else:
                    all_matrices[index] = np.zeros(bins)
            for i, index in enumerate(sorted_indices):
                if not self._is_running:
                    raise InterruptedError("Stopped")
                self.progress.emit(int(((num_slices + i + 1) / (num_slices * 2)) * 100),
                                   f"生成热力图 {index + 1}/{num_slices}")
                density_matrix = all_matrices[index]
                heatmap_pixmap = create_density_heatmap(density_matrix, self.colormap_name, vmin=0, vmax=max_density)
                density_params = {
                    "grid_resolution": self.grid_resolution,
                    "colormap": self.colormap_name,
                    "xy_bounds": self.overall_xy_bounds,
                    "max_density_scale": float(max_density)
                }  # Ensure max_density is float for JSON
                self.density_map_ready.emit(index, density_matrix, heatmap_pixmap, density_params)
            self.finished.emit(True)
        except InterruptedError:
            print("INFO: Density thread stopped."); self.finished.emit(False)
        except Exception as e:
            print(f"ERROR: Density thread error: {e}"); self.finished.emit(False)

    def stop(self):
        self._is_running = False


# --- Main Window Class ---
class BatchSliceViewerWindow(QWidget):
    BITMAP_EXPORT_RESOLUTION = (1024, 1024)
    DEFAULT_DENSITY_RESOLUTION = 512
    AVAILABLE_COLORMAPS = sorted(plt.colormaps())  # Get available matplotlib colormaps
    # Define diverging colormaps suitable for difference maps
    DIVERGING_COLORMAPS = sorted([cm for cm in AVAILABLE_COLORMAPS if
                                  cm.lower() in ['rdbu', 'bwr', 'coolwarm', 'seismic', 'piyg', 'prgn', 'brbg', 'puor']])

    # Define logic operations display names and corresponding numpy functions/lambda
    LOGIC_OPERATIONS = {
        "差分 (A-B)": lambda a, b: a - b,
        "差分 (B-A)": lambda a, b: b - a,
        "绝对差分 |A-B|": lambda a, b: np.abs(a - b),
        "并集 (A | B)": lambda a, b: np.logical_or(a > 0, b > 0).astype(a.dtype),  # Keep original dtype
        "交集 (A & B)": lambda a, b: np.logical_and(a > 0, b > 0).astype(a.dtype),
        "异或 (A ^ B)": lambda a, b: np.logical_xor(a > 0, b > 0).astype(a.dtype),
        "均值 ((A+B)/2)": lambda a, b: (a + b) / 2.0,
        # Add more as needed
    }

    def __init__(self, point_cloud, source_filename="Unknown", parent=None):
        super().__init__(parent)
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow __init__ started.")
        self.setWindowTitle("批量切片与密度分析器")  # Updated title
        self.setMinimumSize(1150, 800)  # Slightly larger default size
        self.setWindowFlags(Qt.WindowType.Window)
        self.setStyleSheet(StylesheetManager.get_light_theme())

        self.original_point_cloud = point_cloud
        self.source_filename = source_filename
        # Data storage
        self.current_slices = {}
        self.slice_metadata = {}
        self.density_matrices = {}
        self.density_pixmaps = {}
        self.density_params = {}
        self.logic_op_result_matrix = None
        self.logic_op_result_pixmap = None
        self.batch_op_results = {}  # { (op_str, i, j): result_matrix }
        self.batch_op_pixmaps = {}  # { (op_str, i, j): result_pixmap }
        self.batch_op_params = {}  # { "op_str": str, "k": int, "loop": bool, "indices": list }
        # UI State
        self.selected_slice_a = None
        self.selected_slice_b = None
        self.current_density_display_mode = "选中切片"  # "选中切片", "逻辑运算结果", "批量运算结果"
        self.current_density_display_index = 0  # Index for "选中切片" mode
        self.current_batch_op_display_key = None  # Key for "批量运算结果" mode (op_str, i, j)
        # Threads & Timers
        self.slice_processing_thread = None
        self.density_processing_thread = None
        self.batch_op_thread = None  # Consider a thread for batch ops if slow
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_next_batch_result)
        # Widgets
        self.progress_dialog = None
        self.plotter = None

        self.setup_ui()
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow __init__ finished.")

    def setup_ui(self):
        # ... (Main layout and splitter setup remain the same) ...
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow setup_ui started.")
        main_layout = QHBoxLayout(self)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)
        self.setup_left_panel()
        self.setup_center_panel()
        self.setup_right_panel()
        self.splitter.setSizes([280, 600, 300])  # Adjusted sizes
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow setup_ui finished.")

    def setup_left_panel(self):
        # ... (List widget setup remains the same, connects _show_list_context_menu) ...
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
        self.slice_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.slice_list_widget.customContextMenuRequested.connect(self._show_list_context_menu)
        list_group_layout.addWidget(self.slice_list_widget)
        list_button_layout = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self.slice_list_widget.selectAll)
        deselect_all_btn = QPushButton("全不选")
        deselect_all_btn.clicked.connect(self.slice_list_widget.clearSelection)
        export_selected_btn = QPushButton("导出选中")
        export_selected_btn.clicked.connect(self._export_selected_data)
        list_button_layout.addWidget(select_all_btn)
        list_button_layout.addWidget(deselect_all_btn)
        list_button_layout.addStretch()
        list_button_layout.addWidget(export_selected_btn)
        list_group_layout.addLayout(list_button_layout)
        left_layout.addWidget(list_group)
        self.splitter.addWidget(left_panel)

    def setup_center_panel(self):
        # ... (Stacked widget setup remains the same) ...
        center_panel = QWidget();
        center_layout = QVBoxLayout(center_panel);
        center_layout.setContentsMargins(0, 0, 0, 0)
        self.center_stacked_widget = QStackedWidget();
        center_layout.addWidget(self.center_stacked_widget)
        # Page 0: 3D Plotter
        self.plotter_widget = QWidget();
        plotter_layout = QVBoxLayout(self.plotter_widget);
        plotter_layout.setContentsMargins(0, 0, 0, 0)
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow setup_ui - Creating QtInteractor...")
        try:
            self.plotter = QtInteractor(parent=self.plotter_widget);
            plotter_layout.addWidget(self.plotter)
            QTimer.singleShot(200, self._initialize_plotter_view)
        except Exception as e:
            print(f"ERROR: Failed to create QtInteractor: {e}"); self.plotter = None; error_label = QLabel(
                f"无法初始化3D视图。\n错误: {e}"); error_label.setAlignment(
                Qt.AlignmentFlag.AlignCenter); plotter_layout.addWidget(error_label)
        self.center_stacked_widget.addWidget(self.plotter_widget)
        # Page 1: 2D Density View
        self.density_view_label = QLabel("请先计算密度图");
        self.density_view_label.setAlignment(Qt.AlignmentFlag.AlignCenter);
        self.density_view_label.setScaledContents(False)
        self.center_stacked_widget.addWidget(self.density_view_label)
        self.splitter.addWidget(center_panel)

    def setup_right_panel(self):
        # ... (Tab widget setup remains the same) ...
        right_panel = QWidget();
        right_panel.setMinimumWidth(300);
        right_panel.setMaximumWidth(450);
        right_layout = QVBoxLayout(right_panel);
        right_layout.setContentsMargins(5, 5, 5, 5)
        self.right_tab_widget = QTabWidget();
        right_layout.addWidget(self.right_tab_widget)
        view_control_tab = QWidget();
        vc_layout = QVBoxLayout(view_control_tab);
        self.setup_view_control_tab(vc_layout);
        self.right_tab_widget.addTab(view_control_tab, "视图控制")
        density_tab = QWidget();
        density_layout = QVBoxLayout(density_tab);
        self.setup_density_analysis_tab(density_layout);
        self.right_tab_widget.addTab(density_tab, "密度分析")
        self.right_tab_widget.currentChanged.connect(self._handle_tab_change)
        self.splitter.addWidget(right_panel)

    def setup_view_control_tab(self, layout):
        # ... (Implementation remains the same) ...
        slicing_group = QGroupBox("切片参数")
        slicing_layout = QVBoxLayout(slicing_group)
        num_slices_layout = QHBoxLayout()
        num_slices_layout.addWidget(QLabel("切片数量:"))
        self.num_slices_spin = QSpinBox()
        self.num_slices_spin.setRange(1, 500)
        self.num_slices_spin.setValue(10)
        num_slices_layout.addWidget(self.num_slices_spin)
        slicing_layout.addLayout(num_slices_layout)
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
        slicing_layout.addWidget(self.limit_thickness_check)
        layout.addWidget(slicing_group)
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
        generate_slices_btn.clicked.connect(self._start_slice_processing)
        action_layout.addWidget(generate_slices_btn)
        export_all_btn = QPushButton("导出所有数据")
        export_all_btn.clicked.connect(self._export_all_data)
        action_layout.addWidget(export_all_btn)
        layout.addWidget(action_group)
        layout.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

    def setup_density_analysis_tab(self, layout):
        """Populates the 'Density Analysis' tab with refined controls."""
        # Density Calculation Group
        density_calc_group = QGroupBox("密度计算")
        dcg_layout = QFormLayout(density_calc_group)
        self.density_resolution_combo = QComboBox()
        self.density_resolution_combo.addItems(["256x256", "512x512", "1024x1024", "2048x2048"])
        self.density_resolution_combo.setCurrentText(
            f"{self.DEFAULT_DENSITY_RESOLUTION}x{self.DEFAULT_DENSITY_RESOLUTION}")
        dcg_layout.addRow("密度网格分辨率:", self.density_resolution_combo)
        self.density_colormap_combo = QComboBox()
        self.density_colormap_combo.addItems(self.AVAILABLE_COLORMAPS)
        self.density_colormap_combo.setCurrentText("viridis")
        dcg_layout.addRow("颜色映射 (主):", self.density_colormap_combo)
        # --- NEW: Diff Colormap ---
        self.diff_colormap_combo = QComboBox()
        self.diff_colormap_combo.addItems(self.DIVERGING_COLORMAPS)
        self.diff_colormap_combo.setCurrentText("RdBu")
        dcg_layout.addRow("颜色映射 (差分):", self.diff_colormap_combo)
        # --- End NEW ---
        self.update_density_btn = QPushButton("计算/更新密度图")
        self.update_density_btn.clicked.connect(self._start_density_processing)
        dcg_layout.addRow(self.update_density_btn)
        layout.addWidget(density_calc_group)

        # Display Control Group
        display_group = QGroupBox("显示控制")
        dg_layout = QFormLayout(display_group)
        self.density_display_combo = QComboBox()
        # Add modes dynamically? Or fixed for now.
        self.density_display_combo.addItems(["选中切片", "单次运算结果", "批量运算结果"])
        self.density_display_combo.setEnabled(False)
        self.density_display_combo.currentIndexChanged.connect(self._update_center_view)
        dg_layout.addRow("显示内容:", self.density_display_combo)
        layout.addWidget(display_group)

        # Single Logic Operation Group
        logic_op_group = QGroupBox("单次逻辑运算")
        log_layout = QVBoxLayout(logic_op_group)
        slice_selection_layout = QHBoxLayout()
        fm = QFontMetrics(self.font())  # For text width calculation
        max_label_width = fm.horizontalAdvance("A: 999") + 10  # Calculate width needed for label
        self.slice_a_label = QLabel("A: 未选")
        self.slice_a_label.setMinimumWidth(max_label_width)
        clear_a_btn = QPushButton("清除A")
        clear_a_btn.clicked.connect(lambda: self._set_logic_operand('A', None))
        slice_selection_layout.addWidget(self.slice_a_label)
        slice_selection_layout.addWidget(clear_a_btn)
        log_layout.addLayout(slice_selection_layout)
        slice_b_layout = QHBoxLayout()
        self.slice_b_label = QLabel("B: 未选")
        self.slice_b_label.setMinimumWidth(max_label_width)
        clear_b_btn = QPushButton("清除B")
        clear_b_btn.clicked.connect(lambda: self._set_logic_operand('B', None))
        slice_b_layout.addWidget(self.slice_b_label)
        slice_b_layout.addWidget(clear_b_btn)
        log_layout.addLayout(slice_b_layout)
        op_layout = QHBoxLayout()
        op_layout.addWidget(QLabel("操作:"))
        self.logic_op_combo = QComboBox()
        self.logic_op_combo.addItems(list(self.LOGIC_OPERATIONS.keys()))
        op_layout.addWidget(self.logic_op_combo)
        log_layout.addLayout(op_layout)
        self.compute_logic_op_btn = QPushButton("计算单次运算")
        self.compute_logic_op_btn.clicked.connect(self._compute_logic_operation)
        self.compute_logic_op_btn.setEnabled(False)
        log_layout.addWidget(self.compute_logic_op_btn)
        layout.addWidget(logic_op_group)

        # --- NEW: Batch Logic Operation Group ---
        batch_op_group = QGroupBox("批量逻辑运算")
        bog_layout = QVBoxLayout(batch_op_group)
        batch_op_form_layout = QFormLayout()
        self.batch_op_combo = QComboBox()
        self.batch_op_combo.addItems(list(self.LOGIC_OPERATIONS.keys()))  # Share operations
        batch_op_form_layout.addRow("运算类型:", self.batch_op_combo)
        self.batch_op_k_spin = QSpinBox()
        self.batch_op_k_spin.setRange(1, 100)
        self.batch_op_k_spin.setValue(1)
        batch_op_form_layout.addRow("步长 k:", self.batch_op_k_spin)
        self.batch_op_loop_check = QCheckBox("循环应用")
        self.batch_op_loop_check.setChecked(False)
        batch_op_form_layout.addRow("", self.batch_op_loop_check)
        bog_layout.addLayout(batch_op_form_layout)
        self.execute_batch_op_btn = QPushButton("执行批量运算")
        self.execute_batch_op_btn.clicked.connect(self._start_batch_logic_operation)
        bog_layout.addWidget(self.execute_batch_op_btn)
        # Batch results preview
        self.batch_results_slider = QSlider(Qt.Orientation.Horizontal)
        self.batch_results_slider.setEnabled(False)
        self.batch_results_slider.valueChanged.connect(self._update_batch_preview_display)
        self.batch_results_label = QLabel("结果: - / -")
        self.batch_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.play_pause_btn = QPushButton("播放")
        self.play_pause_btn.setCheckable(True)
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.toggled.connect(self._toggle_play_batch_results)
        preview_layout = QHBoxLayout()
        preview_layout.addWidget(self.batch_results_slider)
        preview_layout.addWidget(self.play_pause_btn)
        bog_layout.addLayout(preview_layout)
        bog_layout.addWidget(self.batch_results_label)
        layout.addWidget(batch_op_group)
        # --- End NEW ---

        layout.addStretch()

    # --- Initialization ---
    def _initialize_plotter_view(self):
        # ... (Remains the same) ...
        if self.plotter is None: return
        try:
            self.plotter.set_background("white")
            self.plotter.add_text("请在右侧面板设置参数并点击“生成切片”", position="upper_left", font_size=12,
                                  name="init_text")
            self.plotter.render()
        except Exception as e:
            print(f"ERROR: Plotter init failed: {e}")

    # --- Processing ---
    def _start_slice_processing(self):
        # ... (Reset logic added previously remains) ...
        if self.plotter is None or self.original_point_cloud is None or self.original_point_cloud.n_points == 0 or (
                self.slice_processing_thread and self.slice_processing_thread.isRunning()): return
        self.slice_list_widget.clear()
        self.current_slices.clear()
        self.slice_metadata.clear()
        self.density_matrices.clear()
        self.density_pixmaps.clear()
        self.density_params.clear()
        self.selected_slice_a = None
        self.selected_slice_b = None
        self._clear_logic_op_results()
        self._clear_batch_op_results()
        self._update_logic_op_ui()
        self._update_batch_op_ui()
        try:
            self.plotter.clear()
            self.plotter.remove_actor("init_text", render=False)
            self.plotter.add_text(
                "正在生成切片...", position="upper_left", font_size=12,
                name="status_text"
            )
            self.plotter.render()
            QApplication.processEvents()
        except Exception as e:
            print(f"ERROR: Failed to clear plotter: {e}")
        num_slices = self.num_slices_spin.value()
        thickness = self.thickness_spin.value()
        limit_thickness = self.limit_thickness_check.isChecked()
        thumbnail_size = self.slice_list_widget.iconSize()
        self.progress_dialog = QProgressDialog("正在处理切片...", "取消", 0, 100, self)
        self.progress_dialog.setWindowTitle("切片处理")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        self.progress_dialog.canceled.connect(self._cancel_processing)
        QTimer.singleShot(50, self.progress_dialog.show)
        self.slice_processing_thread = SliceProcessingThread(self.original_point_cloud, num_slices, thickness,
                                                             limit_thickness, thumbnail_size)
        self.slice_processing_thread.progress.connect(self._update_progress)
        self.slice_processing_thread.slice_ready.connect(self._collect_slice_data)
        self.slice_processing_thread.thumbnail_ready.connect(self._add_thumbnail_item)
        self.slice_processing_thread.finished.connect(self._slice_processing_finished)
        self.slice_processing_thread.start()

    def _start_density_processing(self):
        # ... (Reset logic added previously remains) ...
        if not self.current_slices:
            QMessageBox.warning(self, "无切片", "请先生成切片数据。")
            return
        if self.density_processing_thread and self.density_processing_thread.isRunning():
            QMessageBox.warning(self,"处理中","正在计算密度图...")
            return
        self.density_matrices.clear()
        self.density_pixmaps.clear()
        self.density_params.clear()
        self.logic_op_result_matrix = None
        self.logic_op_result_pixmap = None
        self.selected_slice_a = None
        self.selected_slice_b = None
        self._clear_batch_op_results()
        self._update_logic_op_ui()
        self._update_batch_op_ui()
        try:
            if self.density_display_combo.count() > 0: self.density_display_combo.setCurrentIndex(0)  # Reset to "选中切片"
            self.density_display_combo.setEnabled(False)  # Disable until done
        except Exception as e:
            print(f"Warning: Error resetting density combo: {e}")
        self._update_center_view()
        resolution_text = self.density_resolution_combo.currentText()
        try:
            grid_resolution = int(resolution_text.split('x')[0])
        except:
            grid_resolution = self.DEFAULT_DENSITY_RESOLUTION
        colormap_name = self.density_colormap_combo.currentText()
        overall_xy_bounds = get_overall_xy_bounds(self.current_slices)
        if overall_xy_bounds is None:
            QMessageBox.warning(self, "无有效边界", "无法计算有效XY边界。")
            return
        self.progress_dialog = QProgressDialog("正在计算密度图...", "取消", 0, 100, self)
        self.progress_dialog.setWindowTitle("密度计算")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        self.progress_dialog.canceled.connect(self._cancel_processing)
        QTimer.singleShot(50, self.progress_dialog.show)
        self.density_processing_thread = DensityProcessingThread(self.current_slices, overall_xy_bounds,
                                                                 grid_resolution, colormap_name)
        self.density_processing_thread.progress.connect(self._update_progress)
        self.density_processing_thread.density_map_ready.connect(self._collect_density_data)
        self.density_processing_thread.finished.connect(self._density_processing_finished)
        self.density_processing_thread.start()

    def _start_batch_logic_operation(self):
        """Executes the batch logic operation."""
        if DEBUG_MODE: print("DEBUG: _start_batch_logic_operation called.")
        if not self.density_matrices:
            QMessageBox.warning(self, "无密度数据", "请先计算密度图。")
            return

        operation_str = self.batch_op_combo.currentText()
        k = self.batch_op_k_spin.value()
        loop = self.batch_op_loop_check.isChecked()
        operation_func = self.LOGIC_OPERATIONS.get(operation_str)

        if not operation_func:
            QMessageBox.critical(self, "错误", f"未知的批量操作: {operation_str}")
            return

        # Clear previous batch results
        self._clear_batch_op_results()
        self._update_batch_op_ui()  # Update UI (e.g., disable button during calc)

        sorted_indices = sorted(self.density_matrices.keys())
        num_slices = len(sorted_indices)
        if num_slices < 2 or k <= 0:
            QMessageBox.warning(self, "参数错误", "至少需要两个切片且步长 k 必须大于 0。")
            self._update_batch_op_ui()
            return

        # --- Perform calculation (can be moved to thread if slow) ---
        results_matrices = {}
        calculated_indices = []
        if DEBUG_MODE: print(f"DEBUG: Starting batch op: {operation_str}, k={k}, loop={loop}")
        for i_idx, current_index in enumerate(sorted_indices):
            next_i_idx = i_idx + k
            if next_i_idx < num_slices:
                next_index = sorted_indices[next_i_idx]
            elif loop and num_slices >= k:  # Check num_slices >= k for looping
                # Wrap around using modulo logic correctly
                next_i_idx = next_i_idx % num_slices
                next_index = sorted_indices[next_i_idx]
            else:
                continue  # Not enough elements ahead and not looping

            matrix_curr = self.density_matrices.get(current_index)
            matrix_next = self.density_matrices.get(next_index)

            if matrix_curr is not None and matrix_next is not None and matrix_curr.shape == matrix_next.shape:
                try:
                    if DEBUG_MODE: print(f"DEBUG: Calculating batch op for ({next_index}, {current_index})")
                    result_matrix = operation_func(matrix_next, matrix_curr)  # Apply op(a[i+k], a[i])
                    op_key = (operation_str, next_index, current_index)
                    self.batch_op_results[op_key] = result_matrix
                    calculated_indices.append(op_key)
                except Exception as e:
                    print(f"ERROR: Batch operation failed for indices ({next_index}, {current_index}): {e}")
            elif DEBUG_MODE:
                print(
                    f"DEBUG: Skipping batch op for ({next_index}, {current_index}) due to missing matrix or shape mismatch.")

        if not calculated_indices:
            QMessageBox.warning(self, "无结果", "未能成功计算任何批量运算结果。")
            self._update_batch_op_ui()
            return

        # --- Generate Pixmaps for results (can also be threaded) ---
        # Determine colormap and scaling for results
        result_colormap = self.diff_colormap_combo.currentText() if '差分' in operation_str else self.density_colormap_combo.currentText()
        vmin_res, vmax_res = None, None  # Calculate scaling based on all results
        all_result_values = np.concatenate(
            [m.flatten() for m in self.batch_op_results.values()]) if self.batch_op_results else np.array([0])

        if '差分' in operation_str:
            abs_max = np.max(np.abs(all_result_values)) if all_result_values.size > 0 else 1
            vmin_res, vmax_res = -abs_max, abs_max
        else:  # Binary or other non-difference maps
            vmin_res = np.min(all_result_values) if all_result_values.size > 0 else 0
            vmax_res = np.max(all_result_values) if all_result_values.size > 0 else 1
            if vmin_res == vmax_res: vmax_res = vmin_res + 1  # Avoid zero range

        if DEBUG_MODE: print(
            f"DEBUG: Generating batch result pixmaps. Colormap={result_colormap}, vmin={vmin_res}, vmax={vmax_res}")
        for key, matrix in self.batch_op_results.items():
            self.batch_op_pixmaps[key] = create_density_heatmap(matrix, result_colormap, vmin=vmin_res, vmax=vmax_res)

        # Store params used for this batch op
        self.batch_op_params = {"op_str": operation_str, "k": k, "loop": loop, "indices": calculated_indices,
                                "vmin": vmin_res, "vmax": vmax_res, "colormap": result_colormap}
        if DEBUG_MODE: print(f"DEBUG: Batch operation complete. {len(self.batch_op_results)} results generated.")

        # Update UI for preview
        self._update_batch_op_ui()
        if calculated_indices:
            # Set display mode and show the first result
            self.density_display_combo.setCurrentText("批量运算结果")
            self.batch_results_slider.setValue(0)  # Trigger update via slider's valueChanged
            self._update_batch_preview_display(0)  # Explicitly update for value 0
            self._update_center_view()  # Switch stacked widget if needed

    # --- Data Collection Callbacks ---
    # ... (_update_progress, _cancel_processing, _collect_slice_data, _add_thumbnail_item, _collect_density_data) ...

    def _update_3d_view_presentation(self):
        """
        Updates the 3D view presentation based on current slices and
        presentation parameters (offset, size, color) without regenerating slice data.
        (Implementation should be the same as the one provided in the previous correct response)
        """
        if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation called.")
        if self.plotter is None or self.center_stacked_widget.currentIndex() != 0:
            if DEBUG_MODE: print(
                "DEBUG: _update_3d_view_presentation - Plotter is None or not in 3D view mode, returning.")
            return
        self.center_stacked_widget.setCurrentIndex(0)  # Ensure stack is on plotter page
        try:
            if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - Clearing actors.")
            self.plotter.clear_actors()
        except Exception as e:
            print(f"ERROR: Failed to clear plotter actors in _update_3d_view_presentation: {e}")

        if not self.current_slices:
            if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - No slices exist to display.")
            self.plotter.add_text("无切片可显示。\n请点击“生成切片并预览”", position="upper_left", font_size=12,
                                  name="status_text")
            try:
                self.plotter.render()
            except Exception as render_err:
                print(f"ERROR: Render failed after clearing actors with no slices: {render_err}")
            return

        offset_value = self.offset_spin.value()
        point_size = self.point_size_spin.value()
        use_color = self.use_color_check.isChecked()
        if DEBUG_MODE: print(
            f"DEBUG: _update_3d_view_presentation - Params: offset={offset_value}, point_size={point_size}, use_color={use_color}")

        actors = []
        current_offset = 0.0
        sorted_indices = sorted(self.current_slices.keys())
        all_bounds = []

        if DEBUG_MODE: print(
            f"DEBUG: _update_3d_view_presentation - Re-adding meshes for {len(sorted_indices)} slices...")
        for i in sorted_indices:
            slice_data = self.current_slices.get(i)
            if slice_data is None or slice_data.n_points == 0:
                current_offset += offset_value
                continue
            offset_slice = slice_data.copy(deep=True)
            offset_slice.points[:, 2] += current_offset
            if offset_slice.bounds[0] < offset_slice.bounds[1]: all_bounds.extend(offset_slice.bounds)
            try:
                if 'colors' in offset_slice.point_data and use_color:
                    actor = self.plotter.add_mesh(offset_slice, scalars='colors', rgb=True, point_size=point_size)
                else:
                    actor = self.plotter.add_mesh(offset_slice, color='grey', point_size=point_size)
                if actor:
                    actors.append(actor)
                else:
                    print(f"WARNING: Failed to add actor for slice {i}.")
            except Exception as e:
                print(f"ERROR: Error adding slice {i}: {e}")
            current_offset += offset_value

        if actors:
            if DEBUG_MODE: print(f"DEBUG: _update_3d_view_presentation - {len(actors)} actors added. Resetting camera.")
            try:
                if all_bounds:
                    min_x = min(all_bounds[0::6]) if all_bounds else 0
                    max_x = max(all_bounds[1::6]) if all_bounds else 1
                    min_y = min(all_bounds[2::6]) if all_bounds else 0
                    max_y = max(all_bounds[3::6]) if all_bounds else 1
                    min_z = min(all_bounds[4::6]) if all_bounds else 0
                    max_z = max(all_bounds[5::6]) if all_bounds else 1
                    overall_bounds = [min_x, max_x, min_y, max_y, min_z, max_z]
                    if DEBUG_MODE: print(f"DEBUG: Resetting camera to overall bounds: {overall_bounds}")
                    self.plotter.reset_camera(bounds=overall_bounds)
                else:
                    self.plotter.reset_camera()
                self.plotter.view_vector([1, -1, 0.5], viewup=[0, 0, 1])
                if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - Camera reset.")
            except Exception as e:
                print(f"ERROR: Error resetting camera: {e}")
        elif self.current_slices:
            if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - No actors added.")
            self.plotter.add_text("所有切片均为空。", position="upper_left", font_size=12, name="status_text")
        try:
            if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - Calling plotter.render().")
            self.plotter.render()
            if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - plotter.render() finished.")
        except Exception as e:
            print(f"ERROR: Exception during final plotter.render(): {e}")

    def _update_progress(self, value, message):
        """Update progress dialog."""
        # (Implementation from previous correct response)
        if self.progress_dialog:
            try:
                self.progress_dialog.setValue(value)
                self.progress_dialog.setLabelText(message)
            except RuntimeError:  # Handle case where dialog might have been closed prematurely
                if DEBUG_MODE: print("DEBUG: Progress dialog accessed after deletion.")
                self.progress_dialog = None  # Reset reference

    def _cancel_processing(self):
        """Handle cancellation from progress dialog or close event."""
        # (Implementation from previous correct response)
        if DEBUG_MODE: print("DEBUG: _cancel_processing called.")
        thread_stopped = False
        if self.slice_processing_thread and self.slice_processing_thread.isRunning():
            print("INFO: Canceling slice processing...")
            self.slice_processing_thread.stop()
            thread_stopped = True
        if self.density_processing_thread and self.density_processing_thread.isRunning():
            print("INFO: Canceling density processing...")
            self.density_processing_thread.stop()
            thread_stopped = True
        # Also stop batch op thread if added later
        # if self.batch_op_thread and self.batch_op_thread.isRunning():
        #     print("INFO: Canceling batch operation processing...")
        #     self.batch_op_thread.stop()
        #     thread_stopped = True

        # If cancellation came from progress dialog, it handles itself
        # If cancellation came from closeEvent, we might need to update UI state if thread was stopped
        # if thread_stopped and self.progress_dialog and not self.progress_dialog.wasCanceled():
        # Optionally update UI to reflect cancellation initiated elsewhere
        # pass

    def _collect_slice_data(self, index, slice_data, height_range):
        """Collect slice data from the thread."""
        # (Implementation from previous correct response)
        if DEBUG_MODE: print(
            f"DEBUG: _collect_slice_data received for index {index}. Data valid: {slice_data is not None}")
        self.current_slices[index] = slice_data
        # Associate height range directly with metadata when thumbnail is ready?
        # Or store temporarily if needed:
        # if not hasattr(self, '_temp_height_ranges'): self._temp_height_ranges = {}
        # self._temp_height_ranges[index] = height_range

    def _add_thumbnail_item(self, index, pixmap, metadata):
        """Add thumbnail item to the list (called from thread)."""
        # (Implementation from previous correct response)
        if DEBUG_MODE: print(
            f"DEBUG: _add_thumbnail_item received for index {index}. Pixmap valid: {not pixmap.isNull()}")
        item = QListWidgetItem(f"Slice {index}")
        item.setIcon(QIcon(pixmap))
        item.setData(Qt.ItemDataRole.UserRole, index)
        self.slice_list_widget.addItem(item)
        self.slice_metadata[index] = metadata  # Metadata now includes height_range passed from thread
        if DEBUG_MODE: print(f"DEBUG: _add_thumbnail_item - Item added for index {index}, metadata stored.")

    def _collect_density_data(self, index, density_matrix, heatmap_pixmap, density_params):
        """Collect density data from the thread."""
        # (Implementation from previous correct response)
        if DEBUG_MODE: print(
            f"DEBUG: _collect_density_data for index {index}. Matrix shape: {density_matrix.shape}, Pixmap valid: {not heatmap_pixmap.isNull()}")
        self.density_matrices[index] = density_matrix
        self.density_pixmaps[index] = heatmap_pixmap
        self.density_params[index] = density_params

    # --- Processing Finished Callbacks ---
    def _slice_processing_finished(self, success):
        # ... (Logic remains mostly the same, ensure UI updates) ...
        if self.progress_dialog:
            try:
                self.progress_dialog.setValue(100)
                self.progress_dialog.close()
                self.progress_dialog = None
            except RuntimeError:
                self.progress_dialog = None
        self.slice_processing_thread = None
        if self.plotter:
            try:
                self.plotter.remove_actor("status_text", render=False)
            except Exception:
                pass
        if success:
            print(f"INFO: Successfully processed {len(self.current_slices)} slices.")
            self._update_3d_view_presentation()
            self.update_density_btn.setEnabled(True)  # Enable density calc
        else:
            (
                QMessageBox.information if self.progress_dialog and self.progress_dialog.wasCanceled() else QMessageBox.warning)(
                self, "结果", "切片处理取消或失败.")
            self.update_density_btn.setEnabled(False)  # Disable if no slices
        self._update_logic_op_ui()
        self._update_batch_op_ui()  # Update enable states

    def _density_processing_finished(self, success):
        # ... (Logic remains mostly the same, ensure UI updates) ...
        if self.progress_dialog:
            try:
                self.progress_dialog.setValue(100)
                self.progress_dialog.close()
                self.progress_dialog = None
            except RuntimeError:
                self.progress_dialog = None
        self.density_processing_thread = None
        if success:
            print(f"INFO: Processed densities for {len(self.density_matrices)} slices.")
            self.density_display_combo.setEnabled(True)
            first_available_index = next(iter(sorted(self.density_pixmaps.keys())), -1)
            if first_available_index != -1:
                self.current_density_display_mode = "选中切片"
                self.density_display_combo.setCurrentText(self.current_density_display_mode)
                # Ensure list widget has selection or select first item programmatically
                if not self.slice_list_widget.selectedItems() and self.slice_list_widget.count() > 0:
                    self.slice_list_widget.setCurrentRow(
                        first_available_index)  # Select the row corresponding to the index if possible
                else:  # If selection exists or setting row fails, update based on current selection
                    self._on_selection_changed()  # Trigger update based on selection (or default)
            else:
                self.density_view_label.setText("无有效密度图生成")
        else:
            (
                QMessageBox.information if self.progress_dialog and self.progress_dialog.wasCanceled() else QMessageBox.warning
            )(
                self, "结果", "密度计算取消或失败."
            )
            self.density_display_combo.setEnabled(False)
        self._update_logic_op_ui()
        self._update_batch_op_ui()  # Update enable states

    # --- UI Update and Interaction ---
    def _handle_tab_change(self, index):
        # ... (Logic remains the same) ...
        if index == 0:
            self.center_stacked_widget.setCurrentIndex(0)
        elif index == 1:
            self.center_stacked_widget.setCurrentIndex(1)
            self._update_density_display_label()

    def _update_center_view(self):
        # ... (Logic remains the same) ...
        current_tab_index = self.right_tab_widget.currentIndex()
        if current_tab_index == 0:
            self.center_stacked_widget.setCurrentIndex(0)  # if self.plotter: try: self.plotter.render() except: pass
        elif current_tab_index == 1:
            self.center_stacked_widget.setCurrentIndex(1)
            self._update_density_display_label()

    def _update_density_display_label(self):
        # ... (Handles "选中切片", "单次运算结果", "批量运算结果") ...
        pixmap_to_show = QPixmap()
        display_mode = self.density_display_combo.currentText()
        label_text = ""
        if display_mode == "选中切片":
            selected_items = self.slice_list_widget.selectedItems()
            target_index = -1
            if selected_items:
                target_index = selected_items[0].data(Qt.ItemDataRole.UserRole)
            elif self.density_pixmaps:
                target_index = next(iter(sorted(self.density_pixmaps.keys())), -1)  # Fallback to first available
            if target_index != -1 and target_index in self.density_pixmaps:
                self.current_density_display_index = target_index
                pixmap_to_show = self.density_pixmaps[target_index]
                label_text = f"切片 {target_index} 密度图"
            else:
                label_text = f"切片密度图不可用 (索引: {target_index})"
        elif display_mode == "单次运算结果":
            if self.logic_op_result_pixmap and not self.logic_op_result_pixmap.isNull():
                pixmap_to_show = self.logic_op_result_pixmap
                label_text = "单次运算结果"
            else:
                label_text = "无单次运算结果"
        elif display_mode == "批量运算结果":
            if self.current_batch_op_display_key in self.batch_op_pixmaps:
                pixmap_to_show = self.batch_op_pixmaps[self.current_batch_op_display_key]
                label_text = f"批量结果: {self.current_batch_op_display_key}"
            else:
                label_text = "无批量运算结果或未选择"
        else:
            label_text = "请先计算密度图"
        if not pixmap_to_show.isNull():
            scaled_pixmap = pixmap_to_show.scaled(self.density_view_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation)
            self.density_view_label.setPixmap(scaled_pixmap)
            self.density_view_label.setText("")
        else:
            self.density_view_label.setText(label_text)
            self.density_view_label.setPixmap(QPixmap())

    def resizeEvent(self, event):
        # ... (Logic remains the same) ...
        super().resizeEvent(event)
        if self.center_stacked_widget.currentIndex() == 1: self._update_density_display_label()

    def _on_selection_changed(self):
        # --- MODIFIED: Only update if in correct display mode ---
        display_mode = self.density_display_combo.currentText()
        if display_mode == "选中切片":
            selected_items = self.slice_list_widget.selectedItems()
            if selected_items:
                first_selected_index = selected_items[0].data(Qt.ItemDataRole.UserRole)
                # Check if density data actually exists for this index before updating
                if first_selected_index in self.density_pixmaps:
                    self.current_density_display_index = first_selected_index
                    self._update_density_display_label()  # Update label content directly
                    if DEBUG_MODE: print(
                        f"DEBUG: List selection changed, density view set to index {first_selected_index}")
                elif DEBUG_MODE:
                    print(f"DEBUG: List selected index {first_selected_index}, but no density pixmap found.")
            # else: # Optional: Handle case where selection is cleared
            #     self.density_view_label.setText("请选择一个切片预览密度")
            #     self.density_view_label.setPixmap(QPixmap())
        # --- End MODIFICATION ---

    # --- Logic Op Methods ---
    def _show_list_context_menu(self, pos: QPoint):
        # ... (Use refined logic with "清除" options in main panel) ...
        item = self.slice_list_widget.itemAt(pos)
        if not item: return
        index = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu()
        action_a = QAction(f"设为逻辑运算切片 A", self)
        action_b = QAction(f"设为逻辑运算切片 B", self)
        action_a.triggered.connect(lambda: self._set_logic_operand('A', index))
        action_b.triggered.connect(lambda: self._set_logic_operand('B', index))
        can_select = index in self.density_matrices  # Check if density data exists
        action_a.setEnabled(can_select)
        action_b.setEnabled(can_select)
        if not can_select: action_a.setText("设为切片 A (无密度数据)")
        action_b.setText("设为切片 B (无密度数据)")
        menu.addAction(action_a)
        menu.addAction(action_b)
        menu.exec(self.slice_list_widget.mapToGlobal(pos))

    def _set_logic_operand(self, operand, index):
        # ... (Implementation remains the same, updates label and UI state) ...
        if operand == 'A':
            self.selected_slice_a = index
        elif operand == 'B':
            self.selected_slice_b = index
        elif index is None:  # Handle clear case
            if operand == 'A':
                self.selected_slice_a = None
            elif operand == 'B':
                self.selected_slice_b = None
        self._update_logic_op_ui()

    def _update_logic_op_ui(self):
        # ... (Update labels and enable compute button) ...
        self.slice_a_label.setText(f"A: {self.selected_slice_a if self.selected_slice_a is not None else '未选'}")
        self.slice_b_label.setText(f"B: {self.selected_slice_b if self.selected_slice_b is not None else '未选'}")
        ready = self.selected_slice_a is not None and self.selected_slice_b is not None \
                and self.selected_slice_a in self.density_matrices \
                and self.selected_slice_b in self.density_matrices
        self.compute_logic_op_btn.setEnabled(ready)

    def _compute_logic_operation(self):
        # ... (Implementation remains the same, uses selected colormap for diff) ...
        if not self.compute_logic_op_btn.isEnabled(): return
        idx_a = self.selected_slice_a
        idx_b = self.selected_slice_b
        operation_str = self.logic_op_combo.currentText()
        matrix_a = self.density_matrices.get(idx_a)
        matrix_b = self.density_matrices.get(idx_b)
        operation_func = self.LOGIC_OPERATIONS.get(operation_str)
        if matrix_a is None or matrix_b is None or operation_func is None: return
        if matrix_a.shape != matrix_b.shape:
            QMessageBox.critical(self, "错误", "密度矩阵形状不匹配。")
            return
        try:
            result_matrix = operation_func(matrix_a, matrix_b)
            result_colormap = self.diff_colormap_combo.currentText() if '差分' in operation_str else self.density_colormap_combo.currentText()
            vmin_res, vmax_res = None, None
            if '差分' in operation_str:
                abs_max = np.max(np.abs(result_matrix)) if result_matrix.size > 0 else 1
                vmin_res, vmax_res = -abs_max, abs_max
            else:
                vmin_res, vmax_res = 0, 1  # Assume binary 0/1 for non-diff logical ops
            self.logic_op_result_matrix = result_matrix
            self.logic_op_result_pixmap = create_density_heatmap(result_matrix, result_colormap, vmin=vmin_res,
                                                                 vmax=vmax_res)
            self.density_display_combo.setCurrentText("单次运算结果")  # Switch view
            self._update_center_view()  # Update display immediately
        except Exception as e:
            print(f"ERROR: Logic op failed: {e}")
            QMessageBox.critical(self, "错误运算失败: {e}")
            self.logic_op_result_matrix = None
            self.logic_op_result_pixmap = None

    def _clear_logic_op_results(self):
        """Clears single logic operation results and selections."""
        self.selected_slice_a = None
        self.selected_slice_b = None
        self.logic_op_result_matrix = None
        self.logic_op_result_pixmap = None
        # Don't switch display mode here, let user control it

    # --- Batch Logic Op Methods ---
    def _clear_batch_op_results(self):
        """Clears batch operation results and resets UI."""
        self.batch_op_results.clear()
        self.batch_op_pixmaps.clear()
        self.batch_op_params.clear()
        self.current_batch_op_display_key = None
        self.play_timer.stop()
        self.play_pause_btn.setChecked(False)
        self.play_pause_btn.setText("播放")

    def _update_batch_op_ui(self):
        """Updates the enable state and range of batch op controls."""
        has_results = bool(self.batch_op_results)
        self.batch_results_slider.setEnabled(has_results)
        self.play_pause_btn.setEnabled(has_results)

        if has_results:
            num_results = len(self.batch_op_results)
            self.batch_results_slider.setRange(0, max(0, num_results - 1))  # Range is 0 to N-1
            self.batch_results_label.setText(f"结果: {self.batch_results_slider.value() + 1} / {num_results}")
        else:
            self.batch_results_slider.setRange(0, 0)
            self.batch_results_label.setText("结果: - / -")
            self.play_timer.stop()  # Ensure timer stops if results cleared
            self.play_pause_btn.setChecked(False)
            self.play_pause_btn.setText("播放")

        # Enable batch execution button only if density data exists
        self.execute_batch_op_btn.setEnabled(bool(self.density_matrices))

    def _update_batch_preview_display(self, slider_index):
        """Updates the central display based on the batch result slider."""
        if not self.batch_op_params or not self.batch_op_pixmaps: return

        indices = self.batch_op_params.get("indices", [])
        if 0 <= slider_index < len(indices):
            key = indices[slider_index]
            self.current_batch_op_display_key = key
            num_results = len(indices)
            self.batch_results_label.setText(
                f"结果: {slider_index + 1} / {num_results} ({key[1]} vs {key[2]})")  # Show indices used

            # Only update the central view if the batch mode is selected
            if self.density_display_combo.currentText() == "批量运算结果":
                self._update_density_display_label()
        else:
            # Handle out of bounds index if needed
            self.current_batch_op_display_key = None
            self.batch_results_label.setText(f"结果: - / {len(indices)}")

    def _toggle_play_batch_results(self, checked):
        """Starts or stops the playback timer for batch results."""
        if checked:  # Play button pressed
            if self.batch_op_results:
                self.play_pause_btn.setText("暂停")
                self.play_timer.start(500)  # Update every 500ms (adjust as needed)
            else:
                self.play_pause_btn.setChecked(False)  # Cannot play without results
        else:  # Pause button pressed or toggled off
            self.play_pause_btn.setText("播放")
            self.play_timer.stop()

    def _play_next_batch_result(self):
        """Advances the batch result slider for playback."""
        if not self.batch_results_slider.isEnabled():
            self.play_timer.stop()
            self.play_pause_btn.setChecked(False)
            return

        current_value = self.batch_results_slider.value()
        max_value = self.batch_results_slider.maximum()

        next_value = current_value + 1
        if next_value > max_value:
            next_value = 0  # Loop back to start

        self.batch_results_slider.setValue(next_value)

    # --- Export Methods (Combined) ---
    def _export_bitmaps(self, indices_to_export):  # Consider renaming to _export_data
        # ... (Implementation remains mostly the same, but now saves PCD and Density Matrix/Heatmap) ...
        if DEBUG_MODE: print(f"DEBUG: _export_data called for indices: {indices_to_export}")
        if not indices_to_export: return
        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not export_dir: return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_export_path = os.path.join(export_dir, f"batch_slice_export_{timestamp}")
        os.makedirs(base_export_path, exist_ok=True)
        global_params = {
            "export_time": datetime.datetime.now().isoformat(),
            "original_point_cloud_source": self.source_filename,
            "num_slices_param": self.num_slices_spin.value(),
            "slice_thickness_param": self.thickness_spin.value(),
            "limit_thickness": self.limit_thickness_check.isChecked()
        }
        global_params_file = os.path.join(base_export_path, "export_parameters.json")
        with open(global_params_file, 'w', encoding='utf-8') as f:
            json.dump(global_params, f, ensure_ascii=False, indent=2)
        overall_xy_bounds = get_overall_xy_bounds(self.current_slices)
        export_progress = QProgressDialog("正在导出数据...", "取消", 0, len(indices_to_export), self)
        export_progress.setWindowTitle("导出进度")
        export_progress.setWindowModality(Qt.WindowModality.WindowModal)
        export_progress.setAutoClose(True)
        export_progress.setAutoReset(True)
        export_progress.show()
        exported_count = 0
        exported_pcd_count = 0
        exported_density_count = 0
        exported_bitmap_count = 0
        for i, index in enumerate(indices_to_export):
            export_progress.setValue(i)
            if export_progress.wasCanceled():
                print("INFO: Export canceled.")
                break
            slice_data_pv = self.current_slices.get(index)
            metadata = self.slice_metadata.get(index)
            density_matrix = self.density_matrices.get(index)
            density_pixmap = self.density_pixmaps.get(index)
            density_params_saved = self.density_params.get(index)
            if metadata is None: print(f"WARNING: Meta missing {index}."); continue
            img_np = None
            view_params_render = None
            render_error_msg = None
            bitmap_saved = False
            pcd_saved = False
            density_saved = False
            # Render Bitmap
            if not metadata.get("is_empty", False) and slice_data_pv is not None:
                export_progress.setLabelText(f"渲染位图 {index}...")
                QApplication.processEvents()
                try:
                    img_np, view_params_render = render_slice_to_image(slice_data_pv, self.BITMAP_EXPORT_RESOLUTION,
                                                                       overall_xy_bounds, False)
                except Exception as r_err:
                    render_error_msg = str(r_err)
                if img_np is None and not render_error_msg: render_error_msg = "Bitmap rendering failed"
            # Save Metadata
            meta_filename = os.path.join(base_export_path, f"slice_{index}_metadata.json")
            metadata["view_params_render"] = view_params_render if img_np is not None else None
            if render_error_msg: metadata["render_error"] = render_error_msg
            if density_params_saved: metadata["density_params"] = density_params_saved
            export_data = {"slice_index": index, "metadata": metadata}
            try:
                with open(meta_filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            except Exception as meta_err:
                print(f"ERROR saving meta {index}: {meta_err}")
            # Save Bitmap
            if img_np is not None:
                try:
                    bitmap_filename = os.path.join(base_export_path, f"slice_{index}_bitmap.png");
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    if cv2.imwrite(bitmap_filename, img_bgr):
                        bitmap_saved = True
                        exported_bitmap_count += 1
                except Exception as bm_err:
                    print(f"ERROR saving bitmap {index}: {bm_err}")
            # Save PCD
            if slice_data_pv is not None and slice_data_pv.n_points > 0:
                pcd_filename = os.path.join(base_export_path, f"slice_{index}.pcd");
                try:
                    export_progress.setLabelText(f"保存 PCD {index}...")
                    QApplication.processEvents()
                    points = slice_data_pv.points
                    o3d_pcd = o3d.geometry.PointCloud()
                    o3d_pcd.points = o3d.utility.Vector3dVector(points)
                    if 'colors' in slice_data_pv.point_data:
                        colors = slice_data_pv['colors']
                        o3d_pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
                    if o3d.io.write_point_cloud(pcd_filename, o3d_pcd, False, True):
                        pcd_saved = True
                        exported_pcd_count += 1
                except Exception as pcd_err:
                    print(f"ERROR saving PCD {index}: {pcd_err}")
            elif metadata.get("is_empty", False):
                pcd_saved = True
            # Save Density
            if density_matrix is not None:
                try:
                    export_progress.setLabelText(f"保存密度 {index}...")
                    QApplication.processEvents()
                    matrix_filename = os.path.join(base_export_path, f"slice_{index}_density_matrix.npy")
                    np.save(matrix_filename, density_matrix)
                    heatmap_filename = os.path.join(base_export_path, f"slice_{index}_density_heatmap.png")
                    if density_pixmap and not density_pixmap.isNull():
                        density_pixmap.save(heatmap_filename, "PNG")
                        density_saved = True;
                        exported_density_count += 1
                except Exception as den_err:
                    print(f"ERROR saving density {index}: {den_err}")
            elif index in self.density_matrices:
                density_saved = True  # Attempted
            if bitmap_saved or pcd_saved or density_saved or metadata.get("is_empty", False): exported_count += 1
        export_progress.setValue(len(indices_to_export))
        QMessageBox.information(self, "导出完成",
                                f"处理完成 {len(indices_to_export)} 项。\n成功导出 {exported_count} 个切片的各类数据。\n(位图: {exported_bitmap_count}, PCD: {exported_pcd_count}, 密度: {exported_density_count})\n保存在:\n{base_export_path}")

    def _export_selected_data(self):  # Renamed
        selected_items = self.slice_list_widget.selectedItems()
        if not selected_items: QMessageBox.warning(self, "未选择", "请先选择要导出的项。"); return
        indices_to_export = sorted([item.data(Qt.ItemDataRole.UserRole) for item in selected_items])
        self._export_bitmaps(indices_to_export)  # Call combined export

    def _export_all_data(self):
        if not self.slice_metadata: QMessageBox.warning(self, "无数据", "请先生成切片。"); return
        indices_to_export = sorted(list(self.slice_metadata.keys()))
        self._export_bitmaps(indices_to_export)

    # --- Close Event ---
    def closeEvent(self, event):
        # ... (Remains the same) ...
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow closeEvent called.")
        self._cancel_processing()
        if (self.slice_processing_thread and self.slice_processing_thread.isRunning()) or \
                (self.density_processing_thread and self.density_processing_thread.isRunning()):
            print("INFO: Waiting for processing thread(s) to finish before closing...")
            if self.slice_processing_thread: self.slice_processing_thread.wait(1500)
            if self.density_processing_thread: self.density_processing_thread.wait(1500)
        if self.plotter:
            if DEBUG_MODE: print("DEBUG: Closing plotter in closeEvent.")
            try:
                self.plotter.close()
            except Exception as e:
                print(f"ERROR: Exception while closing plotter in closeEvent: {e}")
        super().closeEvent(event)
