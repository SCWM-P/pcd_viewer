# pcd_viewer/ui/batch_slice_viewer_window.py

import os
import json
import datetime
import numpy as np
import pyvista as pv
import cv2
import traceback
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QListWidget,
                             QListWidgetItem, QPushButton, QSplitter, QGroupBox,
                             QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
                             QMessageBox, QAbstractItemView, QProgressBar, QSpacerItem,
                             QSizePolicy, QProgressDialog, QApplication)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QIcon, QPainter
from pyvistaqt import QtInteractor

# 导入项目模块
from ..utils.point_cloud_handler import PointCloudHandler
from ..utils.stylesheet_manager import StylesheetManager

DEBUG_MODE = False

# --- Thumbnail/Bitmap Rendering Helper ---
def render_slice_to_image(slice_data, size, is_thumbnail=True):
    """
    Renders a single slice to a NumPy image array using an off-screen plotter.
    """
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
        if is_thumbnail:
            actor = plotter.add_mesh(slice_data, color='darkgrey', point_size=1)
        else:
            if 'colors' in slice_data.point_data:
                actor = plotter.add_mesh(slice_data, scalars='colors', rgb=True, point_size=2)
            else:
                actor = plotter.add_mesh(slice_data, color='blue', point_size=2)
        if DEBUG_MODE: print(f"DEBUG: render_slice_to_image - Mesh added: {actor is not None}")

        if DEBUG_MODE: print("DEBUG: render_slice_to_image - Setting view_xy()")
        plotter.view_xy()
        if DEBUG_MODE: print(f"DEBUG: render_slice_to_image - Resetting camera to bounds: {slice_data.bounds}")
        if slice_data.bounds[0] < slice_data.bounds[1]:
            plotter.reset_camera(bounds=slice_data.bounds)
            if DEBUG_MODE: print("DEBUG: render_slice_to_image - Camera reset done.")
        else:
            if DEBUG_MODE: print("DEBUG: render_slice_to_image - Invalid bounds, skipping camera reset.")

        if DEBUG_MODE: print("DEBUG: render_slice_to_image - Taking screenshot...")
        img_np = plotter.screenshot(return_img=True)
        if DEBUG_MODE: print(f"DEBUG: render_slice_to_image - Screenshot taken, shape: {img_np.shape if img_np is not None else 'None'}")

        cam = plotter.camera
        if DEBUG_MODE: print(f"DEBUG: render_slice_to_image - Getting camera parameters. Camera object: {cam}")
        view_params = {
            "position": list(cam.position),
            "focal_point": list(cam.focal_point),
            "up": list(cam.up),
            "parallel_projection": cam.parallel_projection,
            "parallel_scale": cam.parallel_scale,
            "slice_bounds": list(slice_data.bounds),
            "render_window_size": [img_width, img_height],
        }
        if DEBUG_MODE: print(f"DEBUG: render_slice_to_image - View params collected: {view_params}")

        return img_np, view_params

    except Exception as e:
        print(f"ERROR: Error rendering slice to image: {e}")
        if DEBUG_MODE: traceback.print_exc()
        return None, {}
    finally:
        if plotter:
            if DEBUG_MODE: print("DEBUG: render_slice_to_image - Closing plotter.")
            try:
                plotter.close()
            except Exception as close_e:
                 print(f"ERROR: Exception while closing plotter in render_slice_to_image: {close_e}")
        if DEBUG_MODE: print("DEBUG: render_slice_to_image finished.")


# --- Background Thread for Processing ---
class SliceProcessingThread(QThread):
    # ... (signals remain the same) ...
    progress = pyqtSignal(int, str)
    slice_ready = pyqtSignal(int, object, tuple)
    thumbnail_ready = pyqtSignal(int, QPixmap, dict)
    finished = pyqtSignal(bool)

    # ... (__init__ remains the same) ...
    def __init__(self, point_cloud, num_slices, thickness, thumbnail_size, parent=None):
        super().__init__(parent)
        self.point_cloud = point_cloud
        self.num_slices = num_slices
        self.thickness = thickness
        self.thumbnail_size = thumbnail_size
        self._is_running = True
        if DEBUG_MODE: print(f"DEBUG: SliceProcessingThread initialized. num_slices={num_slices}, thickness={thickness}")


    def run(self):
        # ... (run method setup and slicing phase remain the same) ...
        if DEBUG_MODE: print("DEBUG: SliceProcessingThread run started.")
        if self.point_cloud is None or self.num_slices <= 0 or self.thickness <= 0:
            if DEBUG_MODE: print("DEBUG: SliceProcessingThread run - Invalid parameters or point cloud.")
            self.finished.emit(False)
            return

        try:
            bounds = self.point_cloud.bounds
            min_z, max_z = bounds[4], bounds[5]
            total_height = max_z - min_z
            if DEBUG_MODE: print(f"DEBUG: SliceProcessingThread - Point cloud bounds Z: {min_z} to {max_z}, total_height={total_height}")

            if total_height <= 0:
                if DEBUG_MODE: print("DEBUG: SliceProcessingThread - Point cloud total height is zero or negative.")
                self.finished.emit(False)
                return

            all_points = self.point_cloud.points
            has_colors = 'colors' in self.point_cloud.point_data
            if has_colors:
                all_colors = self.point_cloud['colors']

            step = total_height / self.num_slices
            current_start_z = min_z
            total_steps = self.num_slices * 2 # Slicing + Thumbnail generation
            if DEBUG_MODE: print(f"DEBUG: SliceProcessingThread - Calculated step={step}")

            # --- Slicing Phase ---
            generated_slices = []
            height_ranges = []
            if DEBUG_MODE: print("DEBUG: SliceProcessingThread - Starting slicing phase...")
            for i in range(self.num_slices):
                if not self._is_running: raise InterruptedError("Processing stopped by user request.")
                self.progress.emit(int((i + 1) / total_steps * 100), f"正在生成切片 {i+1}/{self.num_slices}")
                if DEBUG_MODE: print(f"DEBUG: SliceProcessingThread - Generating slice {i}...")

                slice_start_z = current_start_z # Slice from the start of the interval
                slice_end_z = slice_start_z + self.thickness
                slice_end_z = min(slice_end_z, max_z + 1e-6) # Add small tolerance
                slice_start_z = min(slice_start_z, slice_end_z) # Ensure start <= end

                if DEBUG_MODE: print(f"DEBUG: SliceProcessingThread - Slice {i} Z range: {slice_start_z:.4f} - {slice_end_z:.4f}")

                indices = np.where((all_points[:, 2] >= slice_start_z) & (all_points[:, 2] <= slice_end_z))[0]
                height_ranges.append((slice_start_z, slice_end_z))
                if DEBUG_MODE: print(f"DEBUG: SliceProcessingThread - Slice {i} found {len(indices)} points.")

                if len(indices) > 0:
                    slice_points = all_points[indices]
                    slice_cloud = pv.PolyData(slice_points)
                    if has_colors:
                        slice_cloud['colors'] = all_colors[indices]
                    generated_slices.append(slice_cloud)
                    self.slice_ready.emit(i, slice_cloud, (slice_start_z, slice_end_z))
                else:
                    generated_slices.append(None) # Keep placeholder
                    self.slice_ready.emit(i, None, (slice_start_z, slice_end_z))

                current_start_z += step # Move start for the next interval

            # --- Thumbnail Generation Phase ---
            if DEBUG_MODE: print("DEBUG: SliceProcessingThread - Starting thumbnail generation phase...")
            for i in range(self.num_slices):
                if not self._is_running: raise InterruptedError("Processing stopped by user request.")
                self.progress.emit(int((self.num_slices + i + 1) / total_steps * 100), f"生成缩略图 {i+1}/{self.num_slices}")
                if DEBUG_MODE: print(f"DEBUG: SliceProcessingThread - Generating thumbnail {i}...")

                slice_data = generated_slices[i]
                img_np, view_params = render_slice_to_image(slice_data, self.thumbnail_size, is_thumbnail=True)
                if DEBUG_MODE: print(f"DEBUG: SliceProcessingThread - Thumbnail {i} rendered. Image valid: {img_np is not None}")

                metadata = {
                    "index": i,
                    "height_range": height_ranges[i],
                    "view_params": view_params,
                    "is_empty": slice_data is None or slice_data.n_points == 0
                }

                if img_np is not None:
                    try:
                        h, w, ch = img_np.shape
                        image_data_bytes = img_np.tobytes()
                        q_img = QImage(image_data_bytes, w, h, w * ch, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_img)
                        if pixmap.isNull():
                             print(f"WARNING: SliceProcessingThread - QPixmap created from QImage is null for thumbnail {i}.")
                             raise ValueError("Created QPixmap is null")
                        scaled_pixmap = pixmap.scaled(self.thumbnail_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        self.thumbnail_ready.emit(i, scaled_pixmap, metadata)
                        if DEBUG_MODE: print(f"DEBUG: SliceProcessingThread - Thumbnail {i} QPixmap emitted.")
                    except Exception as qimage_err:
                         print(f"ERROR: SliceProcessingThread - Failed to create QImage/QPixmap for thumbnail {i}: {qimage_err}")
                         if DEBUG_MODE: traceback.print_exc() # Print stack trace for QImage error
                         placeholder_pixmap = QPixmap(self.thumbnail_size); placeholder_pixmap.fill(Qt.GlobalColor.darkRed)
                         self.thumbnail_ready.emit(i, placeholder_pixmap, metadata) # Emit error placeholder

                else:
                    # Create placeholder for empty/failed thumbnail render
                    if DEBUG_MODE: print(f"DEBUG: SliceProcessingThread - Creating placeholder thumbnail for slice {i}.")
                    placeholder_pixmap = QPixmap(self.thumbnail_size)
                    placeholder_pixmap.fill(Qt.GlobalColor.lightGray)
                    painter = QPainter(placeholder_pixmap)
                    painter.drawText(placeholder_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, f"Slice {i}\n(Empty)")
                    painter.end()
                    self.thumbnail_ready.emit(i, placeholder_pixmap, metadata)

            if DEBUG_MODE: print("DEBUG: SliceProcessingThread - Processing finished successfully.")
            self.finished.emit(True)

        except InterruptedError:
             print("INFO: Slice processing thread stopped by user.")
             self.finished.emit(False)
        except Exception as e:
            print(f"ERROR: Unhandled error during slice processing thread: {e}")
            if DEBUG_MODE: traceback.print_exc()
            self.finished.emit(False)

    # ... (stop method remains the same) ...
    def stop(self):
        if DEBUG_MODE: print("DEBUG: SliceProcessingThread stop requested.")
        self._is_running = False


# --- Main Window Class ---
class BatchSliceViewerWindow(QWidget):
    # ... (BITMAP_EXPORT_RESOLUTION remains the same) ...
    BITMAP_EXPORT_RESOLUTION = (1024, 1024)

    def __init__(self, point_cloud, source_filename="Unknown", parent=None):
        # Note: parent argument is kept here but not used when called from main_window
        super().__init__(parent) # Pass parent to QWidget constructor
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow __init__ started.")
        self.setWindowTitle("批量切片查看器")
        self.setMinimumSize(1000, 700)
        self.setWindowFlags(Qt.WindowType.Window)

        # --- FIX: Apply Stylesheet ---
        self.setStyleSheet(StylesheetManager.get_light_theme())
        # --- End FIX ---

        self.original_point_cloud = point_cloud
        self.source_filename = source_filename
        self.current_slices = {}
        self.slice_metadata = {}
        self.processing_thread = None
        self.progress_dialog = None
        self.plotter = None

        self.setup_ui()
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow __init__ finished.")

    def setup_ui(self):
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow setup_ui started.")
        main_layout = QHBoxLayout(self)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)

        # --- Left Panel (Thumbnails) ---
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
        list_group_layout.addWidget(self.slice_list_widget)
        list_button_layout = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self.slice_list_widget.selectAll)
        deselect_all_btn = QPushButton("全不选")
        deselect_all_btn.clicked.connect(self.slice_list_widget.clearSelection)
        export_selected_bitmaps_btn = QPushButton("导出选中位图")
        export_selected_bitmaps_btn.clicked.connect(self._export_selected_bitmaps)
        list_button_layout.addWidget(select_all_btn)
        list_button_layout.addWidget(deselect_all_btn)
        list_button_layout.addStretch()
        list_button_layout.addWidget(export_selected_bitmaps_btn)
        list_group_layout.addLayout(list_button_layout)
        left_layout.addWidget(list_group)
        self.splitter.addWidget(left_panel)

        # --- Center Panel (3D View) ---
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow setup_ui - Creating QtInteractor...")
        try:
            self.plotter = QtInteractor(parent=center_panel) # Explicitly set parent
            if DEBUG_MODE: print(f"DEBUG: BatchSliceViewerWindow setup_ui - QtInteractor created: {self.plotter}")
            center_layout.addWidget(self.plotter)
            QTimer.singleShot(200, self._initialize_plotter_view) # Increased delay slightly
        except Exception as e:
             print(f"ERROR: Failed to create QtInteractor: {e}")
             if DEBUG_MODE: traceback.print_exc()
             error_label = QLabel(f"无法初始化3D视图。\n请检查图形驱动和依赖项。\n错误: {e}")
             error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
             error_label.setWordWrap(True)
             center_layout.addWidget(error_label)
             self.plotter = None

        self.splitter.addWidget(center_panel)

        # --- Right Panel (Parameters & Controls) ---
        right_panel = QWidget()
        right_panel.setMinimumWidth(250)
        right_panel.setMaximumWidth(350)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
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
        right_layout.addWidget(slicing_group)
        # Visualization Parameters Group
        viz_group = QGroupBox("3D视图参数")
        viz_layout = QVBoxLayout(viz_group)
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("切片间垂直偏移:"))
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(0.0, 10.0); self.offset_spin.setSingleStep(0.1)
        self.offset_spin.setValue(0.5)
        self.offset_spin.valueChanged.connect(self._update_3d_view_presentation) # Only updates appearance
        offset_layout.addWidget(self.offset_spin)
        viz_layout.addLayout(offset_layout)
        point_size_layout = QHBoxLayout()
        point_size_layout.addWidget(QLabel("点大小:"))
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 10); self.point_size_spin.setValue(2)
        self.point_size_spin.valueChanged.connect(self._update_3d_view_presentation) # Only updates appearance
        point_size_layout.addWidget(self.point_size_spin)
        viz_layout.addLayout(point_size_layout)
        self.use_color_check = QCheckBox("显示原始颜色")
        self.use_color_check.setChecked(True)
        self.use_color_check.stateChanged.connect(self._update_3d_view_presentation) # Only updates appearance
        viz_layout.addWidget(self.use_color_check)
        right_layout.addWidget(viz_group)
        # Action Buttons Group
        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout(action_group)
        generate_slices_btn = QPushButton("生成切片并预览") # Changed Label
        generate_slices_btn.setToolTip("根据当前参数生成切片数据、缩略图并更新3D视图")
        generate_slices_btn.clicked.connect(self._start_slice_processing) # Connect to start method
        action_layout.addWidget(generate_slices_btn)
        export_all_btn = QPushButton("导出所有数据")
        export_all_btn.setToolTip("将所有切片的位图、元数据和参数导出")
        export_all_btn.clicked.connect(self._export_all_data)
        action_layout.addWidget(export_all_btn)
        right_layout.addWidget(action_group)
        right_layout.addStretch()
        right_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        right_layout.addWidget(close_btn)
        self.splitter.addWidget(right_panel)

        # --- Splitter Setup ---
        self.splitter.setSizes([250, 600, 250])
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow setup_ui finished.")

    def _initialize_plotter_view(self):
        """Initialize the plotter view appearance after a short delay."""
        if self.plotter is None:
            if DEBUG_MODE: print("DEBUG: _initialize_plotter_view - Plotter is None, cannot initialize.")
            return
        if DEBUG_MODE: print("DEBUG: _initialize_plotter_view called.")
        try:
            if DEBUG_MODE: print("DEBUG: _initialize_plotter_view - Setting background...")
            self.plotter.set_background("white") # Try setting background here
            if DEBUG_MODE: print("DEBUG: _initialize_plotter_view - Adding initial text...")
            self.plotter.add_text("请在右侧面板设置参数并点击“生成切片”", position="upper_left", font_size=12, name="init_text") # Give text actor a name
            if DEBUG_MODE: print("DEBUG: _initialize_plotter_view - Calling render()...")
            self.plotter.render()
            if DEBUG_MODE: print("DEBUG: _initialize_plotter_view - Initialization render finished.")
        except Exception as e:
            print(f"ERROR: Failed during plotter initialization: {e}")
            if DEBUG_MODE: traceback.print_exc()
            try:
                 self.plotter.add_text(f"渲染初始化错误:\n{e}", position='center', color='red', font_size=10)
                 self.plotter.render()
            except:
                 pass

    def _start_slice_processing(self):
        if DEBUG_MODE: print("DEBUG: _start_slice_processing called.")
        if self.plotter is None:
             QMessageBox.critical(self, "错误", "3D视图未能成功初始化，无法进行切片处理。")
             return

        if self.original_point_cloud is None or self.original_point_cloud.n_points == 0:
            QMessageBox.warning(self, "无数据", "没有有效的点云数据可供切片。请先在主窗口加载点云。")
            return

        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.warning(self, "处理中", "当前正在进行切片处理，请稍候或取消。")
            return

        if DEBUG_MODE: print("DEBUG: _start_slice_processing - Clearing previous results.")
        self.slice_list_widget.clear()
        self.current_slices.clear()
        self.slice_metadata.clear()
        try:
            self.plotter.clear()
            self.plotter.remove_actor("init_text", render=False)
            self.plotter.add_text("正在生成切片...", position="upper_left", font_size=12, name="status_text")
            self.plotter.render()
            QApplication.processEvents()
        except Exception as e:
             print(f"ERROR: Failed to clear plotter or add text: {e}")


        num_slices = self.num_slices_spin.value()
        thickness = self.thickness_spin.value()
        thumbnail_size = self.slice_list_widget.iconSize()

        if DEBUG_MODE: print("DEBUG: _start_slice_processing - Setting up progress dialog.")
        self.progress_dialog = QProgressDialog("正在处理切片...", "取消", 0, 100, self)
        self.progress_dialog.setWindowTitle("切片处理")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        self.progress_dialog.canceled.connect(self._cancel_processing)
        QTimer.singleShot(50, self.progress_dialog.show)


        if DEBUG_MODE: print("DEBUG: _start_slice_processing - Starting SliceProcessingThread.")
        self.processing_thread = SliceProcessingThread(
            self.original_point_cloud, num_slices, thickness, thumbnail_size
        )
        self.processing_thread.progress.connect(self._update_progress)
        self.processing_thread.slice_ready.connect(self._collect_slice_data)
        self.processing_thread.thumbnail_ready.connect(self._add_thumbnail_item)
        self.processing_thread.finished.connect(self._processing_finished)
        self.processing_thread.start()
        if DEBUG_MODE: print("DEBUG: _start_slice_processing - Thread started.")

    def _update_progress(self, value, message):
        if self.progress_dialog:
            try:
                self.progress_dialog.setValue(value)
                self.progress_dialog.setLabelText(message)
            except RuntimeError:
                 if DEBUG_MODE: print("DEBUG: Progress dialog accessed after deletion.")
                 self.progress_dialog = None

    def _cancel_processing(self):
        if DEBUG_MODE: print("DEBUG: _cancel_processing called.")
        if self.processing_thread and self.processing_thread.isRunning():
            print("INFO: Cancellation requested by user.")
            self.processing_thread.stop()

    def _collect_slice_data(self, index, slice_data, height_range):
        if DEBUG_MODE: print(f"DEBUG: _collect_slice_data received for index {index}. Data valid: {slice_data is not None}")
        self.current_slices[index] = slice_data

    def _add_thumbnail_item(self, index, pixmap, metadata):
        if DEBUG_MODE: print(f"DEBUG: _add_thumbnail_item received for index {index}. Pixmap valid: {not pixmap.isNull()}")
        item = QListWidgetItem(f"Slice {index}")
        item.setIcon(QIcon(pixmap))
        item.setData(Qt.ItemDataRole.UserRole, index)
        self.slice_list_widget.addItem(item)
        self.slice_metadata[index] = metadata
        if DEBUG_MODE: print(f"DEBUG: _add_thumbnail_item - Item added for index {index}, metadata stored.")

    def _processing_finished(self, success):
        if DEBUG_MODE: print(f"DEBUG: _processing_finished called. Success: {success}")
        if self.progress_dialog:
            try:
                self.progress_dialog.setValue(100)
            except RuntimeError:
                 self.progress_dialog = None
        self.processing_thread = None

        if self.plotter:
             try:
                 self.plotter.remove_actor("status_text", render=False)
             except Exception as e:
                 print(f"WARNING: Could not remove status text: {e}")

        if success:
            print(f"INFO: Successfully processed {len(self.current_slices)} slices.")
            self._update_3d_view_presentation()
        else:
            was_canceled = False
            if self.progress_dialog:
                try:
                    was_canceled = self.progress_dialog.wasCanceled()
                except RuntimeError:
                    was_canceled = True

            if was_canceled:
                 QMessageBox.information(self, "已取消", "切片处理已取消。")
                 if self.plotter:
                     self.plotter.clear()
                     self.plotter.add_text("处理已取消", position="upper_left", font_size=12)
                     self.plotter.render()
            else:
                 QMessageBox.warning(self, "处理失败", "切片处理过程中发生错误或未生成有效切片。")
                 if self.plotter:
                     self.plotter.clear()
                     self.plotter.add_text("处理失败", position="upper_left", font_size=12)
                     self.plotter.render()

        if self.progress_dialog:
            try:
                self.progress_dialog.close()
            except RuntimeError:
                 pass
            self.progress_dialog = None

    def _update_3d_view_presentation(self):
        if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation called.")
        if self.plotter is None:
            if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - Plotter is None, returning.")
            return

        try:
            if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - Clearing actors.")
            self.plotter.clear_actors()
        except Exception as e:
            print(f"ERROR: Failed to clear plotter actors in _update_3d_view_presentation: {e}")

        if not self.current_slices:
            if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - No slices to display.")
            try:
                self.plotter.render()
            except Exception as render_err:
                 print(f"ERROR: Render failed after clearing actors with no slices: {render_err}")
            return

        offset_value = self.offset_spin.value()
        point_size = self.point_size_spin.value()
        use_color = self.use_color_check.isChecked()
        if DEBUG_MODE: print(f"DEBUG: _update_3d_view_presentation - Params: offset={offset_value}, point_size={point_size}, use_color={use_color}")

        actors = []
        current_offset = 0.0
        sorted_indices = sorted(self.current_slices.keys())

        if DEBUG_MODE: print(f"DEBUG: _update_3d_view_presentation - Looping through {len(sorted_indices)} slices...")
        all_bounds = []

        for i in sorted_indices:
            slice_data = self.current_slices.get(i)
            if DEBUG_MODE: print(f"DEBUG: Processing slice {i}. Data valid: {slice_data is not None}")
            if slice_data is None or slice_data.n_points == 0:
                current_offset += offset_value
                continue

            offset_slice = slice_data.copy(deep=True)
            offset_slice.points[:, 2] += current_offset
            # Check bounds validity before adding
            if offset_slice.bounds[0] < offset_slice.bounds[1]:
                 all_bounds.extend(offset_slice.bounds)
            if DEBUG_MODE: print(f"DEBUG: Slice {i} offset applied. Current offset: {current_offset}")

            try:
                if DEBUG_MODE: print(f"DEBUG: Adding mesh for slice {i}...")
                if 'colors' in offset_slice.point_data and use_color:
                    actor = self.plotter.add_mesh(offset_slice, scalars='colors', rgb=True, point_size=point_size)
                else:
                    actor = self.plotter.add_mesh(offset_slice, color='grey', point_size=point_size)

                if actor:
                    actors.append(actor)
                    if DEBUG_MODE: print(f"DEBUG: Actor added successfully for slice {i}.")
                else:
                     print(f"WARNING: Failed to add actor for slice {i} (add_mesh returned None).")

            except Exception as e:
                print(f"ERROR: Error adding slice {i} to plotter: {e}")
                if DEBUG_MODE: traceback.print_exc()

            current_offset += offset_value

        if actors:
            if DEBUG_MODE: print(f"DEBUG: _update_3d_view_presentation - {len(actors)} actors added. Resetting camera.")
            try:
                if all_bounds:
                     # Calculate robust overall bounds, avoid min/max on empty list
                     min_x = min(all_bounds[0::6]) if all_bounds[0::6] else 0
                     max_x = max(all_bounds[1::6]) if all_bounds[1::6] else 1
                     min_y = min(all_bounds[2::6]) if all_bounds[2::6] else 0
                     max_y = max(all_bounds[3::6]) if all_bounds[3::6] else 1
                     min_z_off = min(all_bounds[4::6]) if all_bounds[4::6] else 0
                     max_z_off = max(all_bounds[5::6]) if all_bounds[5::6] else 1
                     overall_bounds = [min_x, max_x, min_y, max_y, min_z_off, max_z_off]

                     if DEBUG_MODE: print(f"DEBUG: Resetting camera to overall bounds: {overall_bounds}")
                     # Add padding to bounds for better view
                     padding = 0.1 * max(max_x - min_x, max_y - min_y, max_z_off - min_z_off) if all_bounds else 1.0
                     self.plotter.reset_camera(bounds=overall_bounds)
                     # self.plotter.camera.Zoom(0.9) # Zoom out slightly
                else:
                     self.plotter.reset_camera()

                self.plotter.view_vector([1, -1, 0.5], viewup=[0, 0, 1])
                if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - Camera reset and view set.")
            except Exception as e:
                print(f"ERROR: Error resetting camera: {e}")
                if DEBUG_MODE: traceback.print_exc()
        elif not self.current_slices:
             if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - No actors added and no slices exist.")
             # Text handled by _processing_finished
        else:
             if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - No actors added but slices exist (all empty?).")
             self.plotter.add_text("所有生成的切片均为空。", position="upper_left", font_size=12)

        try:
            if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - Calling plotter.render().")
            self.plotter.render()
            if DEBUG_MODE: print("DEBUG: _update_3d_view_presentation - plotter.render() finished.")
        except Exception as e:
            print(f"ERROR: Exception during final plotter.render(): {e}")

    def _on_selection_changed(self):
        selected_items = self.slice_list_widget.selectedItems()
        selected_indices = [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]
        if DEBUG_MODE: print(f"DEBUG: _on_selection_changed - Selected indices: {selected_indices}")

    def _export_bitmaps(self, indices_to_export):
        if DEBUG_MODE: print(f"DEBUG: _export_bitmaps called for indices: {indices_to_export}")
        if not indices_to_export: return

        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not export_dir:
            if DEBUG_MODE: print("DEBUG: _export_bitmaps - Export cancelled by user.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_export_path = os.path.join(export_dir, f"batch_slice_export_{timestamp}")
        os.makedirs(base_export_path, exist_ok=True)
        if DEBUG_MODE: print(f"DEBUG: _export_bitmaps - Exporting to: {base_export_path}")

        global_params = {
            "export_time": datetime.datetime.now().isoformat(),
            "original_point_cloud_source": self.source_filename,
            "num_slices_param": self.num_slices_spin.value(),
            "slice_thickness_param": self.thickness_spin.value(),
        }
        global_params_file = os.path.join(base_export_path, "export_parameters.json")
        if DEBUG_MODE: print("DEBUG: _export_bitmaps - Saving global parameters...")
        with open(global_params_file, 'w', encoding='utf-8') as f:
            json.dump(global_params, f, ensure_ascii=False, indent=2)

        export_progress = QProgressDialog("正在导出位图...", "取消", 0, len(indices_to_export), self)
        export_progress.setWindowTitle("导出进度")
        export_progress.setWindowModality(Qt.WindowModality.WindowModal)
        export_progress.setAutoClose(True)
        export_progress.setAutoReset(True)
        export_progress.show()

        exported_count = 0
        if DEBUG_MODE: print("DEBUG: _export_bitmaps - Starting export loop...")
        for i, index in enumerate(indices_to_export):
             export_progress.setValue(i)
             if export_progress.wasCanceled():
                 print("INFO: Export canceled by user.")
                 break

             if DEBUG_MODE: print(f"DEBUG: _export_bitmaps - Processing index {index}...")
             if index in self.current_slices and index in self.slice_metadata:
                slice_data = self.current_slices.get(index) # Use get for safety
                metadata = self.slice_metadata.get(index)

                # Double check metadata exists
                if metadata is None:
                    print(f"WARNING: Metadata missing for slice index {index} during export. Skipping.")
                    continue

                img_np = None
                view_params_render = None # Initialize

                if metadata.get("is_empty", False):
                     print(f"INFO: Skipping bitmap rendering for empty slice {index}.")
                     view_params_render = metadata.get("view_params")
                else:
                    export_progress.setLabelText(f"正在渲染切片 {index}...")
                    QApplication.processEvents()

                    if DEBUG_MODE: print(f"DEBUG: _export_bitmaps - Rendering bitmap for slice {index}...")
                    img_np, view_params_render = render_slice_to_image(
                        slice_data, self.BITMAP_EXPORT_RESOLUTION, is_thumbnail=False
                    )
                    if DEBUG_MODE: print(f"DEBUG: _export_bitmaps - Bitmap rendered for slice {index}. Valid: {img_np is not None}")

                # Save metadata
                meta_filename = os.path.join(base_export_path, f"slice_{index}_metadata.json")
                if DEBUG_MODE: print(f"DEBUG: _export_bitmaps - Saving metadata to {meta_filename}...")
                metadata["view_params_render"] = view_params_render if img_np is not None else None
                if img_np is None and not metadata.get("is_empty", False):
                     metadata["render_error"] = "Failed to generate bitmap"
                export_data = {"slice_index": index, "metadata": metadata}
                try:
                    with open(meta_filename, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, ensure_ascii=False, indent=2)
                    if DEBUG_MODE: print(f"DEBUG: _export_bitmaps - Metadata saved.")
                except Exception as meta_save_err:
                    print(f"ERROR: Failed to save metadata for slice {index}: {meta_save_err}")

                # Save bitmap if rendering was successful
                if img_np is not None:
                    try:
                        bitmap_filename = os.path.join(base_export_path, f"slice_{index}_bitmap.png")
                        if DEBUG_MODE: print(f"DEBUG: _export_bitmaps - Saving bitmap to {bitmap_filename}...")
                        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        success = cv2.imwrite(bitmap_filename, img_bgr)
                        if not success:
                             print(f"ERROR: Failed to save bitmap file: {bitmap_filename}")
                        else:
                             if DEBUG_MODE: print(f"DEBUG: _export_bitmaps - Bitmap saved successfully.")
                             exported_count += 1
                    except Exception as export_err:
                         print(f"ERROR: Failed to save bitmap for slice {index}: {export_err}")
                         if DEBUG_MODE: traceback.print_exc()
                elif metadata.get("is_empty", False):
                     exported_count += 1 # Count empty slices where metadata was saved

             else:
                print(f"WARNING: Data or metadata missing for slice index {index} during export loop. Skipping.")

        export_progress.setValue(len(indices_to_export))
        if DEBUG_MODE: print(f"DEBUG: _export_bitmaps finished. Exported count: {exported_count}")
        QMessageBox.information(self, "导出完成", f"成功导出 {exported_count} 个切片的数据到:\n{base_export_path}")

    def _export_selected_bitmaps(self):
        selected_items = self.slice_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "未选择", "请先在左侧列表中选择要导出的项。")
            return
        indices_to_export = sorted([item.data(Qt.ItemDataRole.UserRole) for item in selected_items])
        self._export_bitmaps(indices_to_export)

    def _export_all_data(self):
        if not self.current_slices or not self.slice_metadata:
            QMessageBox.warning(self, "无数据", "没有有效的切片数据或元数据可供导出。请先生成切片。")
            return
        indices_to_export = sorted(list(self.slice_metadata.keys())) # Export based on available metadata
        self._export_bitmaps(indices_to_export)

    def closeEvent(self, event):
        if DEBUG_MODE: print("DEBUG: BatchSliceViewerWindow closeEvent called.")
        self._cancel_processing()
        if self.processing_thread and self.processing_thread.isRunning():
            print("INFO: Waiting for processing thread to finish before closing...")
            finished = self.processing_thread.wait(3000)
            if not finished:
                print("WARNING: Processing thread did not finish in time. Window closing anyway.")
        if self.plotter:
             if DEBUG_MODE: print("DEBUG: Closing plotter in closeEvent.")
             try:
                 # Attempt cleanup, but be prepared for errors if context is invalid
                 self.plotter.close()
             except Exception as e:
                 print(f"ERROR: Exception while closing plotter in closeEvent: {e}")
        super().closeEvent(event)