import os
import sys
import datetime
import traceback
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QFileDialog, QSplitter, QStatusBar)
from PyQt6.QtCore import Qt
from pyvistaqt import QtInteractor, MainWindow

# 导入自定义模块
from .ui.sidebar_builder import SidebarBuilder
from .ui.toolbar_builder import ToolbarBuilder
from .ui.screenshot_dialog import ScreenshotDialog
from .utils.point_cloud_handler import PointCloudHandler
from .utils.visualization import VisualizationManager
from .utils.stylesheet_manager import StylesheetManager


class PCDViewerWindow(MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PCD Viewer with Slicing")
        self.resize(1200, 800)

        # --- 初始化变量 ---
        self.point_cloud = None  # 原始点云
        self.current_slice_cloud = None  # 当前切片点云
        self.pcd_actor = None  # 点云渲染器actor
        self.pcd_bounds = None  # 点云边界
        self.is_sidebar_visible = True  # 侧边栏可见性状态

        # --- 初始化UI构建器 ---
        self.sidebar_builder = SidebarBuilder(self)
        self.toolbar_builder = ToolbarBuilder(self)

        # --- 创建主界面 ---
        self.setup_ui()

        # --- 初始化状态栏 ---
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        # --- 应用样式 ---
        self.setStyleSheet(StylesheetManager.get_light_theme())

        # --- 初始化显示 ---
        try:
            self.load_pcd_file(os.path.join(os.path.dirname(__file__), "samples", "one_floor.pcd"))
        except Exception as e:
            self.statusBar.showMessage(f"Error loading initial file: {str(e)}")

    def setup_ui(self):
        """创建用户界面"""
        # --- 主布局容器 ---
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- 创建分割器 ---
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)

        # --- 创建侧边栏 ---
        self.sidebar = self.sidebar_builder.build()

        # --- 创建 3D 视图区域 ---
        self.plotter_widget = QWidget()
        plotter_layout = QVBoxLayout(self.plotter_widget)
        plotter_layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        plotter_layout.addWidget(self.plotter)
        self.plotter.set_background("white")
        self.plotter.enable_mesh_picking()

        # --- 添加部件到分割器 ---
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.plotter_widget)

        # 设置分割比例
        self.splitter.setSizes([250, 950])

        # --- 创建工具栏 ---
        self.addToolBar(self.toolbar_builder.build())

    def toggle_sidebar(self):
        """切换侧边栏的可见性"""
        if self.is_sidebar_visible:
            self.splitter.setSizes([0, self.width()])
            self.is_sidebar_visible = False
        else:
            self.splitter.setSizes([250, self.width() - 250])
            self.is_sidebar_visible = True

    def reset_view(self):
        """重置视图"""
        self.plotter.reset_camera()
        self.statusBar.showMessage("视图已重置")

    def open_pcd_file(self):
        """打开PCD文件对话框"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择PCD文件", "",
                                                   "点云文件 (*.pcd *.ply *.xyz *.pts);;所有文件 (*)")
        if file_path:
            self.load_pcd_file(file_path)

    def load_pcd_file(self, file_path):
        """加载点云文件"""
        try:
            self.statusBar.showMessage(f"正在加载: {os.path.basename(file_path)}...")

            # 使用PointCloudHandler加载点云
            self.point_cloud, self.pcd_bounds, point_count = PointCloudHandler.load_from_file(file_path)

            # 更新状态
            self.statusBar.showMessage(f"已加载: {os.path.basename(file_path)}, {point_count} 点")

            # 更新可视化
            self.update_visualization()

            # 更新信息面板
            self.update_info_panel()

            return True
        except Exception as e:
            self.statusBar.showMessage(f"加载文件失败: {str(e)}")
            self.point_cloud = None
            self.pcd_bounds = None
            return False

    def update_visualization(self):
        """更新3D视图中的点云显示"""
        if self.point_cloud is None:
            return

        try:
            # 获取当前UI控件的值
            height_ratio = self.sidebar_builder.slice_height_slider.value() / 100.0
            thickness_text = self.sidebar_builder.thickness_input.text()
            try:
                thickness_ratio = float(thickness_text)
            except ValueError:
                thickness_ratio = 0.1
                self.sidebar_builder.thickness_input.setText("0.1")

            # 进行点云切片
            sliced_cloud, slice_point_count, height_range = PointCloudHandler.slice_by_height(
                self.point_cloud, height_ratio, thickness_ratio
            )

            # 更新高度标签
            self.sidebar_builder.update_height_label(height_ratio, height_range)

            # 如果切片为空，显示完整点云
            if sliced_cloud is None or slice_point_count == 0:
                sliced_cloud = self.point_cloud
                self.statusBar.showMessage("当前切片区域内没有点，显示完整点云")

            # 存储当前切片点云
            self.current_slice_cloud = sliced_cloud

            # 渲染点云
            use_colors = self.sidebar_builder.color_checkbox.isChecked()
            point_size = self.sidebar_builder.point_size_spinner.value()
            render_mode = "Mesh" if self.sidebar_builder.render_mode.currentText() == "网格" else "Points"

            # 使用VisualizationManager显示点云
            self.pcd_actor = VisualizationManager.display_point_cloud(
                self.plotter, sliced_cloud, use_colors, point_size, render_mode
            )

            # 更新切片信息
            self.update_slice_info()

        except Exception as e:
            self.statusBar.showMessage(f"更新显示时出错: {str(e)}")

    def save_screenshot(self):
        """使用自定义对话框保存当前视图截图"""
        dialog = ScreenshotDialog(self)
        if not dialog.exec():
            return

        settings = dialog.get_settings()

        # 构建默认文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"pcd_view_{timestamp}.{settings['format']}"

        # 打开文件保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存截图", default_filename,
            f"{settings['format'].upper()} Files (*.{settings['format']})"
        )

        if not file_path:
            return

        # 使用VisualizationManager保存截图
        success = VisualizationManager.save_screenshot(self.plotter, file_path, settings)

        if success:
            self.statusBar.showMessage(f"截图已保存到: {os.path.basename(file_path)}")
        else:
            self.statusBar.showMessage("保存截图失败")

    def update_info_panel(self):
        """更新信息面板显示"""
        if self.point_cloud is not None:
            # 获取点云信息
            cloud_info = PointCloudHandler.get_cloud_info(self.point_cloud)

            # 获取切片信息
            slice_info = None
            if self.current_slice_cloud is not None:
                slice_info = (len(self.current_slice_cloud.points), None)

            # 更新信息面板
            self.sidebar_builder.update_info_panel(cloud_info, slice_info)

    def update_slice_info(self):
        """更新切片信息"""
        if self.current_slice_cloud is not None:
            slice_info = (len(self.current_slice_cloud.points), None)
            # 只更新切片部分信息
            self.sidebar_builder.slice_info_label.setText(f"当前切片: {slice_info[0]} 点")