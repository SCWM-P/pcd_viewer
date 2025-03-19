# pcd_viewer/ui/line_detection_dialog.py

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QSlider, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QCheckBox, QTabWidget, QWidget, QSplitter,
                             QFormLayout, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QDateTime
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor
import numpy as np
import cv2
import os
import json
import datetime


class LayeredImageDisplay(QWidget):
    """使用独立图层实现的图像显示控件"""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #f0f0f0;")

        # 图层数据
        self.original_image = None  # 原始图像数据（numpy数组）
        self.result_image = None  # 结果图像数据（numpy数组）

        # 图层控制
        self.show_original = True
        self.show_result = True
        self.opacity = 0.5  # 结果图层的不透明度

        # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 创建标签用于显示
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_label.setText("无图像")
        layout.addWidget(self.display_label)

    def set_original_image(self, image):
        """设置原始图像"""
        self.original_image = image
        self.update_display()

    def set_result_image(self, image):
        """设置结果图像"""
        self.result_image = image
        self.update_display()

    def set_layer_visibility(self, show_original, show_result):
        """设置图层可见性"""
        self.show_original = show_original
        self.show_result = show_result
        self.update_display()

    def set_opacity(self, value):
        """设置结果图层的不透明度 (0-100)"""
        self.opacity = value / 100.0
        self.update_display()

    def update_display(self):
        """更新显示"""
        if self.original_image is None and self.result_image is None:
            self.display_label.setText("无图像")
            return

        # 计算需要的图像大小
        height, width = 0, 0
        if self.original_image is not None:
            height, width = self.original_image.shape[:2]
        elif self.result_image is not None:
            height, width = self.result_image.shape[:2]

        # 创建空白画布
        canvas = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA格式，完全透明

        # 绘制原始图层（如果可见）
        if self.show_original and self.original_image is not None:
            # 将RGB图像转换为RGBA
            if self.original_image.shape[2] == 3:
                # 添加Alpha通道（完全不透明）
                original_rgba = np.zeros((height, width, 4), dtype=np.uint8)
                original_rgba[:, :, :3] = self.original_image
                original_rgba[:, :, 3] = 255
            else:
                original_rgba = self.original_image.copy()

            # 将原始图像复制到画布
            canvas = original_rgba.copy()

        # 绘制结果图层（如果可见）
        if self.show_result and self.result_image is not None:
            # 将结果图像转换为RGBA
            result_rgba = np.zeros((height, width, 4), dtype=np.uint8)

            # 只处理非零像素（假设结果图像中0像素为透明区域）
            non_zero_mask = np.any(self.result_image > 0, axis=2)
            result_rgba[non_zero_mask, :3] = self.result_image[non_zero_mask]
            result_rgba[non_zero_mask, 3] = int(255 * self.opacity)  # 设置不透明度

            # 如果原始图层不可见，或者没有原始图像，直接使用结果图像
            if not self.show_original or self.original_image is None:
                canvas = result_rgba.copy()
            else:
                # 否则，根据Alpha通道混合两个图层
                alpha = result_rgba[:, :, 3:4] / 255.0
                canvas[:, :, :3] = (1 - alpha) * canvas[:, :, :3] + alpha * result_rgba[:, :, :3]
                canvas[:, :, 3] = np.maximum(canvas[:, :, 3], result_rgba[:, :, 3])  # 取两者中的最大Alpha值

        # 转换为Qt图像并显示
        height, width = canvas.shape[:2]
        bytes_per_line = 4 * width
        q_img = QImage(canvas.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_img)

        # 缩放图像以适应标签大小
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.display_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.display_label.setPixmap(scaled_pixmap)
        else:
            self.display_label.setText("图像显示错误")

    def resizeEvent(self, event):
        """处理大小变化事件"""
        super().resizeEvent(event)
        if hasattr(self, 'display_label') and self.display_label.pixmap():
            # 更新显示以适应新的大小
            self.update_display()

    def get_current_display_image(self):
        """获取当前显示的完整图像"""
        if hasattr(self, 'display_label') and self.display_label.pixmap():
            # 从QPixmap获取QImage
            image = self.display_label.pixmap().toImage()

            # 转换为numpy数组
            width = image.width()
            height = image.height()
            ptr = image.constBits()
            ptr.setsize(height * width * 4)  # RGBA
            arr = np.array(ptr).reshape(height, width, 4)

            # 返回RGB部分
            return arr[:, :, :3].copy()
        return None


class HoughParametersWidget(QWidget):
    """霍夫变换参数设置界面"""

    paramChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """设置界面"""
        layout = QFormLayout(self)

        # 阈值
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(10, 300)
        self.threshold_slider.setValue(100)
        self.threshold_slider.valueChanged.connect(self.on_param_changed)
        layout.addRow("阈值:", self.threshold_slider)
        self.threshold_value = QLabel("100")
        layout.addRow("", self.threshold_value)
        self.threshold_slider.valueChanged.connect(
            lambda val: self.threshold_value.setText(str(val))
        )

        # 最小线长度
        self.min_line_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_line_slider.setRange(10, 200)
        self.min_line_slider.setValue(50)
        self.min_line_slider.valueChanged.connect(self.on_param_changed)
        layout.addRow("最小线长:", self.min_line_slider)
        self.min_line_value = QLabel("50")
        layout.addRow("", self.min_line_value)
        self.min_line_slider.valueChanged.connect(
            lambda val: self.min_line_value.setText(str(val))
        )

        # 最大线间隙
        self.max_gap_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_gap_slider.setRange(1, 50)
        self.max_gap_slider.setValue(10)
        self.max_gap_slider.valueChanged.connect(self.on_param_changed)
        layout.addRow("最大线间隙:", self.max_gap_slider)
        self.max_gap_value = QLabel("10")
        layout.addRow("", self.max_gap_value)
        self.max_gap_slider.valueChanged.connect(
            lambda val: self.max_gap_value.setText(str(val))
        )

    def on_param_changed(self):
        """参数变化时发射信号"""
        self.paramChanged.emit()

    def get_params(self):
        """获取当前参数"""
        return {
            "threshold": self.threshold_slider.value(),
            "min_line_length": self.min_line_slider.value(),
            "max_line_gap": self.max_gap_slider.value()
        }


class RANSACParametersWidget(QWidget):
    """RANSAC参数设置界面"""

    paramChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        """设置界面"""
        layout = QFormLayout(self)

        # 残差阈值
        self.residual_spin = QDoubleSpinBox()
        self.residual_spin.setRange(0.1, 10.0)
        self.residual_spin.setSingleStep(0.1)
        self.residual_spin.setValue(2.0)
        self.residual_spin.valueChanged.connect(self.on_param_changed)
        layout.addRow("残差阈值:", self.residual_spin)

        # 最大迭代次数
        self.max_trials_spin = QSpinBox()
        self.max_trials_spin.setRange(100, 5000)
        self.max_trials_spin.setSingleStep(100)
        self.max_trials_spin.setValue(1000)
        self.max_trials_spin.valueChanged.connect(self.on_param_changed)
        layout.addRow("最大迭代次数:", self.max_trials_spin)

    def on_param_changed(self):
        """参数变化时发射信号"""
        self.paramChanged.emit()

    def get_params(self):
        """获取当前参数"""
        return {
            "residual_threshold": self.residual_spin.value(),
            "max_trials": self.max_trials_spin.value()
        }


class LineDetectionDialog(QWidget):
    """直线检测对话框"""

    def __init__(self, parent=None, cloud_file_name=""):
        super().__init__(parent)
        self.setWindowTitle("直线检测")
        self.resize(900, 600)
        # 添加窗口标志，使其能有最小化、最大化按钮
        self.setWindowFlags(Qt.WindowType.Window)

        # 图像数据
        self.original_image = None
        self.result_image = None
        self.detected_lines = []
        self.cloud_file_name = cloud_file_name if cloud_file_name else "未命名"

        # 是否连续检测
        self.continuous_detection = False

        # 从父窗口导入直线检测器
        from ..utils.line_detection import LineDetectionManager
        self.line_manager = LineDetectionManager()

        self.setup_ui()

    def setup_ui(self):
        """设置界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)  # 移除边距以最大化空间利用

        # 创建水平分割器
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)

        # 创建左侧图像显示区容器
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(5, 5, 5, 5)

        # 图像显示控件
        self.image_display = LayeredImageDisplay()
        image_layout.addWidget(self.image_display)

        # 创建右侧控制面板容器
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)

        # 添加控制面板组件
        # 算法选择
        algo_group = QGroupBox("算法选择")
        algo_layout = QVBoxLayout(algo_group)
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(self.line_manager.get_detector_names())
        self.algo_combo.currentTextChanged.connect(self.on_algorithm_changed)
        algo_layout.addWidget(self.algo_combo)
        control_layout.addWidget(algo_group)

        # 参数设置标签页
        self.param_tabs = QTabWidget()

        # 霍夫变换参数页
        self.hough_param_widget = HoughParametersWidget()
        self.hough_param_widget.paramChanged.connect(self.check_continuous_detection)
        self.param_tabs.addTab(self.hough_param_widget, "霍夫变换")

        # RANSAC参数页
        self.ransac_param_widget = RANSACParametersWidget()
        self.ransac_param_widget.paramChanged.connect(self.check_continuous_detection)
        self.param_tabs.addTab(self.ransac_param_widget, "RANSAC")

        control_layout.addWidget(self.param_tabs)

        # 连续检测选项
        self.continuous_detection_check = QCheckBox("连续检测")
        self.continuous_detection_check.setChecked(False)
        self.continuous_detection_check.stateChanged.connect(
            lambda state: setattr(self, 'continuous_detection', state == Qt.CheckState.Checked.value)
        )
        control_layout.addWidget(self.continuous_detection_check)

        # 图层控制
        layer_group = QGroupBox("图层控制")
        layer_layout = QVBoxLayout(layer_group)

        # 图层可见性
        self.show_original_check = QCheckBox("显示原始图像")
        self.show_original_check.setChecked(True)
        self.show_original_check.stateChanged.connect(self.update_layer_visibility)
        layer_layout.addWidget(self.show_original_check)

        self.show_result_check = QCheckBox("显示检测结果")
        self.show_result_check.setChecked(True)
        self.show_result_check.stateChanged.connect(self.update_layer_visibility)
        layer_layout.addWidget(self.show_result_check)

        # 不透明度控制
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("不透明度:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        layer_layout.addLayout(opacity_layout)

        control_layout.addWidget(layer_group)

        # 检测控制
        detect_group = QGroupBox("检测控制")
        detect_layout = QVBoxLayout(detect_group)

        self.detect_btn = QPushButton("检测直线")
        self.detect_btn.clicked.connect(self.detect_lines)
        detect_layout.addWidget(self.detect_btn)

        # 线条颜色和粗细
        line_color_layout = QHBoxLayout()
        line_color_layout.addWidget(QLabel("线条颜色:"))
        self.line_color_combo = QComboBox()
        self.line_color_combo.addItems(["绿色", "红色", "蓝色", "黄色"])
        self.line_color_combo.currentTextChanged.connect(self.on_line_color_changed)
        line_color_layout.addWidget(self.line_color_combo)
        detect_layout.addLayout(line_color_layout)

        line_width_layout = QHBoxLayout()
        line_width_layout.addWidget(QLabel("线条粗细:"))
        self.line_width_spin = QSpinBox()
        self.line_width_spin.setRange(1, 10)
        self.line_width_spin.setValue(2)
        self.line_width_spin.valueChanged.connect(self.on_line_width_changed)
        line_width_layout.addWidget(self.line_width_spin)
        detect_layout.addLayout(line_width_layout)

        control_layout.addWidget(detect_group)

        # 导出功能组
        export_group = QGroupBox("导出")
        export_layout = QVBoxLayout(export_group)

        self.export_btn = QPushButton("导出检测结果")
        self.export_btn.clicked.connect(self.export_results)
        export_layout.addWidget(self.export_btn)

        control_layout.addWidget(export_group)

        # 添加伸缩因子以填充空间
        control_layout.addStretch()

        # 添加关闭按钮
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)
        control_layout.addWidget(self.close_btn)

        # 添加到分割器中
        self.main_splitter.addWidget(image_container)
        self.main_splitter.addWidget(control_panel)

        # 设置初始分割比例 - 可以根据需要调整这些数值
        # 例如 [700, 200] 表示左侧占700像素，右侧占200像素
        self.main_splitter.setSizes([700, 200])

        # 添加调整比例的按钮/功能
        # self.add_ratio_control()

    # def add_ratio_control(self):
    #     """添加比例控制功能"""
    #     # 创建比例预设按钮组
    #     ratio_group = QGroupBox("布局比例")
    #     ratio_layout = QHBoxLayout(ratio_group)
    #
    #     # 75:25 比例按钮
    #     ratio_75_25_btn = QPushButton("75:25")
    #     ratio_75_25_btn.clicked.connect(lambda: self.set_splitter_ratio(0.75))
    #     ratio_layout.addWidget(ratio_75_25_btn)
    #
    #     # 70:30 比例按钮
    #     ratio_70_30_btn = QPushButton("70:30")
    #     ratio_70_30_btn.clicked.connect(lambda: self.set_splitter_ratio(0.70))
    #     ratio_layout.addWidget(ratio_70_30_btn)
    #
    #     # 60:40 比例按钮
    #     ratio_60_40_btn = QPushButton("60:40")
    #     ratio_60_40_btn.clicked.connect(lambda: self.set_splitter_ratio(0.60))
    #     ratio_layout.addWidget(ratio_60_40_btn)
    #
    #     # 50:50 比例按钮
    #     ratio_50_50_btn = QPushButton("50:50")
    #     ratio_50_50_btn.clicked.connect(lambda: self.set_splitter_ratio(0.50))
    #     ratio_layout.addWidget(ratio_50_50_btn)
    #
    #     # 将比例控制组添加到右侧面板顶部
    #     control_panel = self.main_splitter.widget(1)
    #     control_layout = control_panel.layout()
    #     control_layout.insertWidget(0, ratio_group)

    def set_splitter_ratio(self, left_ratio):
        """
        设置分割器左右比例

        Args:
            left_ratio (float): 左侧所占比例 (0.0-1.0)
        """
        total_width = self.main_splitter.width()
        left_width = int(total_width * left_ratio)
        right_width = total_width - left_width
        self.main_splitter.setSizes([left_width, right_width])

    def check_continuous_detection(self):
        """根据连续检测状态决定是否立即更新"""
        if self.continuous_detection:
            self.detect_lines()
        # 否则不做任何事，等待用户点击"检测直线"按钮

    def set_image(self, image):
        """
        设置要处理的图像

        Args:
            image: Numpy数组格式的图像（RGB）
        """
        if image is not None:
            # OpenCV使用BGR格式，而Qt使用RGB
            self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_display.set_original_image(self.original_image)
            self.result_image = None
            self.detected_lines = []

    def on_algorithm_changed(self, algorithm_name):
        """
        当选择的算法改变时

        Args:
            algorithm_name: 选择的算法名称
        """
        self.line_manager.set_current_detector(algorithm_name)

        # 切换参数标签页
        if algorithm_name == "Hough":
            self.param_tabs.setCurrentWidget(self.hough_param_widget)
        elif algorithm_name == "RANSAC":
            self.param_tabs.setCurrentWidget(self.ransac_param_widget)

        # 如果启用了连续检测，则重新检测直线
        if self.continuous_detection and self.original_image is not None:
            self.detect_lines()

    def on_opacity_changed(self, value):
        """改变结果图层的不透明度"""
        self.image_display.set_opacity(value)

    def update_layer_visibility(self):
        """更新图层可见性"""
        show_original = self.show_original_check.isChecked()
        show_result = self.show_result_check.isChecked()
        self.image_display.set_layer_visibility(show_original, show_result)

    def on_line_color_changed(self):
        """线条颜色变化时重新绘制"""
        if self.original_image is not None and len(self.detected_lines) > 0:
            self.update_result_display()

    def on_line_width_changed(self):
        """线条粗细变化时重新绘制"""
        if self.original_image is not None and len(self.detected_lines) > 0:
            self.update_result_display()

    def get_line_color(self):
        """获取当前选择的线条颜色"""
        color_name = self.line_color_combo.currentText()
        if color_name == "绿色":
            return (0, 255, 0)
        elif color_name == "红色":
            return (255, 0, 0)
        elif color_name == "蓝色":
            return (0, 0, 255)
        elif color_name == "黄色":
            return (255, 255, 0)
        else:
            return (0, 255, 0)  # 默认绿色

    def detect_lines(self):
        """检测直线"""
        if self.original_image is None:
            return

        # 获取当前选择的算法
        detector = self.line_manager.get_current_detector()

        # 设置参数
        if self.algo_combo.currentText() == "Hough":
            params = self.hough_param_widget.get_params()
        else:
            params = self.ransac_param_widget.get_params()

        detector.set_params(params)

        # 检测直线
        self.detected_lines = detector.detect(self.original_image)

        # 更新结果显示
        self.update_result_display()

    def update_result_display(self):
        """更新结果图像显示"""
        if self.original_image is None or len(self.detected_lines) == 0:
            return

        # 获取当前设置
        line_color = self.get_line_color()
        line_thickness = self.line_width_spin.value()

        # 创建透明背景的图像，只包含检测到的直线
        h, w = self.original_image.shape[:2]
        transparent_result = np.zeros((h, w, 3), dtype=np.uint8)  # 创建空白图像

        # 绘制直线
        for line in self.detected_lines:
            x1, y1, x2, y2 = line
            cv2.line(transparent_result, (x1, y1), (x2, y2), line_color, line_thickness)

        # 更新显示
        self.result_image = transparent_result
        self.image_display.set_result_image(self.result_image)

    def export_results(self):
        """导出检测结果"""
        if self.original_image is None or len(self.detected_lines) == 0:
            QMessageBox.warning(self, "导出失败", "没有可导出的检测结果")
            return

        # 创建导出目录
        algorithm_name = self.algo_combo.currentText()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_folder_name = f"{os.path.splitext(self.cloud_file_name)[0]}_{algorithm_name}_{timestamp}"

        # 选择保存位置
        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not export_dir:
            return

        export_path = os.path.join(export_dir, export_folder_name)
        try:
            if not os.path.exists(export_path):
                os.makedirs(export_path)

            # 1. 导出JSON数据
            json_data = {
                "export_time": datetime.datetime.now().isoformat(),
                "algorithm": algorithm_name,
                "parameters": self.hough_param_widget.get_params() if algorithm_name == "Hough" else self.ransac_param_widget.get_params(),
                "lines": []
            }

            for i, line in enumerate(self.detected_lines):
                x1, y1, x2, y2 = line
                line_data = {
                    "id": i,
                    "start_point": [int(x1), int(y1)],
                    "end_point": [int(x2), int(y2)],
                    "length": round(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 2)
                }
                json_data["lines"].append(line_data)

            with open(os.path.join(export_path, "lines.json"), 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            # 2. 导出图片
            # 原始图像
            original_path = os.path.join(export_path, "original.png")
            cv2.imwrite(original_path, cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR))

            # 结果图像（只有线条）
            result_path = os.path.join(export_path, "lines_only.png")
            cv2.imwrite(result_path, cv2.cvtColor(self.result_image, cv2.COLOR_RGB2BGR))

            # 叠加图像 - 设置为同时显示两个图层
            self.show_original_check.setChecked(True)
            self.show_result_check.setChecked(True)
            self.image_display.set_opacity(50)
            self.update_layer_visibility()

            # 获取当前显示的图像
            combined_image = self.image_display.get_current_display_image()
            if combined_image is not None:
                combined_path = os.path.join(export_path, "combined.png")
                cv2.imwrite(combined_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

            QMessageBox.information(self, "导出成功", f"检测结果已导出到:\n{export_path}")

        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"导出过程中发生错误:\n{str(e)}")


# 修改 line_detection.py 中的 draw_detected_lines 函数，添加透明背景支持
def draw_detected_lines(image, lines, color=(0, 255, 0), thickness=2, transparent_bg=False):
    """
    在图像上绘制检测到的直线

    Args:
        image: 输入图像（RGB或BGR）
        lines: 检测到的直线列表，每条线为 [x1, y1, x2, y2] 格式
        color: 线条颜色，默认为绿色
        thickness: 线条粗细
        transparent_bg: 是否使用透明背景（只绘制线条）

    Returns:
        image_with_lines: 绘制了直线的图像
    """
    if transparent_bg:
        # 创建透明背景图像
        image_with_lines = np.zeros_like(image)
    else:
        # 创建副本以免修改原图
        image_with_lines = image.copy()

    # 绘制每条直线
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(image_with_lines, (x1, y1), (x2, y2), color, thickness)

    return image_with_lines