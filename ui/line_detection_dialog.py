# pcd_viewer/ui/line_detection_dialog.py

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QSlider, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QCheckBox, QTabWidget, QWidget, QSplitter,
                             QFormLayout)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor
import numpy as np
import cv2


class ImageDisplayWidget(QLabel):
    """用于显示图像的自定义控件"""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #f0f0f0; border: 1px solid #dddddd;")
        self.setText("无图像")
        self.pixmap = None
        self.original_pixmap = None
        self.result_pixmap = None
        self.opacity = 0.5  # 图层混合的透明度
        self.show_original = True
        self.show_result = True

    def set_original_image(self, image):
        """设置原始图像"""
        if image is not None:
            # 转换为Qt格式
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
            self.original_pixmap = QPixmap.fromImage(qImg)
            self.update_display()
        else:
            self.original_pixmap = None
            self.setText("无原始图像")

    def set_result_image(self, image):
        """设置结果图像"""
        if image is not None:
            # 转换为Qt格式
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
            self.result_pixmap = QPixmap.fromImage(qImg)
            self.update_display()
        else:
            self.result_pixmap = None

    def set_opacity(self, value):
        """设置图层混合的透明度"""
        self.opacity = value / 100.0
        self.update_display()

    def set_layer_visibility(self, original, result):
        """设置图层可见性"""
        self.show_original = original
        self.show_result = result
        self.update_display()

    def update_display(self):
        """更新显示"""
        if self.original_pixmap is None and self.result_pixmap is None:
            self.setText("无图像")
            return

        # 确定要显示的图像
        if self.show_original and not self.show_result:
            # 只显示原始图像
            self.pixmap = self.original_pixmap
        elif not self.show_original and self.show_result:
            # 只显示结果图像
            self.pixmap = self.result_pixmap
        elif self.show_original and self.show_result and self.original_pixmap and self.result_pixmap:
            # 混合两个图像
            self.pixmap = QPixmap(self.original_pixmap.size())
            self.pixmap.fill(QColor(0, 0, 0, 0))
            painter = QPainter(self.pixmap)
            painter.drawPixmap(0, 0, self.original_pixmap)
            painter.setOpacity(self.opacity)
            painter.drawPixmap(0, 0, self.result_pixmap)
            painter.end()
        else:
            # 显示可用的那个
            self.pixmap = self.original_pixmap if self.original_pixmap else self.result_pixmap

        # 更新显示
        if self.pixmap:
            # 根据控件大小缩放
            scaled_pixmap = self.pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
        else:
            self.setText("无图像")

    def resizeEvent(self, event):
        """重载大小变化事件，确保图像正确缩放"""
        super().resizeEvent(event)
        if self.pixmap:
            self.update_display()


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


class LineDetectionDialog(QDialog):
    """直线检测对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("直线检测")
        self.resize(900, 600)

        # 图像数据
        self.original_image = None
        self.result_image = None
        self.detected_lines = []

        # 从父窗口导入直线检测器
        from ..utils.line_detection import LineDetectionManager
        self.line_manager = LineDetectionManager()

        self.setup_ui()

    def setup_ui(self):
        """设置界面"""
        main_layout = QVBoxLayout(self)

        # 上部分：图像显示和控制
        top_layout = QHBoxLayout()

        # 图像显示区
        self.image_display = ImageDisplayWidget()
        top_layout.addWidget(self.image_display)

        # 右侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)

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
        self.hough_param_widget.paramChanged.connect(self.on_params_changed)
        self.param_tabs.addTab(self.hough_param_widget, "霍夫变换")

        # RANSAC参数页
        self.ransac_param_widget = RANSACParametersWidget()
        self.ransac_param_widget.paramChanged.connect(self.on_params_changed)
        self.param_tabs.addTab(self.ransac_param_widget, "RANSAC")

        control_layout.addWidget(self.param_tabs)

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

        # 线条颜色
        line_color_layout = QHBoxLayout()
        line_color_layout.addWidget(QLabel("线条颜色:"))
        self.line_color_combo = QComboBox()
        self.line_color_combo.addItems(["绿色", "红色", "蓝色", "黄色"])
        self.line_color_combo.currentTextChanged.connect(self.on_line_color_changed)
        line_color_layout.addWidget(self.line_color_combo)
        detect_layout.addLayout(line_color_layout)

        # 线条粗细
        line_width_layout = QHBoxLayout()
        line_width_layout.addWidget(QLabel("线条粗细:"))
        self.line_width_spin = QSpinBox()
        self.line_width_spin.setRange(1, 10)
        self.line_width_spin.setValue(2)
        self.line_width_spin.valueChanged.connect(self.on_line_width_changed)
        line_width_layout.addWidget(self.line_width_spin)
        detect_layout.addLayout(line_width_layout)

        control_layout.addWidget(detect_group)

        # 添加伸缩因子以填充空间
        control_layout.addStretch()

        # 添加关闭按钮
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.accept)
        control_layout.addWidget(self.close_btn)

        top_layout.addWidget(control_panel)
        main_layout.addLayout(top_layout)

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

        # 重新检测直线
        if self.original_image is not None:
            self.detect_lines()

    def on_params_changed(self):
        """参数变化时重新检测直线"""
        if self.original_image is not None:
            self.detect_lines()

    def on_opacity_changed(self, value):
        """改变图层混合的不透明度"""
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

        # 绘制直线
        from ..utils.line_detection import draw_detected_lines
        self.result_image = draw_detected_lines(
            self.original_image,
            self.detected_lines,
            color=line_color,
            thickness=line_thickness
        )

        # 更新显示
        self.image_display.set_result_image(self.result_image)