from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QWidget, QFrame,
                             QPushButton, QCheckBox, QSlider, QLineEdit, QLabel,
                             QGroupBox, QTabWidget, QComboBox, QSpinBox)
from PyQt6.QtCore import Qt
from .height_distribution_widget import HeightDistributionWidget
import qtawesome as qta


class SidebarBuilder:
    """构建应用程序侧边栏"""

    def __init__(self, parent):
        """
        初始化侧边栏构建器

        Args:
            parent: 父窗口，用于连接信号和槽
        """
        self.parent = parent
        self.sidebar_widget = None
        self.tab_widget = None

        # UI控件引用
        self.color_checkbox = None
        self.point_size_spinner = None
        self.render_mode = None
        self.thickness_input = None
        self.slice_height_slider = None
        self.height_value_label = None
        self.points_count_label = None
        self.bounds_label = None
        self.slice_info_label = None
        self.height_dist_widget = None

    def build(self):
        """
        构建侧边栏

        Returns:
            QWidget: 构建好的侧边栏控件
        """
        # 创建侧边栏容器
        self.sidebar_widget = QWidget()
        self.sidebar_widget.setMinimumWidth(250)
        self.sidebar_widget.setMaximumWidth(400)
        sidebar_layout = QVBoxLayout(self.sidebar_widget)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)

        # 创建标签页控件
        self.tab_widget = QTabWidget()
        sidebar_layout.addWidget(self.tab_widget)

        # 添加控制面板标签页
        self._create_control_tab()

        # 添加信息面板标签页
        self._create_info_tab()

        return self.sidebar_widget

    def _create_control_tab(self):
        """创建控制面板标签页"""
        control_tab = QWidget()
        control_layout = QVBoxLayout(control_tab)
        control_layout.setContentsMargins(5, 5, 5, 5)

        # --- 文件操作区 ---
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)

        # 加载按钮
        load_btn = QPushButton("加载点云文件")
        load_btn.setIcon(qta.icon('fa5s.folder-open', color='#555'))
        load_btn.clicked.connect(self.parent.open_pcd_file)
        file_layout.addWidget(load_btn)

        # 保存视图按钮
        save_view_btn = QPushButton("保存当前视图")
        save_view_btn.setIcon(qta.icon('fa5s.camera', color='#555'))
        save_view_btn.clicked.connect(self.parent.save_screenshot)
        file_layout.addWidget(save_view_btn)

        control_layout.addWidget(file_group)

        # --- 可视化设置区 ---
        viz_group = QGroupBox("可视化设置")
        viz_layout = QVBoxLayout(viz_group)

        # 点云颜色设置
        self.color_checkbox = QCheckBox("显示原始颜色")
        self.color_checkbox.setChecked(True)
        self.color_checkbox.stateChanged.connect(self.parent.update_visualization)
        viz_layout.addWidget(self.color_checkbox)

        # 点大小设置
        point_size_layout = QHBoxLayout()
        point_size_layout.addWidget(QLabel("点大小:"))
        self.point_size_spinner = QSpinBox()
        self.point_size_spinner.setRange(1, 10)
        self.point_size_spinner.setValue(2)
        self.point_size_spinner.valueChanged.connect(self.parent.update_visualization)
        point_size_layout.addWidget(self.point_size_spinner)
        viz_layout.addLayout(point_size_layout)

        # 渲染模式
        render_layout = QHBoxLayout()
        render_layout.addWidget(QLabel("渲染模式:"))
        self.render_mode = QComboBox()
        self.render_mode.addItems(["点", "网格"])
        self.render_mode.currentIndexChanged.connect(self.parent.update_visualization)
        render_layout.addWidget(self.render_mode)
        viz_layout.addLayout(render_layout)

        control_layout.addWidget(viz_group)

        # --- 切片设置区 ---
        slice_group = QGroupBox("切片设置")
        slice_layout = QVBoxLayout(slice_group)

        # 厚度设置
        thickness_layout = QHBoxLayout()
        thickness_layout.addWidget(QLabel("厚度:"))
        self.thickness_input = QLineEdit("0.1")
        self.thickness_input.setToolTip("相对于总高度的比例 (0-1)")
        self.thickness_input.textChanged.connect(self.parent.update_visualization)
        thickness_layout.addWidget(self.thickness_input)
        slice_layout.addLayout(thickness_layout)

        # 切片高度滑动条
        slice_height_layout = QVBoxLayout()
        slice_height_label = QLabel("切片高度:")
        slice_height_layout.addWidget(slice_height_label)

        self.slice_height_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_height_slider.setMinimum(0)
        self.slice_height_slider.setMaximum(100)
        self.slice_height_slider.setValue(0)
        self.slice_height_slider.valueChanged.connect(self.parent.update_visualization)
        slice_height_layout.addWidget(self.slice_height_slider)

        # 高度分布直方图
        self.height_dist_widget = HeightDistributionWidget()
        # Connect slider value changed signal to the widget's update slot (pass ratio 0-1)
        self.slice_height_slider.valueChanged.connect(
            lambda value: self.height_dist_widget.update_slider_ratio(value / 100.0)
        )
        slice_height_layout.addWidget(self.height_dist_widget)

        # 显示当前高度值
        self.height_value_label = QLabel("0.00")
        self.height_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slice_height_layout.addWidget(self.height_value_label)

        slice_layout.addLayout(slice_height_layout)
        control_layout.addWidget(slice_group)

        # 添加空间
        control_layout.addStretch()

        # 添加到标签页
        self.tab_widget.addTab(control_tab, "控制")

    def _create_info_tab(self):
        """创建信息面板标签页"""
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)

        # 点云信息显示区
        info_group = QGroupBox("点云信息")
        info_content_layout = QVBoxLayout(info_group)

        # 点数量
        self.points_count_label = QLabel("点数量: 0")
        info_content_layout.addWidget(self.points_count_label)

        # 边界信息
        self.bounds_label = QLabel("边界: [0, 0, 0] - [0, 0, 0]")
        self.bounds_label.setWordWrap(True)
        info_content_layout.addWidget(self.bounds_label)

        # 当前切片信息
        self.slice_info_label = QLabel("当前切片: 0 点")
        info_content_layout.addWidget(self.slice_info_label)

        info_layout.addWidget(info_group)
        info_layout.addStretch()

        # 添加到标签页
        self.tab_widget.addTab(info_tab, "信息")

    def update_info_panel(self, cloud_info, slice_info=None):
        """
        更新信息面板

        Args:
            cloud_info (dict): 点云信息
            slice_info (tuple, optional): 切片信息 (点数, 范围)
        """
        if cloud_info["point_count"] > 0:
            # 更新点数量
            self.points_count_label.setText(f"点数量: {cloud_info['point_count']}")

            # 更新边界信息
            bounds = cloud_info["bounds"]
            bounds_text = f"X: [{bounds[0]:.2f}, {bounds[1]:.2f}]\n"
            bounds_text += f"Y: [{bounds[2]:.2f}, {bounds[3]:.2f}]\n"
            bounds_text += f"Z: [{bounds[4]:.2f}, {bounds[5]:.2f}]"
            self.bounds_label.setText(bounds_text)

        # 更新切片信息
        if slice_info and slice_info[0] > 0:
            self.slice_info_label.setText(f"当前切片: {slice_info[0]} 点")
        else:
            self.slice_info_label.setText("当前切片: 0 点")

    def update_height_label(self, start_ratio, height_range):
        """
        更新高度标签

        Args:
            start_ratio (float): 起始高度比例
            height_range (tuple): 实际高度范围 (start, end)
        """
        self.height_value_label.setText(
            f"{start_ratio:.2f} ({height_range[0]:.2f} - {height_range[1]:.2f})"
        )