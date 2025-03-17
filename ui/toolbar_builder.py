from PyQt6.QtWidgets import QToolBar
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QAction
import qtawesome as qta


class ToolbarBuilder:
    """构建应用程序工具栏"""

    def __init__(self, parent):
        """
        初始化工具栏构建器

        Args:
            parent: 父窗口，用于连接信号和槽
        """
        self.parent = parent
        self.toolbar = None
        self.toggle_sidebar_action = None

    def build(self):
        """
        构建工具栏

        Returns:
            QToolBar: 构建好的工具栏
        """
        self.toolbar = QToolBar("主工具栏")
        self.toolbar.setIconSize(QSize(24, 24))
        self.toolbar.setStyleSheet("QToolBar { border-bottom: 1px solid #eeeeee; }")

        icon_color = "#555555"  # 深灰色图标更适合白色背景

        # 切换侧边栏按钮
        self.toggle_sidebar_action = QAction(qta.icon('fa5s.bars', color=icon_color), "切换侧边栏", self.parent)
        self.toggle_sidebar_action.triggered.connect(self.parent.toggle_sidebar)
        self.toolbar.addAction(self.toggle_sidebar_action)

        self.toolbar.addSeparator()

        # 加载文件按钮
        load_action = QAction(qta.icon('fa5s.file-import', color=icon_color), "加载点云", self.parent)
        load_action.triggered.connect(self.parent.open_pcd_file)
        self.toolbar.addAction(load_action)

        # 保存截图按钮
        screenshot_action = QAction(qta.icon('fa5s.camera', color=icon_color), "保存截图", self.parent)
        screenshot_action.triggered.connect(self.parent.save_screenshot)
        self.toolbar.addAction(screenshot_action)

        self.toolbar.addSeparator()

        # 重置视图按钮
        reset_view_action = QAction(qta.icon('fa5s.sync', color=icon_color), "重置视图", self.parent)
        reset_view_action.triggered.connect(self.parent.reset_view)
        self.toolbar.addAction(reset_view_action)

        # 顶视图按钮
        top_view_action = QAction(qta.icon('fa5s.arrow-down', color=icon_color), "顶视图", self.parent)
        top_view_action.triggered.connect(lambda: self.parent.plotter.view_xy())
        self.toolbar.addAction(top_view_action)

        return self.toolbar