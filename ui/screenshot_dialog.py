from PyQt6.QtWidgets import (QDialog, QFormLayout, QDialogButtonBox,
                             QComboBox, QDoubleSpinBox, QCheckBox, QVBoxLayout)


class ScreenshotDialog(QDialog):
    """截图设置对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("保存视图设置")
        self.resize(350, 250)
        self.setup_ui()

    def setup_ui(self):
        """设置界面元素"""
        # 创建表单布局
        layout = QFormLayout(self)

        # 保存格式选择
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG (.png)", "JPEG (.jpg)", "TIFF (.tif)", "BMP (.bmp)"])
        layout.addRow("保存格式:", self.format_combo)

        # 图像质量设置 (JPEG等有损格式适用)
        self.quality_spin = QDoubleSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(90)
        self.quality_spin.setSuffix(" %")
        self.quality_spin.setEnabled(False)
        layout.addRow("图像质量(不可用):", self.quality_spin)

        # 图像分辨率设置
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["当前窗口大小", "720p (1280×720)",
                                        "1080p (1920×1080)", "4K (3840×2160)", "自定义"])
        layout.addRow("分辨率:", self.resolution_combo)

        # 保持横纵比设置
        self.keep_ratio = QCheckBox("保持当前窗口横纵比")
        self.keep_ratio.setChecked(True)
        layout.addRow("", self.keep_ratio)

        # 是否显示轴和网格
        self.show_axis = QCheckBox("显示坐标轴")
        self.show_axis.setChecked(False)
        layout.addRow("", self.show_axis)

        # 添加确定和取消按钮
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                           QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addRow(self.button_box)

    def get_settings(self):
        """
        获取用户设置

        Returns:
            dict: 包含所有截图设置的字典
        """
        # 获取文件格式
        format_text = self.format_combo.currentText()
        file_format = format_text.split("(")[1].strip(")").strip(".")

        # 获取分辨率设置
        resolution = self.resolution_combo.currentText()
        if resolution == "当前窗口大小":
            resolution = "current"
        elif resolution == "720p (1280×720)":
            resolution = (1280, 720)
        elif resolution == "1080p (1920×1080)":
            resolution = (1920, 1080)
        elif resolution == "4K (3840×2160)":
            resolution = (3840, 2160)
        else:
            resolution = "custom"  # 在实际应用中可以添加自定义尺寸的输入框

        # 返回所有设置
        return {
            "format": file_format,
            "quality": int(self.quality_spin.value()),
            "resolution": resolution,
            "keep_ratio": self.keep_ratio.isChecked(),
            "show_axis": self.show_axis.isChecked()
        }