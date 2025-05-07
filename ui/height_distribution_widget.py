# pcd_viewer/ui/height_distribution_widget.py (或放在 sidebar_builder.py)

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPolygonF
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, pyqtSlot
import numpy as np


class HeightDistributionWidget(QWidget):
    """绘制点云高度分布直方图的Widget"""
    DEFAULT_BINS = 500  # 默认直方图桶数

    def __init__(self, parent=None):
        super().__init__(parent)
        self.histogram_data = None  # (bin_edges, hist_counts)
        self.normalized_counts = None
        self.max_count = 1
        self.setMinimumHeight(100)  # 设置一个合适的高度
        self.current_slider_ratio = 0.0  # 0.0 to 1.0
        self.current_thickness_ratio = 0.01  # 0.0 to 1.0

        # Tooltip to show density at cursor
        self.setMouseTracking(True)
        self.tooltip_label = QLabel(self)
        self.tooltip_label.setVisible(False)
        self.tooltip_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 180);
            color: white;
            padding: 3px;
            border-radius: 3px;
        """)
        self.tooltip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_histogram_data(self, points_z):
        """根据Z坐标计算并设置直方图数据"""
        if points_z is None or len(points_z) == 0:
            self.histogram_data = None
            self.normalized_counts = None
            self.max_count = 1
            self.update()
            return

        min_z = np.min(points_z)
        max_z = np.max(points_z)
        if max_z == min_z:  # Avoid histogram error for flat cloud
            max_z += 1e-6

        try:
            hist_counts, bin_edges = np.histogram(points_z, bins=self.DEFAULT_BINS, range=(min_z, max_z))
            self.histogram_data = (bin_edges, hist_counts)
            self.max_count = np.max(hist_counts) if np.max(hist_counts) > 0 else 1
            # Normalize counts to 0-1 range for drawing
            self.normalized_counts = hist_counts / self.max_count
            self.update()  # Trigger repaint
        except Exception as e:
            print(f"Error calculating histogram: {e}")
            self.histogram_data = None
            self.normalized_counts = None

    @pyqtSlot(float)
    def update_slider_ratio(self, ratio):
        """更新当前滑动条位置比例 (0.0 to 1.0)"""
        self.current_slider_ratio = ratio
        self.update()  # Trigger repaint to update indicator

    @pyqtSlot(float)
    def update_thickness_ratio(self, ratio):
        """更新当前厚度比例 (0.0 to 1.0)"""
        self.current_thickness_ratio = np.clip(ratio, 0.0, 1.0)
        self.update()  # Trigger repaint to update thickness indicator

    def paintEvent(self, event):
        """绘制直方图和指示器"""
        super().paintEvent(event)
        if self.normalized_counts is None or self.histogram_data is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        num_bins = len(self.normalized_counts)
        bar_width = width / num_bins

        # --- Draw Histogram Bars ---
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(180, 180, 220, 180))  # Light purple-blue bars

        for i, norm_count in enumerate(self.normalized_counts):
            bar_height = norm_count * (height - 2)  # Leave 1px padding top/bottom
            bar_x = i * bar_width
            painter.drawRect(
                int(bar_x), int(height - bar_height - 1),
                int(bar_width + 1), int(bar_height)
            )  # Add 1 to width to avoid gaps

        # --- Draw Thickness Indicator ---
        # Thickness ratio relative to number of bins
        thickness_bins = self.current_thickness_ratio * num_bins
        thickness_width_px = thickness_bins * bar_width

        slider_pos_x_start = self.current_slider_ratio * width
        indicator_start_x = slider_pos_x_start
        indicator_end_x = slider_pos_x_start + thickness_width_px
        indicator_end_x = min(indicator_end_x, width)  # Clamp to widget width

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 150, 255, 90))  # Semi-transparent blue for thickness
        painter.drawRect(int(indicator_start_x), 0, int(indicator_end_x - indicator_start_x), height)

        # --- Draw Current Position Line ---
        slider_pos_x = self.current_slider_ratio * width
        painter.setPen(QPen(QColor(255, 0, 0, 200), 1.5))  # Red indicator line
        painter.drawLine(int(slider_pos_x), 0, int(slider_pos_x), height)

    def mouseMoveEvent(self, event):
        if self.histogram_data is not None and self.normalized_counts is not None:
            pos = event.position()
            width = self.width()
            num_bins = len(self.normalized_counts)
            bin_index = int((pos.x() / width) * num_bins)
            bin_index = max(0, min(bin_index, num_bins - 1))  # Clamp index

            bin_start_z, bin_end_z = self.histogram_data[0][bin_index], self.histogram_data[0][bin_index + 1]
            count = self.histogram_data[1][bin_index]
            density_percent = (count / self.max_count) * 100 if self.max_count > 0 else 0

            tooltip_text = f"高度: {bin_start_z:.2f}-{bin_end_z:.2f}\n点数: {count} ({density_percent:.1f}%)"
            self.tooltip_label.setText(tooltip_text)
            self.tooltip_label.adjustSize()

            # Position tooltip near cursor, avoiding going off-screen
            tip_x = int(pos.x() + 10)
            tip_y = int(pos.y() - self.tooltip_label.height() - 5)
            if tip_x + self.tooltip_label.width() > width:
                tip_x = int(pos.x() - self.tooltip_label.width() - 10)
            if tip_y < 0:
                tip_y = int(pos.y() + 15)

            self.tooltip_label.move(tip_x, tip_y)
            self.tooltip_label.setVisible(True)
        else:
            self.tooltip_label.setVisible(False)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.tooltip_label.setVisible(False)
        super().leaveEvent(event)
