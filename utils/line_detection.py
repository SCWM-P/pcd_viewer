import numpy as np
import cv2
from abc import ABC, abstractmethod


class LineDetector(ABC):
    """直线检测算法的抽象基类"""

    @abstractmethod
    def detect(self, image):
        """
        检测图像中的直线

        Args:
            image: 灰度图像

        Returns:
            lines: 检测到的直线列表
        """
        pass

    @abstractmethod
    def get_params(self):
        """
        获取算法参数

        Returns:
            dict: 参数字典
        """
        pass

    @abstractmethod
    def set_params(self, params):
        """
        设置算法参数

        Args:
            params (dict): 参数字典
        """
        pass


class HoughLineDetector(LineDetector):
    """基于霍夫变换的直线检测"""

    def __init__(self):
        # 默认参数
        self.rho = 1  # 距离分辨率（像素）
        self.theta = np.pi / 180  # 角度分辨率（弧度）
        self.threshold = 100  # 阈值参数，满足此阈值的才被认为是直线
        self.min_line_length = 50  # 最小线段长度
        self.max_line_gap = 10  # 允许将同一条直线上的点连接起来的最大间隔

    def detect(self, image):
        """
        使用霍夫变换检测直线

        Args:
            image: 灰度图像

        Returns:
            lines: 检测到的直线数组 [x1, y1, x2, y2]
        """
        # 确保输入是灰度图像
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(
            edges,
            self.rho,
            self.theta,
            self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        if lines is None:
            return []

        # 将检测结果转换为 [x1, y1, x2, y2] 格式
        return [line[0] for line in lines]

    def get_params(self):
        return {
            "rho": self.rho,
            "theta": self.theta,
            "threshold": self.threshold,
            "min_line_length": self.min_line_length,
            "max_line_gap": self.max_line_gap
        }

    def set_params(self, params):
        if "rho" in params:
            self.rho = params["rho"]
        if "theta" in params:
            self.theta = params["theta"]
        if "threshold" in params:
            self.threshold = params["threshold"]
        if "min_line_length" in params:
            self.min_line_length = params["min_line_length"]
        if "max_line_gap" in params:
            self.max_line_gap = params["max_line_gap"]


class RANSACLineDetector(LineDetector):
    """基于RANSAC算法的直线检测"""

    def __init__(self):
        # 默认参数
        self.min_samples = 2  # 拟合模型所需的最小样本数
        self.residual_threshold = 2.0  # 判断数据点是否符合模型的阈值
        self.max_trials = 1000  # 最大试验次数

    def detect(self, image):
        """
        使用RANSAC检测直线

        Args:
            image: 灰度图像

        Returns:
            lines: 检测到的直线列表 [x1, y1, x2, y2]
        """
        try:
            # 此实现需要scikit-image
            from skimage import feature, measure, color
            from skimage.transform import probabilistic_hough_line

            # 确保输入是灰度图像
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # 边缘检测
            edges = feature.canny(gray, sigma=2)

            # 使用概率霍夫变换获取线段
            lines = probabilistic_hough_line(edges,
                                             line_length=10,
                                             line_gap=3)

            # 转换为 [x1, y1, x2, y2] 格式
            result_lines = []
            for line in lines:
                result_lines.append([line[0][0], line[0][1], line[1][0], line[1][1]])

            return result_lines
        except ImportError:
            # 如果没有scikit-image，使用OpenCV实现
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            return self._detect_lines_cv(edges)

    def _detect_lines_cv(self, edges):
        """使用OpenCV和轮廓分析来检测直线"""
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        lines = []

        for contour in contours:
            if len(contour) > 5:  # 至少需要5个点来拟合
                # 使用最小二乘法拟合直线
                [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

                # 获取轮廓的边界框
                x_min = np.min(contour[:, :, 0])
                x_max = np.max(contour[:, :, 0])

                # 计算线段的端点
                if abs(vx) > 1e-10:  # 避免除以零
                    y1 = int(((x_min - x) * vy / vx) + y)
                    y2 = int(((x_max - x) * vy / vx) + y)
                    lines.append([int(x_min), y1, int(x_max), y2])

        return lines

    def get_params(self):
        return {
            "min_samples": self.min_samples,
            "residual_threshold": self.residual_threshold,
            "max_trials": self.max_trials
        }

    def set_params(self, params):
        if "min_samples" in params:
            self.min_samples = params["min_samples"]
        if "residual_threshold" in params:
            self.residual_threshold = params["residual_threshold"]
        if "max_trials" in params:
            self.max_trials = params["max_trials"]


def draw_detected_lines(image, lines, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制检测到的直线

    Args:
        image: 输入图像（RGB或BGR）
        lines: 检测到的直线列表，每条线为 [x1, y1, x2, y2] 格式
        color: 线条颜色，默认为绿色
        thickness: 线条粗细

    Returns:
        image_with_lines: 绘制了直线的图像
    """
    # 创建副本以免修改原图
    image_with_lines = image.copy()

    # 绘制每条直线
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(image_with_lines, (x1, y1), (x2, y2), color, thickness)

    return image_with_lines


class LineDetectionManager:
    """管理不同的直线检测算法"""

    def __init__(self):
        self.detectors = {
            "Hough": HoughLineDetector(),
            "RANSAC": RANSACLineDetector()
        }
        self.current_detector = "Hough"

    def register_detector(self, name, detector):
        """
        注册新的直线检测器

        Args:
            name (str): 检测器名称
            detector (LineDetector): 检测器实例
        """
        if not isinstance(detector, LineDetector):
            raise TypeError("检测器必须继承自LineDetector类")

        self.detectors[name] = detector

    def get_detector_names(self):
        """
        获取所有注册的检测器名称

        Returns:
            list: 检测器名称列表
        """
        return list(self.detectors.keys())

    def set_current_detector(self, name):
        """
        设置当前使用的检测器

        Args:
            name (str): 检测器名称
        """
        if name not in self.detectors:
            raise ValueError(f"未注册的检测器: {name}")

        self.current_detector = name

    def get_current_detector(self):
        """
        获取当前检测器

        Returns:
            LineDetector: 当前检测器实例
        """
        return self.detectors[self.current_detector]

    def detect_lines(self, image):
        """
        使用当前检测器检测直线

        Args:
            image: 输入图像

        Returns:
            lines: 检测到的直线列表
        """
        return self.detectors[self.current_detector].detect(image)