"""
PCD Viewer - 3D点云可视化工具
具有直线检测拓展功能
"""

__version__ = "1.2.0"
__dependencies__ = [
    "PyQt6",
    "pyvista",
    "open3d",
    "numpy",
    "opencv-python",
    "qtawesome"
]

# Global Settings
DEBUG_MODE = False
BITMAP_EXPORT_RESOLUTION = (1024, 1024)
DEFAULT_DENSITY_RESOLUTION = 1024
RANDOM_SEED = 2025
