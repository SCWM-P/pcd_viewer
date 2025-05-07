import open3d as o3d
import numpy as np
import pyvista as pv
import os
import pandas as pd
import traceback
from PyQt6.QtCore import QThread, pyqtSignal
from .. import DEBUG_MODE


class PointCloudHandler:
    """处理点云数据的加载、切片和基本操作"""

    @staticmethod
    def load_from_file(file_path):
        """
        从文件加载点云
        Args:
            file_path (str): 点云文件的路径
        Returns:
            tuple: (pv.PolyData, 边界, 点数量)
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        try:
            points = None
            colors = None
            point_cloud = None
            if file_ext in ['.pcd', '.ply', '.pts']:  # Original formats
                pcd = o3d.io.read_point_cloud(file_path)
                if not pcd.has_points():
                    print(f"Warning: 文件不包含点: {file_path}")
                    return None, None, 0
                points = np.asarray(pcd.points)
                if pcd.has_colors():
                    colors = np.asarray(pcd.colors)

            elif file_ext in ['.txt', '.xyz']:
                if DEBUG_MODE: print(f"DEBUG: Loading TXT file: {file_path}")
                try:
                    df = pd.read_csv(
                        file_path, sep=r'\s+', header=None, usecols=[0, 1, 2, 3, 4, 5],
                        names=['x', 'y', 'z', 'r', 'g', 'b'],
                        on_bad_lines='warn',
                        engine='python',
                        comment='#'
                    )

                    df.dropna(inplace=True)
                    if df.empty:
                        print(f"Warning: TXT 文件解析后为空或无有效数据行: {file_path}")
                        return None, None, 0

                    points = df[['x', 'y', 'z']].values.astype(np.float64)
                    colors_raw = df[['r', 'g', 'b']].values

                    # Check if colors look like 0-1 float or 0-255 int
                    if np.any(colors_raw > 1.0):  # Heuristic: if any value > 1, assume 0-255
                        if DEBUG_MODE: print("DEBUG: Assuming TXT colors are 0-255, converting to 0-1 float.")
                        colors = colors_raw.astype(np.float64) / 255.0
                    else:
                        if DEBUG_MODE: print("DEBUG: Assuming TXT colors are already 0-1 float.")
                        colors = colors_raw.astype(np.float64)
                    colors = np.clip(colors, 0.0, 1.0)
                    if DEBUG_MODE: print(f"DEBUG: Loaded {len(points)} points from TXT.")

                except pd.errors.EmptyDataError:
                    print(f"Warning: TXT 文件是空的: {file_path}")
                    return None, None, 0
                except ValueError as ve:
                    # Catch errors like wrong number of columns if usecols fails etc.
                    print(f"ERROR: 解析 TXT 文件时值错误 (可能列数不匹配或类型错误): {ve} in {file_path}")
                    if DEBUG_MODE: traceback.print_exc()
                    raise RuntimeError(f"解析 TXT 文件值错误: {ve}")
                except Exception as parse_err:
                    print(f"ERROR: 解析 TXT 文件时发生未知错误: {parse_err} in {file_path}")
                    if DEBUG_MODE: traceback.print_exc()
                    raise RuntimeError(f"解析 TXT 文件出错: {parse_err}")

            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")

            # Create PyVista PolyData
            if points is not None and len(points) > 0:
                point_cloud = pv.PolyData(points)
                if colors is not None:
                    # Ensure colors array shape matches points array length
                    if len(colors) == len(points):
                        point_cloud['colors'] = colors
                    else:
                        print(
                            f"Warning: 颜色数据长度 ({len(colors)}) 与点数据长度 ({len(points)}) 不匹配 in {file_path}. Ignoring colors.")

                # Check for NaN/inf in points which can cause issues later
                if np.any(~np.isfinite(points)):
                    print(f"Warning: 点数据包含 NaN 或 Inf 值 in {file_path}. 可能导致后续操作失败。")
                    # Optionally, filter them out here, but be careful about color correspondence
                    finite_mask = np.all(np.isfinite(points), axis=1)
                    points = points[finite_mask]
                    if colors is not None and len(colors) == len(finite_mask): # Check original length
                        colors = colors[finite_mask]
                    point_cloud = pv.PolyData(points)
                    if colors is not None: point_cloud['colors'] = colors

                return point_cloud, point_cloud.bounds, len(points)
            else:
                # Handle case where loading resulted in no valid points
                return None, None, 0

        except Exception as e:
            print(f"ERROR: 加载点云文件 '{file_path}' 出错: {e}")
            if DEBUG_MODE: traceback.print_exc()
            raise RuntimeError(f"加载点云文件出错: {e}")

    @staticmethod
    def slice_by_height(point_cloud, height_ratio, thickness_ratio):
        """
        根据高度比例和厚度比例对点云进行切片

        Args:
            point_cloud (pv.PolyData): 原始点云数据
            height_ratio (float): 起始高度比例 (0-1)
            thickness_ratio (float): 厚度比例 (相对于总高度)

        Returns:
            tuple: (切片后的点云, 切片点数量, 实际高度范围)
        """
        if point_cloud is None:
            return None, 0, (0, 0)

        bounds = point_cloud.bounds
        min_z, max_z = bounds[4], bounds[5]
        total_height = max_z - min_z

        start_height = min_z + total_height * height_ratio
        thickness = total_height * thickness_ratio
        end_height = start_height + thickness

        points = point_cloud.points
        sliced_points_indices = np.where(
            (points[:, 2] >= start_height) &
            (points[:, 2] <= end_height)
        )[0]

        if len(sliced_points_indices) == 0:
            return None, 0, (start_height, end_height)

        sliced_points = points[sliced_points_indices]
        sliced_cloud = pv.PolyData(sliced_points)

        if 'colors' in point_cloud.point_data:
            sliced_colors = point_cloud['colors'][sliced_points_indices]
            sliced_cloud['colors'] = sliced_colors

        return sliced_cloud, len(sliced_points), (start_height, end_height)

    @staticmethod
    def get_cloud_info(point_cloud):
        """
        获取点云的基本信息

        Args:
            point_cloud (pv.PolyData): 点云数据

        Returns:
            dict: 点云信息字典
        """
        if point_cloud is None:
            return {
                "point_count": 0,
                "bounds": None,
                "has_colors": False
            }

        bounds = point_cloud.bounds
        info = {
            "point_count": len(point_cloud.points),
            "bounds": bounds,
            "has_colors": 'colors' in point_cloud.point_data
        }
        return info

class LoadPointCloudThread(QThread):
    finished_loading = pyqtSignal(object, object, int, str) # pv.PolyData, bounds_tuple, count, filename
    error_occurred = pyqtSignal(str, str) # error_message, filename

    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self._is_running = True

    def run(self):
        if DEBUG_MODE: print(f"DEBUG: LoadPointCloudThread started for {self.file_path}")
        try:
            point_cloud_pv, bounds, point_count = PointCloudHandler.load_from_file(self.file_path)

            if not self._is_running: # Check if cancelled during loading
                if DEBUG_MODE: print(f"DEBUG: LoadPointCloudThread for {self.file_path} cancelled during load.")
                self.error_occurred.emit("加载已取消。", os.path.basename(self.file_path))
                return

            if point_cloud_pv is not None:
                if DEBUG_MODE: print(f"DEBUG: LoadPointCloudThread successfully loaded {self.file_path}")
                self.finished_loading.emit(point_cloud_pv, bounds, point_count, os.path.basename(self.file_path))
            else:
                if DEBUG_MODE: print(f"DEBUG: LoadPointCloudThread: PointCloudHandler.load_from_file returned no data for {self.file_path}")
                self.error_occurred.emit(f"无法加载文件或文件为空: {os.path.basename(self.file_path)}", os.path.basename(self.file_path))

        except RuntimeError as e: # Catch RuntimeErrors raised by load_from_file for more severe issues
            if DEBUG_MODE: print(f"ERROR: LoadPointCloudThread: RuntimeError during loading {self.file_path}: {e}")
            if self._is_running: self.error_occurred.emit(str(e), os.path.basename(self.file_path))
        except Exception as e:
            if DEBUG_MODE:
                print(f"ERROR: LoadPointCloudThread: Unhandled exception during loading {self.file_path}: {e}")
                traceback.print_exc()
            if self._is_running:
                 self.error_occurred.emit(f"加载时发生未知错误: {e}", os.path.basename(self.file_path))

    def stop(self):
        self._is_running = False
        if DEBUG_MODE: print(f"DEBUG: LoadPointCloudThread for {self.file_path} stop requested.")
