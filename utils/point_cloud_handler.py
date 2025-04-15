import open3d as o3d
import numpy as np
import pyvista as pv
import os


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
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            colors = None
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)

            point_cloud = pv.PolyData(points, faces=None)
            if colors is not None:
                point_cloud['colors'] = colors

            return point_cloud, point_cloud.bounds, len(points)
        except Exception as e:
            raise RuntimeError(f"加载点云文件出错: {str(e)}")

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