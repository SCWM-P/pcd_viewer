import open3d as o3d
import numpy as np
import pyvista as pv


def load_point_cloud(file_path):
    """加载点云文件并返回 PyVista PolyData 对象"""
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        colors = None
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)

        point_cloud = pv.PolyData(points, faces=None)
        if colors is not None:
            point_cloud['colors'] = colors

        return point_cloud, len(points)
    except Exception as e:
        raise RuntimeError(f"加载点云文件出错: {str(e)}")


def slice_cloud_by_height(point_cloud, min_z, max_z, start_ratio, thickness_ratio):
    """根据高度参数对点云进行切片"""
    if point_cloud is None:
        return None, 0

    bounds = point_cloud.bounds
    z_min, z_max = bounds[4], bounds[5]
    total_height = z_max - z_min

    start_height = z_min + total_height * start_ratio
    thickness = total_height * thickness_ratio
    end_height = start_height + thickness

    points = point_cloud.points
    sliced_points_indices = np.where((points[:, 2] >= start_height) & (points[:, 2] <= end_height))[0]
    sliced_points = points[sliced_points_indices]

    sliced_cloud = pv.PolyData(sliced_points)

    if 'colors' in point_cloud.point_data:
        sliced_colors = point_cloud['colors'][sliced_points_indices]
        sliced_cloud['colors'] = sliced_colors

    return sliced_cloud, len(sliced_points)