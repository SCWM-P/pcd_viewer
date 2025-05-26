import json
import re
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
import traceback
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from .. import DEBUG_MODE
from .geometry_utils import calculate_global_xy_bounds
from .point_cloud_handler import PointCloudHandler


class SliceDataReader:
    """
    读取 pcd_viewer/batch_slice_output/ 目录下特定批次导出结果的工具类。
    """

    def __init__(self, export_batch_dir=Path(r"./batch_slice_output/")):
        self.batch_dir = Path(export_batch_dir)
        if not self.batch_dir.is_dir():
            raise FileNotFoundError(f"指定的导出目录不存在: {self.batch_dir}")

        self.global_params = None
        self.slice_data = {}
        if DEBUG_MODE: print(f"DEBUG: Initialized SliceDataReader for directory: {self.batch_dir}")

    def read_all(self):
        """
        读取目录下的所有相关文件并加载到内存中。
        """
        if DEBUG_MODE: print("DEBUG: Starting to read all data...")

        # 1. 读取全局参数
        global_params_file = self.batch_dir / "export_parameters.json"
        if global_params_file.exists():
            try:
                with open(global_params_file, 'r', encoding='utf-8') as f:
                    self.global_params = json.load(f)
                if DEBUG_MODE: print(f"DEBUG: Global parameters loaded: {self.global_params}")
            except Exception as e:
                print(f"Warning: 无法读取全局参数文件 {global_params_file}: {e}")
                self.global_params = {}  # Assign empty dict on failure
        else:
            print(f"Warning: 全局参数文件未找到: {global_params_file}")
            self.global_params = {}
        # 2. 查找所有 slice 相关文件
        metadata_files = sorted(self.batch_dir.glob("slice_*_metadata.json"))
        if not metadata_files:
            print(f"Warning: 在 {self.batch_dir} 中未找到元数据文件。")
            return False

        # 提取所有存在的 slice indices
        slice_indices = set()
        pattern = re.compile(r"slice_(\d+)_metadata\.json")
        for meta_file in metadata_files:
            match = pattern.search(meta_file.name)
            if match:
                slice_indices.add(int(match.group(1)))

        if not slice_indices:
            print(f"Warning: 未能从文件名中解析出有效的 slice indices。")
            return False

        sorted_indices = sorted(list(slice_indices))
        if DEBUG_MODE: print(f"DEBUG: Found {len(sorted_indices)} slice indices: {sorted_indices}")

        # 3. 逐个 slice 读取数据
        for index in sorted_indices:
            if DEBUG_MODE: print(f"--- Reading data for Slice {index} ---")
            slice_entry = {}
            # a) 读取 Metadata
            meta_file = self.batch_dir / f"slice_{index}_metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        slice_entry['metadata'] = json.load(f).get('metadata', {})  # Get nested metadata dict
                    if DEBUG_MODE: print(f"DEBUG: Loaded metadata for slice {index}.")
                except Exception as e:
                    print(f"Warning: 无法读取元数据文件 {meta_file}: {e}")
                    slice_entry['metadata'] = None  # Mark as None if failed
            else:
                if DEBUG_MODE: print(f"DEBUG: Metadata file not found for slice {index}.")
                slice_entry['metadata'] = None
            # b) 读取 Bitmap PNG
            bitmap_file = self.batch_dir / f"slice_{index}_bitmap.png"
            if bitmap_file.exists():
                try:
                    # Use OpenCV, which reads as BGR by default
                    bitmap_bgr = cv2.imread(str(bitmap_file))
                    if bitmap_bgr is not None:
                        # Convert to RGB for consistency if needed downstream
                        slice_entry['bitmap'] = cv2.cvtColor(bitmap_bgr, cv2.COLOR_BGR2RGB)
                        if DEBUG_MODE: print(
                            f"DEBUG: Loaded bitmap for slice {index}, shape: {slice_entry['bitmap'].shape}")
                    else:
                        print(f"Warning: OpenCV无法读取位图文件 {bitmap_file}")
                        slice_entry['bitmap'] = None
                except Exception as e:
                    print(f"Warning: 读取位图文件 {bitmap_file} 出错: {e}")
                    slice_entry['bitmap'] = None
            else:
                if DEBUG_MODE: print(f"DEBUG: Bitmap file not found for slice {index}.")
                slice_entry['bitmap'] = None

            # c) 读取 PCD 文件
            pcd_file = self.batch_dir / f"slice_{index}.pcd"
            if pcd_file.exists():
                try:
                    slice_entry['pcd'] = o3d.io.read_point_cloud(str(pcd_file))
                    if DEBUG_MODE: print(
                        f"DEBUG: Loaded PCD for slice {index}, points: {len(slice_entry['pcd'].points)}")
                except Exception as e:
                    print(f"Warning: 无法读取 PCD 文件 {pcd_file}: {e}")
                    slice_entry['pcd'] = None
            else:
                if DEBUG_MODE: print(f"DEBUG: PCD file not found for slice {index}.")
                slice_entry['pcd'] = None

            # d) 读取 Density Matrix (NPY)
            matrix_file = self.batch_dir / f"slice_{index}_density_matrix.npy"
            if matrix_file.exists():
                try:
                    slice_entry['density_matrix'] = np.load(str(matrix_file))
                    if DEBUG_MODE: print(
                        f"DEBUG: Loaded density matrix for slice {index}, shape: {slice_entry['density_matrix'].shape}")
                except Exception as e:
                    print(f"Warning: 无法读取密度矩阵文件 {matrix_file}: {e}")
                    slice_entry['density_matrix'] = None
            else:
                if DEBUG_MODE: print(f"DEBUG: Density matrix file not found for slice {index}.")
                slice_entry['density_matrix'] = None

            # e) 读取 Density Heatmap PNG
            heatmap_file = self.batch_dir / f"slice_{index}_density_heatmap.png"
            if heatmap_file.exists():
                try:
                    # Read as RGB directly if needed, or BGR and convert
                    heatmap_bgr = cv2.imread(str(heatmap_file))
                    if heatmap_bgr is not None:
                        slice_entry['density_heatmap'] = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
                        if DEBUG_MODE: print(
                            f"DEBUG: Loaded density heatmap for slice {index}, shape: {slice_entry['density_heatmap'].shape}")
                    else:
                        print(f"Warning: OpenCV无法读取密度热力图文件 {heatmap_file}")
                        slice_entry['density_heatmap'] = None
                except Exception as e:
                    print(f"Warning: 读取密度热力图文件 {heatmap_file} 出错: {e}")
                    slice_entry['density_heatmap'] = None
            else:
                if DEBUG_MODE: print(f"DEBUG: Density heatmap file not found for slice {index}.")
                slice_entry['density_heatmap'] = None

            # Store the collected data for this slice
            self.slice_data[index] = slice_entry

        if DEBUG_MODE: print(f"DEBUG: Finished reading data. Loaded data for {len(self.slice_data)} slices.")
        return True

    def get_slice(self, index):
        """获取指定索引的切片数据字典"""
        return self.slice_data.get(index)

    def get_all_indices(self):
        """获取所有成功加载数据的切片索引列表"""
        return sorted(list(self.slice_data.keys()))

    def get_pcd(self, index):
        """获取指定索引的 Open3D PointCloud 对象"""
        return self.slice_data.get(index, {}).get('pcd')

    def get_bitmap(self, index):
        """获取指定索引的位图 (RGB NumPy 数组)"""
        return self.slice_data.get(index, {}).get('bitmap')

    def get_metadata(self, index):
        """获取指定索引的元数据字典"""
        return self.slice_data.get(index, {}).get('metadata')

    def get_density_matrix(self, index):
        """获取指定索引的密度矩阵 (NumPy 数组)"""
        return self.slice_data.get(index, {}).get('density_matrix')

    def get_density_heatmap(self, index):
        """获取指定索引的密度热力图 (RGB NumPy 数组)"""
        return self.slice_data.get(index, {}).get('density_heatmap')


def render_slice_to_image(slice_data, size, overall_xy_bounds=None, is_thumbnail=True):
    if DEBUG_MODE: print(f"DEBUG: render_slice_to_image called. is_thumbnail={is_thumbnail}, size={size}")
    if slice_data is None or slice_data.n_points == 0: return None, {}
    plotter = None
    try:
        img_width, img_height = size if isinstance(size, tuple) else (size.width(), size.height())
        plotter = pv.Plotter(off_screen=True, window_size=[img_width, img_height])
        plotter.set_background('white')
        if is_thumbnail:
            actor = plotter.add_mesh(slice_data, color='darkgrey', point_size=1)
        else:
            if 'colors' in slice_data.point_data:
                actor = plotter.add_mesh(slice_data, scalars='colors', rgb=True, point_size=2)
            else:
                actor = plotter.add_mesh(slice_data, color='blue', point_size=2)
        plotter.view_xy()
        bounds_to_use = None
        if overall_xy_bounds:
            zmin = slice_data.bounds[4]
            zmax = slice_data.bounds[5]
            bounds_to_use = overall_xy_bounds + [zmin, zmax]
        elif slice_data and slice_data.bounds[0] < slice_data.bounds[1]:
            bounds_to_use = slice_data.bounds
        if bounds_to_use: plotter.reset_camera(bounds=bounds_to_use)
        img_np = plotter.screenshot(return_img=True)
        cam = plotter.camera
        view_params = {
            "position": list(cam.position), # 相机位置
            "focal_point": list(cam.focal_point),  # 焦点
            "up": list(cam.up),  # 视线方向
            "parallel_projection": cam.parallel_projection,  # 是否为平行投影
            "parallel_scale": cam.parallel_scale,  # 平行投影缩放比例
            "slice_bounds": list(slice_data.bounds),  # 切片边界
            "render_window_size": [img_width, img_height],  # 窗口大小
        }
        return img_np, view_params
    except Exception as e:
        print(f"ERROR: Error rendering slice: {e}")
        return None, {}
    finally:
        if plotter:
            try: plotter.close()
            except Exception: pass


def calculate_and_plot_density(
        pcd_file_path, grid_resolution=1024, colormap='viridis',
        output_density_matrix_path=None,
        show_plot=True
):
    """
    读取点云文件，计算XY平面投影密度，绘制热力图和密度分布柱状图。

    参数:
        pcd_file_path (str): 点云文件的路径。
        grid_resolution (int): 栅格化的分辨率。
        colormap (str): 用于热力图的matplotlib颜色映射名称。
        output_density_matrix_path (str, optional): 如果提供，则将对齐后的密度矩阵保存到此路径。
        show_plot (bool): 是否显示绘制的图形。

    返回:
        tuple: (aligned_density_matrix, xmin, xmax, ymin, ymax, h_min, h_max)
               如果发生错误则返回 (None, None, None, None, None, None, None)
    """
    # 1. 读取点云文件
    try:
        # PointCloudHandler.load_from_file 返回 pv.PolyData, bounds, count
        pcd_pv, _, pcd_num = PointCloudHandler.load_from_file(pcd_file_path)
        if pcd_pv is None or pcd_num == 0:
            print(f"错误: 文件 {pcd_file_path} 加载失败或不包含点。")
            return None, None, None, None, None, None, None
        if DEBUG_MODE: print(f"DEBUG: 点云文件加载成功，点数: {pcd_num}")
        points = pcd_pv.points  # pv.PolyData.points 是 NumPy 数组
    except Exception as e:
        print(f"错误: 无法读取或处理点云文件 {pcd_file_path}: {e}")
        if DEBUG_MODE: traceback.print_exc()
        return None, None, None, None, None, None, None

    # 2. 计算点云XY边界
    xy_bounds = calculate_global_xy_bounds(pcd_pv)  # Pass pv.PolyData object
    if xy_bounds is None:
        print("错误: 无法确定点云的XY边界。")
        return None, None, None, None, None, None, None
    xmin, xmax, ymin, ymax = xy_bounds
    print(f"计算得到的XY边界 (含填充): X=[{xmin:.2f}, {xmax:.2f}], Y=[{ymin:.2f}, {ymax:.2f}]")

    # 3. 栅格化并计算点云数量
    points_xy = points[:, 0:2]
    print(f"正在计算 {grid_resolution}x{grid_resolution} 的密度矩阵...")
    density_matrix_hist, x_edges, y_edges = np.histogram2d(
        points_xy[:, 0],
        points_xy[:, 1],
        bins=[grid_resolution, grid_resolution],
        range=[[xmin, xmax], [ymin, ymax]]
    )
    aligned_density_matrix = np.flipud(density_matrix_hist.T)

    print("密度矩阵计算完成。")
    h_min_val = np.min(aligned_density_matrix)
    h_max_val = np.max(aligned_density_matrix)
    if show_plot:
        # 4. 绘制 matplotlib 密度热力图和密度分布柱状图
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        # --- 子图1: 密度热力图 ---
        ax_heatmap = axes[0]
        im = ax_heatmap.imshow(
            aligned_density_matrix,
            cmap=colormap,
            extent=[xmin, xmax, ymin, ymax],
            origin='lower',
            aspect='equal'
        )
        cbar = fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
        cbar.set_label('点密度 (数量/格)')
        ax_heatmap.set_title(f'XY平面密度热力图 ({grid_resolution}x{grid_resolution} 栅格)\n{Path(pcd_file_path).name}')
        ax_heatmap.set_xlabel('X 坐标 (米)')
        ax_heatmap.set_ylabel('Y 坐标 (米)')
        # --- 子图2: 密度分布柱状图 ---
        ax_hist = axes[1]
        density_values_flat = aligned_density_matrix.flatten()
        # 过滤掉密度为0的格子，使柱状图更能反映非空区域的分布（可选）
        density_values_flat_positive = density_values_flat[density_values_flat > 0]
        if len(density_values_flat_positive) > 0:
            num_hist_bins = max(30, int(grid_resolution / 5))  # 至少10个bins
            if DEBUG_MODE: print(f"DEBUG: 密度分布柱状图的 bins 数量: {num_hist_bins}")
            # 使用 np.histogram 计算数据，然后用 ax.bar 绘制，可以更好地控制外观
            counts, bin_edges_hist = np.histogram(density_values_flat_positive, bins=num_hist_bins)
            bin_centers = (bin_edges_hist[:-1] + bin_edges_hist[1:]) / 2
            ax_hist.bar(
                bin_centers, counts, width=(bin_edges_hist[1] - bin_edges_hist[0]) * 0.9,
                color='skyblue', edgecolor='black'
            )
            ax_hist.set_title('非零格点密度分布')
            ax_hist.set_xlabel('每个格子的点数 (密度)')
            ax_hist.set_ylabel('格子数量 (频数)')
            ax_hist.grid(True, linestyle=':', alpha=0.7)
            # ax_hist.set_yscale('log') # 如果密度差异很大，对数刻度可能更好
        else:
            ax_hist.text(0.5, 0.5, '所有格子密度均为0', ha='center', va='center')
            ax_hist.set_title('非零格点密度分布')

        plt.tight_layout()
        plt.show()

    if output_density_matrix_path:
        try:
            np.save(output_density_matrix_path, aligned_density_matrix)
            print(f"对齐后的密度矩阵已保存到: {output_density_matrix_path}")
        except Exception as e:
            print(f"错误: 保存密度矩阵失败: {e}")

    return aligned_density_matrix, xmin, xmax, ymin, ymax, h_min_val, h_max_val


def create_density_heatmap(density_matrix, colormap_name='viridis', vmin=None, vmax=None):
    """
    Generates an RGBA NumPy array heatmap from a 2D density matrix.
    参数:
        density_matrix (np.ndarray): 2D 密度矩阵。
        colormap_name (str): Matplotlib 颜色映射的名称。
        vmin (float, optional): 颜色映射的最小值。如果为 None，则使用矩阵的最小值
        vmax (float, optional): 颜色映射的最大值。如果为 None，则使用矩阵的最大值
    返回:
        np.ndarray or None: RGBA 热力图图像 (uint8 NumPy 数组, HxWx4) 或在错误时返回 None。
    """
    if density_matrix is None or density_matrix.size == 0:
        if DEBUG_MODE: print("DEBUG: create_density_heatmap - 输入密度矩阵为空。")
        return None
    try:
        # 1. 获取颜色映射对象
        cmap = plt.get_cmap(colormap_name)
        # 2. 归一化密度矩阵
        current_vmin = np.min(density_matrix) if vmin is None else vmin
        current_vmax = np.max(density_matrix) if vmax is None else vmax
        assert current_vmin < current_vmax, "颜色映射的最小值必须小于最大值。"
        norm = mcolors.Normalize(vmin=current_vmin, vmax=current_vmax, clip=True)
        normalized_matrix = norm(density_matrix)
        # 3. 应用颜色映射得到 RGBA 数组 (范围 0-1 float)
        colored_matrix_float = cmap(normalized_matrix)
        # 4. 转换为 uint8 数组 (范围 0-255)
        heatmap_rgba_array = (colored_matrix_float * 255).astype(np.uint8)
        if DEBUG_MODE: print(f"DEBUG: create_density_heatmap - 生成热力图数组，形状: {heatmap_rgba_array.shape}")
        return heatmap_rgba_array
    except Exception as e:
        print(f"ERROR: 创建密度热力图时失败: {e}")
        if DEBUG_MODE: traceback.print_exc()
        return None
