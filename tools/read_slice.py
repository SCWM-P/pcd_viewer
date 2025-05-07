# pcd_viewer/tools/read_slice.py

import os
import json
import cv2
import open3d as o3d
import numpy as np
import glob
import re
import traceback
from pathlib import Path
from .. import DEBUG_MODE # Import global debug flag


class SliceDataReader:
    """
    读取 pcd_viewer/batch_slice_output/ 目录下特定批次导出结果的工具类。
    """

    def __init__(self, export_batch_dir=Path(r"./batch_slice_output/")):
        """
        初始化读取器。

        Args:
            export_batch_dir (str or Path): 指向特定批次导出结果的目录路径
                                            (例如 'pcd_viewer/batch_slice_output/batch_slice_export_20250506_162755')
        """
        self.batch_dir = Path(export_batch_dir)
        if not self.batch_dir.is_dir():
            raise FileNotFoundError(f"指定的导出目录不存在: {self.batch_dir}")

        self.global_params = None
        self.slice_data = {}
        # 字典，键为 slice_index (int)
        # 值为另一个字典: {
        # 'metadata': dict, 'bitmap': np.ndarray, 'pcd': o3d.PointCloud,
        # 'density_matrix': np.ndarray, 'density_heatmap': np.ndarray
        # }
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
                self.global_params = {} # Assign empty dict on failure
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
                        slice_entry['metadata'] = json.load(f).get('metadata', {}) # Get nested metadata dict
                    if DEBUG_MODE: print(f"DEBUG: Loaded metadata for slice {index}.")
                except Exception as e:
                    print(f"Warning: 无法读取元数据文件 {meta_file}: {e}")
                    slice_entry['metadata'] = None # Mark as None if failed
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
                        if DEBUG_MODE: print(f"DEBUG: Loaded bitmap for slice {index}, shape: {slice_entry['bitmap'].shape}")
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
                    if DEBUG_MODE: print(f"DEBUG: Loaded PCD for slice {index}, points: {len(slice_entry['pcd'].points)}")
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
                    if DEBUG_MODE: print(f"DEBUG: Loaded density matrix for slice {index}, shape: {slice_entry['density_matrix'].shape}")
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
                        if DEBUG_MODE: print(f"DEBUG: Loaded density heatmap for slice {index}, shape: {slice_entry['density_heatmap'].shape}")
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

# --- 示例用法 ---
if __name__ == '__main__':
    # 设置为 True 以启用此脚本中的调试输出
    # 注意: 这会覆盖从 __init__ 导入的 DEBUG_MODE (如果它为 False)
    DEBUG_MODE = True

    # 查找最新的导出目录 (示例逻辑)
    output_base_dir = Path(__file__).parent.parent / "batch_slice_output" # Go up one level from tools
    try:
        latest_export_dir = max(output_base_dir.glob("batch_slice_export_*"), key=os.path.getmtime)
        print(f"找到最新的导出目录: {latest_export_dir}")

        # 初始化并读取数据
        reader = SliceDataReader(latest_export_dir)
        if reader.read_all():
            print("\n--- 数据读取成功 ---")
            print(f"全局参数: {reader.global_params}")
            loaded_indices = reader.get_all_indices()
            print(f"已加载切片索引: {loaded_indices}")

            if loaded_indices:
                # 访问第一个切片的数据示例
                first_index = loaded_indices[0]
                print(f"\n--- 第一个切片 (索引 {first_index}) 数据示例 ---")
                metadata = reader.get_metadata(first_index)
                if metadata:
                    print(f"  元数据 (高度范围): {metadata.get('height_range')}")
                    # print(f"  元数据 (视图参数): {metadata.get('view_params')}") # 可能很长
                else:
                    print("  元数据: 未加载")

                bitmap = reader.get_bitmap(first_index)
                if bitmap is not None:
                    print(f"  位图形状: {bitmap.shape}")
                    # cv2.imshow("Bitmap Example", cv2.cvtColor(bitmap, cv2.COLOR_RGB2BGR)) # Example display
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                else:
                    print("  位图: 未加载")

                pcd = reader.get_pcd(first_index)
                if pcd:
                    print(f"  PCD 点数: {len(pcd.points)}")
                    if pcd.has_colors(): print("  PCD 包含颜色")
                else:
                    print("  PCD: 未加载")

                density_matrix = reader.get_density_matrix(first_index)
                if density_matrix is not None:
                    print(f"  密度矩阵形状: {density_matrix.shape}")
                    print(f"  密度矩阵最大值: {np.max(density_matrix)}")
                else:
                    print("  密度矩阵: 未加载")

                density_heatmap = reader.get_density_heatmap(first_index)
                if density_heatmap is not None:
                    print(f"  密度热力图形状: {density_heatmap.shape}")
                else:
                    print("  密度热力图: 未加载")

        else:
            print("数据读取失败或未找到文件。")

    except ValueError:
        print(f"错误: 在 {output_base_dir} 中未找到任何 'batch_slice_export_*' 目录。")
    except FileNotFoundError as fnf_err:
        print(f"错误: {fnf_err}")
    except Exception as e:
        print(f"发生意外错误: {e}")
        traceback.print_exc()