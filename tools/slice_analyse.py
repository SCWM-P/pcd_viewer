# pcd_viewer/tools/slice_analyse.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import traceback
from pathlib import Path
import json
import sys
import cv2
import os
from tqdm import tqdm
matplotlib.use('QtAgg')

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
project_root_parent = project_root.parent
sys.path.append(str(project_root_parent))
from pcd_viewer.utils.geometry_utils import calculate_global_xy_bounds
from pcd_viewer.utils.point_cloud_handler import PointCloudHandler
from pcd_viewer.utils.slice_handler import calculate_and_plot_density, SliceDataReader, create_density_heatmap
from pcd_viewer import DEBUG_MODE

plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False


def probabilistic_morphology_op(
        binary_slice_image, density_matrix,
        kernel, operation_type,
        activation_func=lambda x: x ** 3,
        sensitivity=2.0, bias=1.0,
):
    """
    对二值切片图像执行基于密度的概率形态学操作。

    参数:
        binary_slice_image (np.ndarray): 输入的二值化切片位图 (0为背景, 非0为前景)。
        aligned_density_matrix (np.ndarray): 与切片位图对齐的整体密度矩阵。
        den_min (float): 整体密度矩阵中的最小密度值 (用于归一化)。
        den_max (float): 整体密度矩阵中的最大密度值 (用于归一化)。
        kernel (np.ndarray): 形态学操作的核 (结构元素, e.g., 3x3 全1矩阵)，例如：
            十字形核: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            矩形核: np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        operation_type (str): 'erode' 或 'dilate'。
        activation_func (callable): 密度归一化后的激活函数 p(x)。

    返回:
        np.ndarray: 处理后的二值图像。
    """
    # 预计算归一化分母，避免重复计算和除零
    den_max, den_min = np.max(density_matrix), np.min(density_matrix)
    density_range = den_max - den_min
    if DEBUG_MODE:
        print(f"DEBUG: Probabilistic morphology: op={operation_type}, kernel_shape={kernel.shape}")
        print(f"DEBUG: Density range for normalization: min={den_min}, max={den_max}")

    # 确保输入是二值图像 (0 和 255)
    _, S_prime = cv2.threshold(binary_slice_image, 254, 255, cv2.THRESH_BINARY)
    output_image = S_prime.copy()
    rows, cols = S_prime.shape
    k_rows, k_cols = kernel.shape
    assert k_rows % 2 == 1 and k_cols % 2 == 1, "核长宽必须是奇数"
    assert operation_type in ['erode', 'dilate'], "操作类型必须是 'erode' 或 'dilate'"
    assert binary_slice_image.shape == density_matrix.shape, "切片图像和密度矩阵尺寸不匹配"
    k_center_r, k_center_c = k_rows // 2, k_cols // 2
    if density_range < 1e-6: density_range = 1.0
    probability_matrix = np.zeros_like(density_matrix, dtype=float)
    # 遍历图像中的每个像素 (核的中心)
    for r_idx in tqdm(range(rows)):
        for c_idx in range(cols):
            sum_activated_density = 0.0
            overlapping_foreground_pixels = 0
            # 将核应用于当前位置
            for kr_idx in range(k_rows):
                for kc_idx in range(k_cols):
                    if kernel[kr_idx, kc_idx] == 0:  # 只考虑核中为1的部分
                        continue
                    # 计算当前核元素对应的图像像素坐标
                    img_r = r_idx + kr_idx - k_center_r
                    img_c = c_idx + kc_idx - k_center_c

                    # 检查是否在图像边界内
                    if 0 <= img_r < rows and 0 <= img_c < cols:
                        # 检查该像素在原始二值切片中是否为前景
                        if S_prime[img_r, img_c] < 255:  # 前景像素
                            overlapping_foreground_pixels += 1
                            # 获取对应位置的密度值并归一化密度并应用激活函数
                            density_val = density_matrix[img_r, img_c]
                            normalized_density = np.clip((density_val - den_min) / density_range, 0.0, 1.0)
                            sum_activated_density += activation_func(normalized_density)
            if overlapping_foreground_pixels == 0:  # 无重叠前景像素
                probability_matrix[r_idx, c_idx] = 0.0
                continue
            probability_matrix[r_idx, c_idx] = (np.tanh(sensitivity * (sum_activated_density - bias)) + 1) / 2
            assert 0.0 <= probability_matrix[
                r_idx, c_idx] <= 1.0, f"probability[{r_idx}, {c_idx}] = {probability_matrix[r_idx, c_idx]} 超出范围[0,1] !"
    # 根据概率执行操作
    random_throw = np.random.rand(rows, cols)
    if operation_type == 'erode':
        erode_mask = (S_prime < 255) & (random_throw < (1.0 - probability_matrix))
        output_image[erode_mask] = 255
    elif operation_type == 'dilate':
        dilate_mask = random_throw < probability_matrix
        output_image[dilate_mask] = 0
    return output_image


# --- 辅助函数用于可视化 ---
def display_images(image_dict, main_title="图像处理流程"):
    """使用 Matplotlib 显示多张图像"""
    num_images = len(image_dict)
    if num_images == 0:
        return
    # 动态计算行列数，尽量保持方形
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()  # 将 axes 数组展平，方便索引
    for i, (title, img) in enumerate(image_dict.items()):
        if img is None:
            axes[i].text(0.5, 0.5, '图像为空', ha='center', va='center')
            axes[i].set_title(title)
            axes[i].axis('off')
            continue
        if len(img.shape) == 2:  # 灰度图或二值图
            axes[i].imshow(img, cmap='gray')
        elif len(img.shape) == 3:  # 彩色图 (假设是RGB)
            axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(main_title, fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # 调整布局以适应主标题
    plt.show()


# --- 主工作流程函数 ---
def run_slice_analysis(
        batch_export_dir_path,
        original_pcd_path,
        slice_index_to_process,
        output_resolution: int = 1024,
        density_colormap='viridis',
        morph_kernel_size: [int, list] = 5,
        morph_operations=None,  # 例如: [('erode', 1), ('dilate', 1)] 表示先腐蚀1次，再膨胀1次
        prob_morph_sensitivity=2.0,
        prob_morph_bias=1.0
):
    """
    执行完整的切片分析工作流程。
    """
    global DEBUG_MODE  # Allow modification if script-level arg is passed
    if DEBUG_MODE:
        print("--- 开始切片分析工作流程 ---")
        print(f"批处理导出目录: {batch_export_dir_path}")
        print(f"原始点云路径: {original_pcd_path}")
        print(f"目标切片索引: {slice_index_to_process}")
        print(f"输出分辨率: {output_resolution}x{output_resolution}")

    # --- 1. 使用 SliceDataReader 读取指定批次导出的数据 ---
    if not batch_export_dir_path:
        print("错误: 未提供批处理导出目录路径。")
        return
    try:
        reader = SliceDataReader(batch_export_dir_path)
        if not reader.read_all():
            print(f"错误: 无法从目录 {batch_export_dir_path} 读取数据。")
            return
    except FileNotFoundError:
        print(f"错误: 批处理导出目录 '{batch_export_dir_path}' 未找到。")
        return
    except Exception as e:
        print(f"读取批处理数据时出错: {e}")
        if DEBUG_MODE: traceback.print_exc()
        return

    # 获取要处理的切片位图
    target_slice_bitmap_rgb = reader.get_bitmap(slice_index_to_process)
    if target_slice_bitmap_rgb is None:
        print(f"错误: 在导出的数据中未找到索引为 {slice_index_to_process} 的位图。")
        print(f"可用切片索引: {reader.get_all_indices()}")
        return

    # 将读取的RGB位图转换为BGR以匹配OpenCV的常见输入，再转灰度
    target_slice_bitmap_bgr = cv2.cvtColor(target_slice_bitmap_rgb, cv2.COLOR_RGB2BGR)
    slice_gray = cv2.cvtColor(target_slice_bitmap_bgr, cv2.COLOR_BGR2GRAY)
    _, slice_binary_original = cv2.threshold(slice_gray, 254, 255, cv2.THRESH_BINARY)
    images_to_display = {"原始切片位图 (二值化)": slice_binary_original.copy()}
    if DEBUG_MODE: print(f"成功加载切片 {slice_index_to_process} 的位图，形状: {target_slice_bitmap_rgb.shape}")

    # --- 2. 加载原始点云PCD，计算整体密度信息 ---
    print(f"\n--- 正在从 '{original_pcd_path}' 加载原始点云以计算整体密度 ---")
    try:
        # 使用 PointCloudHandler 加载，它返回 pv.PolyData
        full_pcd_pv, _, num_points = PointCloudHandler.load_from_file(original_pcd_path)
        if full_pcd_pv is None or num_points == 0:
            print(f"错误: 无法加载原始点云或点云为空: {original_pcd_path}")
            return
        if DEBUG_MODE: print(f"原始点云加载成功，点数: {num_points}")
    except Exception as e:
        print(f"加载原始点云 '{original_pcd_path}' 时出错: {e}")
        if DEBUG_MODE: traceback.print_exc()
        return

    # 计算全局XY边界 (这将是所有栅格化操作的统一参考)
    global_xy_bounds = calculate_global_xy_bounds(full_pcd_pv)
    if global_xy_bounds is None:
        print("错误: 无法计算原始点云的全局XY边界。")
        return
    xmin, xmax, ymin, ymax = global_xy_bounds
    if DEBUG_MODE: print(f"计算得到的全局XY边界: X=[{xmin:.2f}, {xmax:.2f}], Y=[{ymin:.2f}, {ymax:.2f}]")

    # 计算整体密度矩阵
    overall_density_matrix_hist, _, _ = np.histogram2d(
        full_pcd_pv.points[:, 0], full_pcd_pv.points[:, 1],
        bins=[output_resolution, output_resolution],
        range=[[xmin, xmax], [ymin, ymax]]
    )
    # 对齐方向 (0,0 左上, Y向下)
    overall_aligned_density_matrix = np.flipud(overall_density_matrix_hist.T)
    den_min_overall = np.min(overall_aligned_density_matrix)
    den_max_overall = np.max(overall_aligned_density_matrix)
    print(f"{output_resolution}x{output_resolution} 的整体密度矩阵计算完成。")

    # 可视化整体密度热力图
    overall_density_heatmap = create_density_heatmap(
        overall_aligned_density_matrix, colormap_name=density_colormap,
        vmin=den_min_overall, vmax=den_max_overall
    )
    if overall_density_heatmap is not None:
        images_to_display["整体密度热力图"] = overall_density_heatmap  # imshow可以直接显示RGBA
        # 如果要用 OpenCV 保存，需要转换
        # cv2.imwrite("overall_density_heatmap.png", cv2.cvtColor(overall_density_heatmap_rgba, cv2.COLOR_RGBA2BGRA))

    # --- 3. 对指定切片投影位图做基于密度和概率的形态学操作 ---
    # 确保切片位图与密度图尺寸一致
    if slice_binary_original.shape != (output_resolution, output_resolution):
        print(
            f"警告: 切片位图尺寸 {slice_binary_original.shape} 与期望分辨率 {(output_resolution, output_resolution)} 不匹配。")
        print("将对切片位图进行缩放以匹配密度矩阵。")
        slice_binary_resized = cv2.resize(
            slice_binary_original,
            (output_resolution, output_resolution),
            interpolation=cv2.INTER_NEAREST
        )
        images_to_display["原始切片位图 (缩放并二值化)"] = slice_binary_resized.copy()
        current_processed_image = slice_binary_resized.copy()
    else:
        current_processed_image = slice_binary_original.copy()

    # --- 4. 可控的多轮形态学操作框架 ---
    if morph_operations is None:
        morph_operations = [
            {'type': 'erode', 'rounds': 1, 'kernel_size': morph_kernel_size,
             'sensitivity': prob_morph_sensitivity, 'bias': prob_morph_bias,
             'activation': 'x_cubed'},
            {'type': 'dilate', 'rounds': 1, 'kernel_size': morph_kernel_size,
             'sensitivity': prob_morph_sensitivity, 'bias': prob_morph_bias,
             'activation': 'x_cubed'}
        ]

    print("\n--- 开始多轮概率形态学操作 ---")
    activation_functions = {
        'x_cubed': lambda x: x ** 3,
        'linear': lambda x: x,
        'x_squared': lambda x: x ** 2,
    }

    for i, op_details in enumerate(morph_operations):
        op_type = op_details['type']
        rounds = op_details.get('rounds', 1)
        k_size = op_details.get('kernel_size', 3)
        sensitivity = op_details.get('sensitivity', 2.0)
        bias = op_details.get('bias', 1.0)
        activation_str = op_details.get('activation', 'x_cubed')
        activation_func = activation_functions.get(activation_str, lambda x: x ** 3)

        kernel = np.ones((k_size, k_size), np.uint8)  # Simple square kernel
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size,k_size)) # Elliptical

        print(
            f"执行第 {i + 1} 组操作: {op_type.capitalize()} {rounds} 轮,"
            f"Kernel: {k_size}x{k_size}, Sens: {sensitivity}, Bias: {bias}, Act: {activation_str}"
        )

        for r in range(rounds):
            if DEBUG_MODE: print(f"轮次 {r + 1}/{rounds}...")
            current_processed_image = probabilistic_morphology_op(
                current_processed_image,  # 输入是上一轮的结果
                overall_aligned_density_matrix,  # 始终使用整体密度
                kernel,
                operation_type=op_type,
                activation_func=activation_func,
                sensitivity=sensitivity,
                bias=bias
            )
            # 可选：在每轮后显示中间结果
            if DEBUG_MODE and rounds > 1:
                images_to_display[f"操作组{i + 1}-{op_type}-轮次{r + 1}"] = current_processed_image.copy()
        images_to_display[f"操作组{i + 1}后 ({op_type.capitalize()} {rounds}轮)"] = current_processed_image.copy()
    print("概率形态学操作完成。")

    # --- 5. 可视化 ---
    final_result_overlay = cv2.cvtColor(current_processed_image, cv2.COLOR_GRAY2BGR)
    # 将原始彩色切片位图的颜色应用到处理后的结构上
    # Create a mask from the processed image (where structure is white)
    structure_mask = current_processed_image > 0
    # Resize original RGB bitmap if it was resized earlier
    if target_slice_bitmap_rgb.shape[0:2] != (output_resolution, output_resolution):
        original_rgb_resized = cv2.resize(
            target_slice_bitmap_rgb,
            (output_resolution, output_resolution),
            interpolation=cv2.INTER_LINEAR
        )
    else:
        original_rgb_resized = target_slice_bitmap_rgb

    final_result_overlay[structure_mask] = original_rgb_resized[structure_mask]
    images_to_display["最终结果 (叠加原图颜色)"] = final_result_overlay

    display_images(images_to_display, f"切片 {slice_index_to_process} 分析结果")
    # Optionally save the final image
    if args.output_image_path:
        Path.mkdir(Path(args.output_image_path).parent, exist_ok=True)
        cv2.imwrite(args.output_image_path, cv2.cvtColor(final_result_overlay, cv2.COLOR_RGB2BGR))
        print(f"\n执行完成！结果图像已保存到 {args.output_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成点云XY平面密度热力图。")
    parser.add_argument(
        "--pcd_file", type=str, default=project_root / "samples" / "one_floor.pcd",
        help="输入的点云文件路径 (.pcd, .ply, .txt, .xyz等Open3D支持的格式)"
    )
    parser.add_argument(
        "--batch_base_dir", type=str,
        help="包含 'batch_slice_export_*' 子目录的父目录路径，将自动查找最新的导出批次。",
        default="../batch_slice_output"
    )
    parser.add_argument(
        "--batch_dir", type=str,
        help="直接指定 'batch_slice_export_YYYYMMDD_HHMMSS' 目录的路径。如果提供，则忽略 --batch_dir。"
    )
    parser.add_argument(
        "--slice_index", type=int, default=None,
        help="要处理的切片索引 (从0开始)"
    )
    parser.add_argument(
        "--resolution", type=int, default=1024,
        help="密度图的栅格分辨率 (例如: 512 表示 512x512)，默认: 1024"
    )
    parser.add_argument(
        "--config_file", type=str, default=None,
        help="可选的JSON配置文件路径，用于指定形态学操作序列和参数。"
    )
    parser.add_argument(
        "--cmap", type=str, default="viridis",
        help="Matplotlib 颜色映射方案 (例如: viridis, plasma, inferno, magma, cividis, jet, hot), 默认: viridis"
    )
    parser.add_argument(
        "--output_image_path", type=str, default= project_root / "img_output" / "prob_morph_final_output.png",
        help="最终处理结果图像的保存路径。"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="启用详细的调试输出。"
    )
    args = parser.parse_args()
    if args.debug: DEBUG_MODE = True
    assert Path(args.pcd_file).exists(), f"错误:输入文件不存在 -> {args.pcd_file}"

    # --- 确定要处理的批处理导出目录 ---
    target_batch_dir = None
    if args.batch_dir:
        target_batch_dir = Path(args.batch_dir)
        assert target_batch_dir.is_dir(), f"错误:指定的批处理目录不存在 -> {target_batch_dir}"
    elif args.batch_base_dir:
        output_base_dir = Path(args.batch_base_dir)
        assert output_base_dir.is_dir(), f"错误: 基础批处理目录不存在 -> {output_base_dir}"
        try:
            latest_export_dir = max(output_base_dir.glob("batch_slice_export_*"), key=os.path.getmtime, default=None)
            if latest_export_dir:
                target_batch_dir = latest_export_dir
                print(f"自动找到最新的批处理导出目录: {target_batch_dir}")
            else:
                print(f"错误: 在 {output_base_dir} 中未找到任何 'batch_slice_export_*' 目录。");
                sys.exit(1)
        except ValueError:  # handles empty glob result for max
            print(f"错误: 在 {output_base_dir} 中未找到任何 'batch_slice_export_*' 目录。")
            sys.exit(1)
        except FileNotFoundError as fnf_err:
            print(f"错误: {fnf_err}")
            sys.exit(1)
        except Exception as e:
            print(f"发生意外错误: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("错误: 请通过 --batch_dir 或 --specific_batch_dir 指定批处理导出目录。")
        sys.exit(1)

    # --- 确定原始点云文件路径 ---
    original_pcd_for_density = args.pcd_file
    if not original_pcd_for_density:
        # 尝试从 SliceDataReader 的 global_params 获取
        try:
            temp_reader = SliceDataReader(target_batch_dir)
            temp_reader.read_all()  # Need to read global params
            if temp_reader.global_params and 'original_point_cloud_source' in temp_reader.global_params:
                source_name = temp_reader.global_params['original_point_cloud_source']
                potential_paths = [project_root / "samples" / source_name, Path(source_name)]
                for p_path in potential_paths:
                    if p_path.exists(): original_pcd_for_density = str(p_path); break
                if not original_pcd_for_density:
                    print(f"警告: 在导出参数中找到源文件名 '{source_name}' 但无法定位该文件。请通过 --pcd_file 指定。")
            else:
                print("警告: 未在导出参数中找到原始点云源信息。请通过 --pcd_file 指定。")
        except Exception as e_reader:
            print(f"尝试从导出参数读取原始PCD路径时出错: {e_reader}")

    if not original_pcd_for_density or not Path(original_pcd_for_density).exists():
        print(f"错误: 无法确定或找到原始点云文件 '{original_pcd_for_density}' 用于密度计算。请使用 --pcd_file 参数指定。")
        sys.exit(1)

    # --- 加载形态学操作配置 ---
    morph_ops_list = None
    prob_sens = 2.0
    prob_bias = 1.0
    kernel_sz = 5

    if args.config_file:
        if Path(args.config_file).exists():
            try:
                with open(args.config_file, 'r', encoding='utf-8') as f:
                    op_config = json.load(f)
                morph_ops_list = op_config.get("probabilistic_morphology_sequence")
                # Allow overriding default sensitivity, bias, kernel from config
                prob_sens = op_config.get("default_prob_sensitivity", prob_sens)
                prob_bias = op_config.get("default_prob_bias", prob_bias)
                kernel_sz = op_config.get("default_prob_kernel_size", kernel_sz)
                if DEBUG_MODE: print(f"从配置文件加载形态学操作序列: {morph_ops_list}")
            except Exception as e_cfg:
                print(f"警告: 读取或解析形态学配置文件 '{args.config_file}' 失败: {e_cfg}")
        else:
            print(f"警告: 指定的形态学配置文件 '{args.config_file}' 未找到，将使用默认操作。")

    if morph_ops_list is None:  # Fallback to default if not loaded from config
        morph_ops_list = [
            {'type': 'erode', 'rounds': 1, 'kernel_size': kernel_sz, 'sensitivity': prob_sens, 'bias': prob_bias,
             'activation': 'x_cubed'},
            {'type': 'dilate', 'rounds': 1, 'kernel_size': kernel_sz, 'sensitivity': prob_sens, 'bias': prob_bias,
             'activation': 'x_cubed'}
        ]
        if DEBUG_MODE: print("使用默认形态学操作序列。")

    run_slice_analysis(
        target_batch_dir,
        original_pcd_for_density,
        args.slice_index,
        output_resolution=args.resolution,
        density_colormap=args.cmap,
        morph_kernel_size=kernel_sz,  # This is now a default, individual ops can override
        morph_operations=morph_ops_list,
        prob_morph_sensitivity=prob_sens,  # Default, can be overridden by sequence
        prob_morph_bias=prob_bias  # Default, can be overridden by sequence
    )
    # calculate_and_plot_density(args.pcd_file, args.resolution, args.cmap)
