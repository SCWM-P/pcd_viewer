# pcd_viewer/tools/line_detect_beta.py

import cv2
import numpy as np
import json
import argparse
import os
import sys
from pathlib import Path
import math

# 确保可以导入上层目录的 read_slice 和 pcd_viewer 包
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from utils.slice_handler import SliceDataReader

# --- 可选: 如果需要骨架化等，导入 skimage ---
try:
    from skimage.morphology import skeletonize

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not found. Skeletonize and RANSAC (skimage) will not be available.")

try:
    from sklearn.linear_model import RANSACRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not found. RANSAC (sklearn) will not be available.")

# --- 全局调试开关 (可以被配置文件覆盖) ---
DEBUG_VERBOSE = False


# --- 辅助函数 ---
def display_image(title, image, config):
    """显示图像并在按下任意键后关闭"""
    if config['global'].get('show_intermediate_steps', False) or title == "Final Result":
        cv2.imshow(title, image)
        cv2.waitKey(config['visualization'].get('wait_key_delay', 0))
        if config['visualization'].get('wait_key_delay', 0) == 0:  # Only destroy if waiting indefinitely
            cv2.destroyWindow(title)


def draw_lines_on_image(image, lines, color=(0, 255, 0), thickness=2):
    """在图像上绘制检测到的直线"""
    output_image = image.copy()
    if lines is not None:
        for line in lines:
            if len(line) == 1:  # Standard HoughLines returns [[rho, theta]]
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(output_image, (x1, y1), (x2, y2), color, thickness)
            elif len(line) == 4:  # HoughLinesP, LSD, etc. return [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, line)  # Ensure integers
                cv2.line(output_image, (x1, y1), (x2, y2), color, thickness)
            elif isinstance(line, (list, np.ndarray)) \
                    and len(line) > 0 \
                    and isinstance(line[0], (list, np.ndarray)) \
                    and len(line[0]) == 4:  # For LSD returning list of lists
                for seg in line:
                    x1, y1, x2, y2 = map(int, seg)
                    cv2.line(output_image, (x1, y1), (x2, y2), color, thickness)

    return output_image


# --- 预处理函数 ---
def preprocess_image(image, config_preprocess):
    """应用配置的预处理步骤"""
    processed_image = image.copy()
    if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed_image  # Assume already grayscale if not 3-channel

    if DEBUG_VERBOSE: display_image("Grayscale", gray, {"global": {"show_intermediate_steps": True},
                                                        "visualization": {"wait_key_delay": 1}})

    # 使用高斯模糊
    if config_preprocess.get('enable_gaussian_blur', False):
        ksize = tuple(config_preprocess['gaussian_kernel_size'])
        sigma_x = config_preprocess['gaussian_sigma_x']
        gray = cv2.GaussianBlur(gray, ksize, sigma_x)
        if DEBUG_VERBOSE: display_image("Gaussian Blur", gray, {"global": {"show_intermediate_steps": True},
                                                                "visualization": {"wait_key_delay": 1}})

    # 使用中值模糊
    if config_preprocess.get('enable_median_blur', False):
        ksize = config_preprocess['median_kernel_size']
        gray = cv2.medianBlur(gray, ksize)
        if DEBUG_VERBOSE: display_image("Median Blur", gray, {"global": {"show_intermediate_steps": True},
                                                              "visualization": {"wait_key_delay": 1}})

    # 使用Canny边缘检测
    edges = None
    if config_preprocess.get('enable_canny', False):
        t1 = config_preprocess['canny_threshold1']
        t2 = config_preprocess['canny_threshold2']
        ap_size = config_preprocess['canny_aperture_size']
        edges = cv2.Canny(gray, t1, t2, apertureSize=ap_size)
        if DEBUG_VERBOSE: display_image("Canny Edges", edges, {"global": {"show_intermediate_steps": True},
                                                               "visualization": {"wait_key_delay": 1}})
    else:
        # If Canny is not used, some algorithms might need a binarized image
        # For now, assume algorithms handle grayscale or edges are explicitly enabled
        pass

    binary_image = edges if edges is not None else gray  # Use Canny edges if available, else grayscale for morph

    # 使用形态学闭运算
    if config_preprocess.get('enable_morph_close', False):
        ksize = tuple(config_preprocess['morph_close_kernel_size'])
        iterations = config_preprocess['morph_close_iterations']
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        if DEBUG_VERBOSE: display_image("Morphological Close", binary_image,
                                        {"global": {"show_intermediate_steps": True},
                                         "visualization": {"wait_key_delay": 1}})

    # 使用形态学开运算
    if config_preprocess.get('enable_morph_open', False):
        ksize = tuple(config_preprocess['morph_open_kernel_size'])
        iterations = config_preprocess['morph_open_iterations']
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        if DEBUG_VERBOSE: display_image("Morphological Open", binary_image,
                                        {"global": {"show_intermediate_steps": True},
                                         "visualization": {"wait_key_delay": 1}})

    binary_image = cv2.dilate(
        binary_image, iterations=1,
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    if DEBUG_VERBOSE:
        display_image(
            "Dilated Binary Image", binary_image,
            {
                "global": {"show_intermediate_steps": True},
                "visualization": {"wait_key_delay": 1}
            }
        )

    # 使用骨架化
    if config_preprocess.get('enable_skeletonize', False):
        if SKIMAGE_AVAILABLE:
            # Skeletonize expects boolean image
            binary_for_skeleton = binary_image > 0  # Convert to boolean
            skeleton = skeletonize(binary_for_skeleton)
            binary_image = (skeleton * 255).astype(np.uint8)
            if DEBUG_VERBOSE: display_image("Skeletonized", binary_image, {"global": {"show_intermediate_steps": True},
                                                                           "visualization": {"wait_key_delay": 1}})
        else:
            print("Warning: Skeletonize enabled but scikit-image not available. Skipping.")

    # Return the image in the format most algorithms expect
    # (e.g., Canny edges if enabled, otherwise grayscale/morphed)
    if edges is not None:
        # If Canny was run, subsequent morph ops were on 'edges' (renamed to binary_image)
        return binary_image  # This is the Canny edges possibly after morph ops
    return gray  # Return original gray if no Canny, or modified gray if only blur was applied


# --- 直线检测算法 ---
def detect_lines_houghp(image, params):
    """使用概率霍夫变换检测直线"""
    # HoughLinesP needs an 8-bit single-channel image (edges)
    if len(image.shape) == 3:
        raise ValueError("HoughLinesP requires a single-channel image (e.g., Canny edges).")

    lines = cv2.HoughLinesP(
        image,
        params['rho'],
        params['theta_degrees'] * np.pi / 180,
        params['threshold'],
        minLineLength=params['min_line_length'],
        maxLineGap=params['max_line_gap']
    )
    return [line[0] for line in lines] if lines is not None else []


def detect_lines_lsd(image, params):
    """使用LSD检测直线"""
    # LSD can work on grayscale directly
    gray_image = image
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2 and image.dtype != np.uint8:  # Ensure uint8 for LSD
        gray_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    lsd = cv2.createLineSegmentDetector(
        refine=params.get('refine', cv2.LSD_REFINE_STD),  # Default to STD if not specified
        scale=params.get('scale', 0.8),
        sigma_scale=params.get('sigma_scale', 0.6),
        quant=params.get('quant', 2.0),
        ang_th=params.get('ang_th', 22.5),
        log_eps=params.get('log_eps', 0.0),  # Typically 0 for no log_eps output
        density_th=params.get('density_th', 0.7),
        n_bins=params.get('n_bins', 1024)
    )
    lines, width, prec, nfa = lsd.detect(gray_image)
    return [line[0] for line in lines] if lines is not None else []


def detect_lines_ransac(image, params):
    """使用RANSAC拟合直线 (基于边缘点)"""
    if not SKIMAGE_AVAILABLE or not SKLEARN_AVAILABLE:
        print("Error: RANSAC detection requires scikit-image and scikit-learn. Skipping.")
        return []

    # RANSAC works on a set of points.
    # If 'image' is an edge map, extract points. Otherwise, this needs rethink.
    if len(image.shape) == 3 or np.max(image) <= 1:  # If it's not a binary edge map
        print("Warning: RANSAC here expects a binary edge map. Trying to use non-zero points.")
        # Fallback: treat non-zero pixels as points if not an edge map
        # This might be very slow for grayscale images.
        # It's better to ensure Canny or other edge detection is run before RANSAC
        # if the input 'image' to this function is from preprocess_image.

    points = np.argwhere(image > 0)  # Get (row, col) of edge pixels
    if len(points) < params['min_samples']:
        print("Warning: Not enough points for RANSAC.")
        return []

    # RANSAC expects (y, x) from argwhere, but fitting is often (x, y)
    # For LineModelND, it treats columns as features.
    # Here, points are [row, col], so effectively y, x.
    # We want to fit x as a function of y, or y as a function of x.
    # Skimage RANSAC with LineModelND fits lines in N-D.
    # For 2D lines, we can fit y = mx + c (or x = my + c for vertical lines).

    from skimage.measure import LineModelND, ransac

    detected_lines = []
    remaining_points = points.copy()
    num_lines_to_find = params.get("num_lines_to_find", 5)  # Example: find up to 5 lines

    for _ in range(num_lines_to_find):
        if len(remaining_points) < params['min_samples']:
            break

        # Fit model
        # data should be (N, D) where D is dimensionality (2 for 2D lines)
        # Points from argwhere are (N, 2) with [row, col]
        try:
            model_robust, inliers = ransac(
                remaining_points,
                LineModelND,
                min_samples=params['min_samples'],
                residual_threshold=params['residual_threshold'],
                max_trials=params['max_trials'],
                stop_probability=params.get('stop_probability', 0.99)  # skimage uses this
            )
        except ValueError as e:  # Can happen if too few unique points for min_samples
            print(f"RANSAC ValueError: {e}. Skipping this line.")
            break
        except Exception as e:
            print(f"RANSAC unexpected error: {e}")
            break

        if inliers is None \
                or sum(inliers) < params.get(
            'stop_n_inliers',
            params['min_samples']
        ):  # stop_n_inliers for sklearn
            # print("RANSAC: Not enough inliers found for a robust line.")
            break  # Stop if not enough inliers

        inlier_points = remaining_points[inliers]

        # Extract line segment from inlier points (e.g., min/max x and y)
        # LineModelND gives origin and direction. We need to project inliers onto it.
        # For simplicity, just take the bounding box of inliers for now.
        # This is a simplification; proper line segment extraction from RANSAC inliers is more involved.
        y_coords, x_coords = inlier_points[:, 0], inlier_points[:, 1]
        # Fit a line to inlier_points to get endpoints easily (e.g. using PCA or min/max projection)
        if len(y_coords) > 1:
            # Get two extreme points from the inliers to define the segment
            # This is a very naive way, proper projection onto the fitted line is better.
            min_idx = np.argmin(y_coords)  # Topmost
            max_idx = np.argmax(y_coords)  # Bottommost
            pt1 = (x_coords[min_idx], y_coords[min_idx])
            pt2 = (x_coords[max_idx], y_coords[max_idx])
            detected_lines.append([pt1[0], pt1[1], pt2[0], pt2[1]])

        # Remove inliers for next iteration
        remaining_points = remaining_points[~inliers]

    return detected_lines


def detect_lines_contours(image, params):
    """使用轮廓和多边形逼近检测直线"""
    # findContours needs a binary image
    if len(image.shape) == 3 or np.max(image) <= 1:  # Check if not binary
        print("Warning: Contours method expects a binary edge map.")
        # Attempt to binarize if it's grayscale
        if len(image.shape) == 2 and image.dtype == np.uint8:
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        else:
            return []

    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < params.get('min_contour_area', 10):  # Filter small contours
            continue
        epsilon = params['approx_poly_epsilon_factor'] * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # approx is a set of vertices, e.g., [[[x1,y1]], [[x2,y2]], ...]
        # Extract line segments from the polygon
        for i in range(len(approx) - 1):
            pt1 = approx[i][0]
            pt2 = approx[i + 1][0]
            if np.linalg.norm(pt1 - pt2) >= params.get('min_line_segment_length', 10):
                lines.append([pt1[0], pt1[1], pt2[0], pt2[1]])
        # Also connect the last point to the first if it's a closed polygon (usually is from findContours)
        if len(approx) > 1:
            pt1 = approx[-1][0]
            pt2 = approx[0][0]
            if np.linalg.norm(pt1 - pt2) >= params.get('min_line_segment_length', 10):
                lines.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    return lines


# Placeholder for EDLines if you integrate an external library
# def detect_lines_edlines(image, params):
#     print("EDLines not implemented in this script yet.")
#     return []

ALGORITHM_FUNCTIONS = {
    "HoughP": detect_lines_houghp,
    "LSD": detect_lines_lsd,
    "RANSAC": detect_lines_ransac,
    "Contours": detect_lines_contours,
    # "EDLines": detect_lines_edlines,
}


# --- 主逻辑 ---
def main(config_path):
    global DEBUG_VERBOSE
    """主执行函数"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件未找到 {config_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: 配置文件格式错误 {config_path}")
        return

    DEBUG_VERBOSE = config['global'].get('show_intermediate_steps', False)  # Set based on config

    # 1. 加载数据
    export_dir = config['global'].get('export_batch_dir')
    slice_idx = config['global'].get('slice_index_to_process', 0)

    if not export_dir:
        print("错误: 配置文件中未指定 'export_batch_dir'。")
        return

    try:
        reader = SliceDataReader(export_dir)
        if not reader.read_all():  # read_all now returns bool
            print(f"错误: 无法从目录 {export_dir} 读取数据。")
            return
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

    original_bitmap = reader.get_bitmap(slice_idx)
    if original_bitmap is None:
        print(f"错误: 无法获取索引为 {slice_idx} 的位图。可用索引: {reader.get_all_indices()}")
        return

    print(f"成功加载切片 {slice_idx} 的位图，形状: {original_bitmap.shape}")
    display_image("Original Bitmap", original_bitmap, config)

    # 2. 预处理
    print("开始预处理...")
    # Convert original_bitmap (RGB) to BGR for OpenCV processing if needed
    # Most OpenCV functions expect BGR if color, or single channel.
    # Our preprocess_image handles conversion to grayscale internally.
    image_for_preprocess = original_bitmap.copy()
    preprocessed_image = preprocess_image(image_for_preprocess, config['preprocessing'])
    print("预处理完成。")
    display_image("Preprocessed Image for Detection", preprocessed_image, config)

    # 3. 直线检测
    algo_name = config.get('detection_algorithm', 'HoughP')
    algo_params = config.get('algorithms', {}).get(algo_name, {})
    print(f"开始使用算法 '{algo_name}' 进行直线检测...")

    if algo_name not in ALGORITHM_FUNCTIONS:
        print(f"错误: 未知的检测算法 '{algo_name}'。可用算法: {list(ALGORITHM_FUNCTIONS.keys())}")
        return

    # Prepare input for the algorithm based on its needs
    detection_input_image = preprocessed_image  # Default to output of preprocess_image
    if algo_name == "LSD":
        if len(original_bitmap.shape) == 3:
            detection_input_image = cv2.cvtColor(original_bitmap, cv2.COLOR_RGB2GRAY)
        else:  # Already grayscale
            detection_input_image = original_bitmap
        # Optionally apply blur to original grayscale for LSD if Canny wasn't run
        if config['preprocessing'].get('enable_gaussian_blur', False) \
                and not config['preprocessing'].get('enable_canny', True):
            ksize = tuple(config['preprocessing']['gaussian_kernel_size'])
            sigma_x = config['preprocessing']['gaussian_sigma_x']
            detection_input_image = cv2.GaussianBlur(detection_input_image, ksize, sigma_x)
        if DEBUG_VERBOSE: display_image("Input for LSD", detection_input_image,
                                        {"global": {"show_intermediate_steps": True},
                                         "visualization": {"wait_key_delay": 1}})

    lines = ALGORITHM_FUNCTIONS[algo_name](detection_input_image, algo_params)
    print(f"检测到 {len(lines) if lines is not None else 0} 条线。")

    # 4. 可视化结果
    result_image = draw_lines_on_image(
        original_bitmap,  # Draw on the original color bitmap
        lines,
        tuple(config['visualization']['line_color']),
        config['visualization']['line_thickness']
    )
    display_image("Final Result", result_image, config)

    # Keep final window open if wait_key_delay is 0
    if config['visualization'].get('wait_key_delay', 0) == 0 and config['global'].get('show_intermediate_steps',
                                                                                      False) is False:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用不同算法对点云切片位图进行直线检测。")
    parser.add_argument(
        "--config",
        type=str,
        default=str(current_dir / "configs" / "default_params.json"),  # Default config
        help="指向参数配置文件的路径 (JSON格式)。"
    )
    # Optional: Add arguments to override specific config values, e.g., --algorithm, --slice_index
    # parser.add_argument("--algorithm", type=str, help="Override detection algorithm from config.")
    # parser.add_argument("--slice_index", type=int, help="Override slice index from config.")
    # parser.add_argument("--batch_dir", type=str, help="Override export_batch_dir from config.")

    args = parser.parse_args()
    if not Path(args.config).exists():
        print(f"错误: 配置文件 '{args.config}' 未找到。请确保路径正确，或创建一个。")
        print("将尝试使用默认路径下的 'default_params.json' (如果存在)。")
        default_config_path = current_dir / "configs" / "default_params.json"
        args.config = str(default_config_path)
    main(args.config)
    # Clean up any remaining OpenCV windows if script ends without waitKey(0) in main
    cv2.destroyAllWindows()
