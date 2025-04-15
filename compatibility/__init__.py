# pcd_viewer/compatibility/__init__.py
import sys
import os
import platform

def apply_linux_hybrid_graphics_workaround():
    """
    检测Linux混合显卡环境并设置必要的环境变量以使用NVIDIA GPU进行渲染
    """
    # print("Checking system compatibility...") # 添加打印信息
    if sys.platform == 'linux':
        try:
            # 检查OpenGL Vendor是否不是NVIDIA
            opengl_vendor = ""
            try:
                # 尝试运行 glxinfo 获取当前 vendor
                import subprocess
                result = subprocess.run(['glxinfo'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if "OpenGL vendor string:" in line:
                            opengl_vendor = line.split(":")[1].strip()
                            print(f"Detected OpenGL vendor: {opengl_vendor}")
                            break
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
                print(f"Could not run glxinfo to check OpenGL vendor: {e}")
                # 如果无法检查，保守起见，我们假设可能需要设置
                opengl_vendor = "Unknown" # 或者保持为空 ""

            # 检查是否存在 nvidia-smi，表明NVIDIA驱动可能已安装
            nvidia_likely_present = False
            try:
                subprocess.run(['nvidia-smi'], capture_output=True, timeout=2)
                nvidia_likely_present = True
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                 print("nvidia-smi not found or timed out.")
                 pass # nvidia-smi 不存在，可能不是NVIDIA环境

            # 只有当检测到系统是Linux，OpenGL Vendor不是NVIDIA (或无法检测)，
            # 并且NVIDIA驱动似乎存在时，才应用环境变量
            if nvidia_likely_present and "nvidia" not in opengl_vendor.lower():
                print(
                    "Applying Linux hybrid graphics workaround (PRIME Render Offload)... \n"
                    "Set __NV_PRIME_RENDER_OFFLOAD=1 and"
                    "Set __GLX_VENDOR_LIBRARY_NAME=nvidia in order to use NVIDIA"
                )
                # 检查环境变量是否已设置，避免重复设置
                if '__NV_PRIME_RENDER_OFFLOAD' not in os.environ:
                    os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'

                if '__GLX_VENDOR_LIBRARY_NAME' not in os.environ:
                    os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

        except Exception as e:
            print(f"An error occurred during compatibility check: {e}")

# 在模块导入时执行兼容性检查和应用
apply_linux_hybrid_graphics_workaround()