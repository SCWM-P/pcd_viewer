import pyvista as pv


class VisualizationManager:
    """管理点云可视化效果"""

    @staticmethod
    def display_point_cloud(plotter, point_cloud, use_colors=True, point_size=2, render_mode="Points"):
        """
        在plotter中显示点云

        Args:
            plotter: pyvistaqt中的QtInteractor实例
            point_cloud: 要显示的点云数据
            use_colors (bool): 是否使用原始颜色
            point_size (int): 点的大小
            render_mode (str): 渲染模式，"Points"或"Mesh"

        Returns:
            actor: 添加到plotter的actor
        """
        if point_cloud is None or len(point_cloud.points) == 0:
            return None

        plotter.clear()  # 清空之前的内容

        actor = None
        if render_mode == "Mesh" and len(point_cloud.points) > 3:
            try:
                # 尝试创建表面
                surf = point_cloud.delaunay_2d()
                if 'colors' in point_cloud.point_data and use_colors:
                    actor = plotter.add_mesh(surf, scalars='colors', rgb=True)
                else:
                    actor = plotter.add_mesh(surf, color="#3498db")
            except Exception:
                # 如果创建表面失败，回退到点模式
                if 'colors' in point_cloud.point_data and use_colors:
                    actor = plotter.add_mesh(point_cloud, scalars='colors', rgb=True, point_size=point_size)
                else:
                    actor = plotter.add_mesh(point_cloud, color="#3498db", point_size=point_size)
        else:
            # 点渲染模式
            if 'colors' in point_cloud.point_data and use_colors:
                actor = plotter.add_mesh(point_cloud, scalars='colors', rgb=True, point_size=point_size)
            else:
                actor = plotter.add_mesh(point_cloud, color="#3498db", point_size=point_size)

        plotter.reset_camera_clipping_range()
        return actor

    @staticmethod
    def save_screenshot(plotter, file_path, settings):
        """
        保存plotter的截图

        Args:
            plotter: pyvistaqt中的QtInteractor实例
            file_path (str): 保存文件路径
            settings (dict): 截图设置

        Returns:
            bool: 是否保存成功
        """
        try:
            # 获取当前坐标轴状态
            original_axes_visible = True

            # 根据设置显示或隐藏坐标轴
            if settings['show_axis']:
                plotter.show_axes()
            else:
                plotter.hide_axes()
                original_axes_visible = False

            # 根据分辨率设置进行保存
            if settings['resolution'] == "current":
                # PyVista 0.44.2 不支持quality参数
                plotter.screenshot(
                    filename=file_path,
                    transparent_background=False
                )
            else:
                # 使用指定分辨率保存
                width, height = settings['resolution']
                if settings['keep_ratio']:
                    # 计算合适的高度或宽度以保持横纵比
                    current_width, current_height = plotter.window_size
                    aspect_ratio = current_width / current_height
                    if width / height > aspect_ratio:
                        width = int(height * aspect_ratio)
                    else:
                        height = int(width / aspect_ratio)

                # PyVista 0.44.2 不支持quality参数
                plotter.screenshot(
                    filename=file_path,
                    window_size=(width, height),
                    transparent_background=False
                )

            # 恢复原始坐标轴状态
            if original_axes_visible:
                plotter.show_axes()
            else:
                plotter.hide_axes()

            return True
        except Exception as e:
            print(f"保存截图失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False