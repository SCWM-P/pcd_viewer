class StylesheetManager:
    """管理应用程序样式表"""

    @staticmethod
    def get_light_theme():
        """返回亮色主题的样式表"""
        return """
        QWidget {
            background-color: white;
            color: #333333;
        }

        QGroupBox {
            font-weight: bold;
            border: 1px solid #dddddd;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            background-color: white;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            background-color: white;
        }

        QPushButton {
            padding: 5px;
            border-radius: 4px;
            background-color: white;
            border: 1px solid #dddddd;
        }

        QPushButton:hover {
            background-color: #f5f5f5;
        }

        QSlider::groove:horizontal {
            height: 8px;
            background: #f0f0f0;
            border-radius: 4px;
        }

        QSlider::handle:horizontal {
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
            background: white;
            border: 1px solid #bbbbbb;
        }

        QTabWidget::pane {
            border: 1px solid #dddddd;
            border-radius: 5px;
            top: -1px;
            background-color: white;
        }

        QTabBar::tab {
            padding: 5px 10px;
            border: 1px solid #dddddd;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            background-color: #f8f8f8;
        }

        QTabBar::tab:selected {
            background: white;
            border-bottom: 1px solid white;
        }

        QToolBar {
            border: none;
            spacing: 3px;
            background-color: white;
            border-bottom: 1px solid #eeeeee;
        }

        QStatusBar {
            background-color: white;
            border-top: 1px solid #eeeeee;
        }
        """