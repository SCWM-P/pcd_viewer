import sys
from PyQt6.QtWidgets import QApplication
from .main_window import PCDViewerWindow


def exception_hook(exctype, value, tb):
    """全局异常处理函数，打印详细错误信息"""
    print('=' * 40)
    print("未捕获异常:")
    import traceback
    traceback.print_exception(exctype, value, tb)
    print('=' * 40)
    sys.__excepthook__(exctype, value, tb)


def main():
    app = QApplication(sys.argv)
    window = PCDViewerWindow()
    window.show()
    sys.exit(app.exec())


sys.excepthook = exception_hook

if __name__ == "__main__":
    main()
