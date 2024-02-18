import os
import logging

class Log():
    def __init__(self, log_dir, model_name, split, level="DEBUG"):
        # 日志器对象
        self.log = logging.getLogger(name=model_name)
        self.log.setLevel(level)
        self.dir = os.path.join(log_dir, f"{split}_logger.log")

    def console_handle(self, level="DEBUG"):
        """控制台处理器"""
        console_handle = logging.StreamHandler()
        console_handle.setLevel(level)
        # 处理器添加格式器
        console_handle.setFormatter(self.get_formatter()[0])

        return console_handle

    def file_handle(self, level="DEBUG"):
        """文件处理器"""
        
        file_handle = logging.FileHandler(filename=self.dir, mode="a", encoding="utf-8")
        file_handle.setLevel(level)
        # 处理器添加格式器
        file_handle.setFormatter(self.get_formatter()[1])

        return file_handle

    def get_formatter(self):
        """格式器"""

        # 定义输出格式
        fmt = "%(asctime)s--->%(name)s--->%(levelname)s--->%(message)s"  
        fmt_2 = "%(asctime)s--->%(name)s--->%(levelname)s--->%(message)s"
        get_fmt = logging.Formatter(fmt=fmt)  # 创建格式器
        file_fmt = logging.Formatter(fmt=fmt_2)  # 创建格式器

        return get_fmt, file_fmt

    def get_log(self):
        # # 日志器添加控制台处理器
        self.log.addHandler(self.console_handle())
        # 日志器添加文件处理器
        self.log.addHandler(self.file_handle())

        return self.log
