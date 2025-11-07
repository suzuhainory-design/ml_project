"""
日志工具类
用于统一管理项目中的日志记录
"""
import logging
import os
from datetime import datetime


class LogUtil:
    """日志工具类，提供统一的日志记录功能"""
    
    def __init__(self, log_dir='../log', log_name=None):
        """
        初始化日志工具
        
        Args:
            log_dir: 日志文件存储目录
            log_name: 日志文件名（不含扩展名），默认使用时间戳
        """
        self.log_dir = log_dir
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名
        if log_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_name = f'log_{timestamp}'
        
        self.log_file = os.path.join(log_dir, f'{log_name}.log')
        
        # 配置日志记录器
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 创建文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, message):
        """记录DEBUG级别日志"""
        self.logger.debug(message)
    
    def info(self, message):
        """记录INFO级别日志"""
        self.logger.info(message)
    
    def warning(self, message):
        """记录WARNING级别日志"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录ERROR级别日志"""
        self.logger.error(message)
    
    def critical(self, message):
        """记录CRITICAL级别日志"""
        self.logger.critical(message)
    
    def get_log_file(self):
        """获取日志文件路径"""
        return self.log_file
    
    def close(self):
        """关闭日志记录器"""
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
