# 导入必要模块

from langchain_community.callbacks import FileCallbackHandler
from typing import Any, Dict


# 任务链调试与监控的回调函数
class DebugCallbackHandler(FileCallbackHandler):
    def __init__(self, log_file="chain_log.txt"):
        """
        初始化调试回调处理程序。
        :param log_file: 日志文件的路径
        """
        self.log_file = log_file
        super().__init__(log_file=self.log_file)

    def on_chain_start(self, chain: Any, inputs: Dict[str, Any]):
        """记录任务链开始的事件"""
        with open(self.log_file, "a") as f:
            f.write(f"\n启动任务链: {chain.__class__.__name__}\n输入: {inputs}\n")

    def on_chain_end(self, outputs: Dict[str, Any]):
        """记录任务链结束的事件"""
        with open(self.log_file, "a") as f:
            f.write(f"输出: {outputs}\n结束任务链。\n")

    def on_chain_error(self, error: Exception):
        """记录任务链中的错误"""
        with open(self.log_file, "a") as f:
            f.write(f"任务链发生错误: {str(error)}\n")