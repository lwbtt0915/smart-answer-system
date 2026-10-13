
# 导入必要模块
from langchain_classic.chains import SequentialChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai.llms import OpenAI
from typing import Any, Dict
import datetime
import os

# 自定义回调处理程序
class CustomCallbackHandler(BaseCallbackHandler):
    def __init__(self, log_dir="logs"):
        """
        初始化自定义回调处理程序。
        :param log_dir: 日志文件存储目录
        """
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "chain_execution_log.txt")

    def _log_event(self, event: str):
        """将事件记录到日志文件中"""
        with open(self.log_file, "a") as f:
            f.write(f"{datetime.datetime.now()} - {event}\n")

    def on_chain_start(self, serialized: Any, inputs: Dict[str, Any], **kwargs: Any):
        """记录任务链启动事件（兼容新版回调签名）"""
        chain_name = serialized.get("name", "未知任务链") if isinstance(serialized, dict) else serialized
        self._log_event(f"启动任务链: {chain_name}\n输入: {inputs}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any):
        """记录任务链结束事件"""
        self._log_event(f"任务链结束\n输出: {outputs}\n")

    def on_chain_error(self, error: Exception, **kwargs: Any):
        """记录任务链错误事件"""
        self._log_event(f"任务链发生错误: {str(error)}\n")


# 任务链设计与调试
class DebuggableTaskChain:
    def __init__(self, callback_handler: CustomCallbackHandler):
        """
        初始化可调试的任务链，集成自定义回调处理程序。
        :param callback_handler: 自定义回调处理程序实例
        """
        self.llm = OpenAI()
        self.callback_handler = callback_handler

    def create_chain(self):
        """构建任务链，用于企业问答系统的分步执行"""

        # 子任务1：问题解析
        question_prompt = PromptTemplate(
            template="请分析以下问题并确定其关键主题。\n问题: {query}",
            input_variables=["query"]
        )
        question_chain = LLMChain(
            llm=self.llm,
            prompt=question_prompt,
            output_key="theme",
            callbacks=[self.callback_handler]
        )

        # 子任务2：文档搜索
        search_prompt = PromptTemplate(
            template="根据主题'{theme}'在企业知识库中搜索信息并总结关键点。",
            input_variables=["theme"]
        )
        search_chain = LLMChain(
            llm=self.llm,
            prompt=search_prompt,
            output_key="summary",
            callbacks=[self.callback_handler]
        )

        # 子任务3：格式化回答
        format_prompt = PromptTemplate(
            template="请根据问题 '{query}' 和找到的内容进行回答，并提供详细说明。\n内容: {summary}",
            input_variables=["query", "summary"]
        )
        format_chain = LLMChain(
            llm=self.llm,
            prompt=format_prompt,
            output_key="final_answer",
            callbacks=[self.callback_handler]
        )

        # 创建任务链
        task_chain = SequentialChain(
            chains=[question_chain, search_chain, format_chain],
            input_variables=["query"],
            output_variables=["final_answer"],
            callbacks=[self.callback_handler]
        )

        return task_chain

    def execute_chain(self, query: str):
        """
        执行任务链并记录执行过程。
        :param query: 用户的查询内容
        :return: 最终答案
        """
        task_chain = self.create_chain()
        self.callback_handler.on_chain_start("企业问答任务链", {"query": query})

        try:
            result = task_chain({"query": query})
            self.callback_handler.on_chain_end(result)
            print("\n任务链执行结果:", result["final_answer"])
            return result["final_answer"]
        except Exception as e:
            self.callback_handler.on_chain_error(e)
            print("任务链执行中遇到错误:", str(e))
            return None


# 模块运行测试
if __name__ == "__main__":
    # 初始化自定义回调处理程序
    callback_handler = CustomCallbackHandler()

    # 初始化任务链模块
    debuggable_task_chain = DebuggableTaskChain(callback_handler)

    # 测试企业问答用例
    query = "公司的年度增长策略是什么？"
    final_answer = debuggable_task_chain.execute_chain(query=query)