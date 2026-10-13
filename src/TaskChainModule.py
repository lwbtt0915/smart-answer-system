# 导入必要模块
from langchain_community.chains import SequentialChain, LLMChain
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import FileCallbackHandler

import DebugCallbackHandler


# 任务链模块
class TaskChainModule:
    def __init__(self):
        """
        初始化任务链模块，包含多层任务链和回调调试工具。
        """
        self.llm = OpenAI()
        self.debug_handler = DebugCallbackHandler.DebugCallbackHandler()

    def create_task_chain(self):
        """构建企业级智能问答的多层任务链"""

        # 子任务1：解析用户问题并确定任务类型
        question_template = PromptTemplate(
            template="请分析以下问题，并指出问题的主题和需要的详细信息。\n问题: {query}",
            input_variables=["query"]
        )
        question_chain = LLMChain(llm=self.llm, prompt=question_template, callback_handler=self.debug_handler)

        # 子任务2：根据主题搜索相似内容
        search_template = PromptTemplate(
            template="根据主题'{theme}'，从企业知识库中寻找相关信息，并提供简要概述。",
            input_variables=["theme"]
        )
        search_chain = LLMChain(llm=self.llm, prompt=search_template, callback_handler=self.debug_handler)

        # 子任务3：生成最终的回答格式
        answer_template = PromptTemplate(
            template="将以下信息格式化为企业问答回答:\n问题: {query}\n主题: {theme}\n内容概述: {summary}",
            input_variables=["query", "theme", "summary"]
        )
        answer_chain = LLMChain(llm=self.llm, prompt=answer_template, callback_handler=self.debug_handler)

        # 创建多层任务链
        task_chain = SequentialChain(
            chains=[question_chain, search_chain, answer_chain],
            input_variables=["query"],
            output_variables=["final_answer"],
            callback_handler=self.debug_handler
        )
        return task_chain

    def execute_task_chain(self, query: str):
        """
        执行多层任务链，返回处理后的结果。
        :param query: 用户提出的问题
        :return: 最终回答
        """
        task_chain = self.create_task_chain()
        try:
            result = task_chain({"query": query})
            print("\n任务链执行结果:", result["final_answer"])
            return result["final_answer"]
        except Exception as e:
            print("任务链执行中遇到错误:", str(e))
            return None


# 模块运行测试
if __name__ == "__main__":
    # 初始化任务链模块
    task_module = TaskChainModule()

    # 测试企业问答用例
    query = "请解释公司的年度预算规划流程"
    final_answer = task_module.execute_task_chain(query)