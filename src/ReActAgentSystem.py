
# 导入必要模块
from langchain_openai.llms import OpenAI
from langchain_core.prompts import PromptTemplate
# ReAct Agent定义
class ReActAgentSystem:
    def __init__(self, document_storage):
        """
        初始化ReAct Agent，基于企业文档库执行问题解答。
        :param document_storage: 文档存储模块
        """
        self.document_storage = document_storage
        self.llm = OpenAI()

    def create_react_prompt(self, query):
        """
        根据查询生成ReAct Prompt。
        :param query: 用户的查询内容
        :return: 生成的Prompt字符串
        """
        react_prompt = PromptTemplate(
            template="请回答以下问题并给出基于企业文档的详细信息:\n问题: {query}\n",
            input_variables=["query"]
        )
        return react_prompt.format(query=query)

    def generate_response(self, query):
        """
        使用ReAct Agent生成回答。
        :param query: 用户的查询内容
        :return: ReAct Agent生成的回答
        """
        prompt = self.create_react_prompt(query)
        response = self.llm(prompt)
        print("\nReAct Agent回答:\n", response)
        return response