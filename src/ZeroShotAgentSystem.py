# 导入必要模块
from langchain.agents import ReActAgent, ZeroShotAgent, AgentExecutor
from langchain_openai.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings


# Zero-shot Agent定义
class ZeroShotAgentSystem:
    def __init__(self):
        """初始化Zero-shot Agent，适用于不带上下文的直接问答。"""
        self.llm = OpenAI()

    def create_zero_shot_prompt(self, query):
        """
        为Zero-shot生成Prompt。
        :param query: 用户的查询内容
        :return: 生成的Prompt字符串
        """
        prompt = PromptTemplate(
            template="请基于常识回答以下问题:\n问题: {query}\n",
            input_variables=["query"]
        )
        return prompt.format(query=query)

    def generate_response(self, query):
        """
        使用Zero-shot Agent生成回答。
        :param query: 用户的查询内容
        :return: Zero-shot Agent生成的回答
        """
        prompt = self.create_zero_shot_prompt(query)
        response = self.llm(prompt)
        print("\nZero-shot Agent回答:\n", response)
        return response