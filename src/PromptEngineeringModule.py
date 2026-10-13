
# 导入必要模块
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.output_parsers import PydanticOutputParser

try:
    from src import EnterpriseQASystem
except ImportError:
    import EnterpriseQASystem

Answer = EnterpriseQASystem.Answer


class PromptEngineeringModule:
    def __init__(self):
        """
        初始化提示词工程模块，包括Prompt模板设计和输出解析器。
        """
        # 初始化OpenAI模型
        self.llm = OpenAI()

        # 自定义输出解析器
        self.parser = PydanticOutputParser(pydantic_object=Answer)

    def create_prompt_template(self, query: str) -> str:
        """
        使用Prompt模板生成结构化的模型输入。
        :param query: 用户的查询
        :return: 完整的Prompt文本
        """
        prompt_template = PromptTemplate(
            template=(
                "请回答以下问题，并提供简洁的答案和详细解释：\n"
                "问题：{question}\n\n"
                "格式如下：\n"
                "{format_instructions}\n\n"
                "回答："
            ),
            input_variables=["question"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        return prompt_template.format(question=query)

    def parse_response(self, response: str) -> Answer:
        """
        使用自定义输出解析器解析模型的生成响应。
        :param response: 模型生成的文本响应
        :return: 解析后的Answer对象
        """
        return self.parser.parse(response)

    def generate_answer(self, query: str) -> Answer:
        """
        生成企业问答的回答，包含简洁答案和详细解释。
        :param query: 用户的企业问答内容
        :return: Answer对象，包含答案和详细信息
        """
        # 创建Prompt模板
        prompt = self.create_prompt_template(query)

        # 模型生成响应
        response = self.llm(prompt)
        print("\n模型原始输出:\n", response)

        # 解析响应为结构化数据
        parsed_answer = self.parse_response(response)

        # 输出解析结果
        print("\n解析后的结果:")
        print(f"问题: {parsed_answer.question}")
        print(f"简洁回答: {parsed_answer.answer}")
        print(f"详细解释: {parsed_answer.details}")

        return parsed_answer


# 模块运行测试
if __name__ == "__main__":
    # 初始化提示词工程模块
    prompt_engineering = PromptEngineeringModule()

    # 测试企业问答用例
    query = "请简要说明公司年终绩效考核流程"
    parsed_answer = prompt_engineering.generate_answer(query)
