
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai.llms import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from typing import List



# 1. 定义模型
class Answer(BaseModel):
    question: str = Field(description="用户提出的问题")
    answer: str = Field(description="对用户问题的简洁回答")
    details: str = Field(description="回答的详细解释")

# 2. 初始化基础解析器
base_parser = PydanticOutputParser(pydantic_object=Answer)

# 3. 初始化LLM（延迟创建：新版 langchain-openai 会在构造时校验 API 凭证，避免导入即报错）
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-3.5-turbo")
    return _llm


# 4. 自动修复解析器（新版正确写法，延迟创建）
_output_parser = None


def _get_output_parser():
    global _output_parser
    if _output_parser is None:
        _output_parser = OutputFixingParser.from_llm(parser=base_parser, llm=_get_llm())
    return _output_parser


class EnterpriseQASystem:
    def __init__(self, document_directory: str, index_path="faiss_index"):
        """
        初始化企业问答系统，包括文档存储、MMR搜索和解析器。
        :param document_directory: 企业文档目录
        :param index_path: FAISS向量存储路径
        """
        # 文档加载和FAISS存储初始化
        print("加载文档并创建向量存储...")
        self.embeddings = OpenAIEmbeddings()
        self.document_loader = TextLoader(document_directory)
        self.vector_store = self.create_vector_store()

        # 初始化模型
        self.llm = OpenAI()
        self.index_path = index_path

    def create_vector_store(self):
        """创建FAISS向量存储，用于MMR搜索"""
        documents = self.document_loader.load()
        vector_store = FAISS.from_documents(documents, self.embeddings)
        vector_store.save_local(self.index_path)
        return vector_store

    def mmr_search(self, query: str, top_k: int = 3) -> List[str]:
        """
        使用MMR搜索进行文档检索，减少重复内容并提升结果的多样性。
        :param query: 用户查询
        :param top_k: 返回的文档数量
        :return: 相似文档内容列表
        """
        query_embedding = self.embeddings.embed(query)
        similar_docs = self.vector_store.maximal_marginal_relevance_search_by_vector(query_embedding, k=top_k)

        # 打印搜索结果
        print("\nMMR搜索结果:")
        for i, doc in enumerate(similar_docs):
            print(f"文档{i + 1}:\n{doc.page_content[:200]}...\n")

        return [doc.page_content for doc in similar_docs]

    def generate_answer(self, query: str) -> Answer:
        """
        生成企业问答的回答，包含简洁答案和详细解释，使用自动修复解析器进行格式化。
        :param query: 用户的企业问答内容
        :return: Answer对象，包含答案和详细信息
        """
        # 使用MMR搜索找到相关文档
        relevant_docs = self.mmr_search(query)
        combined_docs = "\n".join(relevant_docs)

        # Prompt模板定义
        prompt_template = PromptTemplate(
            template=(
                "请回答以下问题，并提供简洁的答案和详细解释：\n"
                "问题: {question}\n"
                "相关内容:\n{context}\n"
                "格式:\n{format_instructions}\n"
                "回答："
            ),
            input_variables=["question", "context"],
            partial_variables={"format_instructions": _get_output_parser().get_format_instructions()}
        )

        prompt = prompt_template.format(question=query, context=combined_docs)

        # 生成响应并使用自动修复解析器
        raw_response = self.llm(prompt)
        print("\n模型原始输出:\n", raw_response)

        # 使用解析器格式化输出
        structured_answer = _get_output_parser().parse(raw_response)

        # 输出最终结果
        print("\n解析后的结构化结果:")
        print(f"问题: {structured_answer.question}")
        print(f"简洁回答: {structured_answer.answer}")
        print(f"详细解释: {structured_answer.details}")

        return structured_answer

    def validate_output(self, answer: Answer) -> bool:
        """
        输出校验，确保结构化结果符合企业问答要求。
        :param answer: Answer对象
        :return: 校验结果，符合要求返回True
        """
        is_valid = bool(answer.question and answer.answer and answer.details)
        print("\n输出校验结果:", "通过" if is_valid else "不通过")
        return is_valid


# 测试集成运行
if __name__ == "__main__":
    # 初始化企业问答系统
    document_directory = "./data/enterprise_docs"
    qa_system = EnterpriseQASystem(document_directory=document_directory)

    # 测试企业问答用例
    query = "请简要说明公司的年度增长策略"

    # 生成回答
    answer = qa_system.generate_answer(query)

    # 校验输出
    if qa_system.validate_output(answer):
        print("最终答案通过校验")
    else:
        print("最终答案未通过校验")
